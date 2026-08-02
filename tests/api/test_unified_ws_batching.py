"""Tests for WS content-frame coalescing and slow-consumer handling.

Covers the two behaviours fixed after review:
1. ``_forward_with_content_batching`` must actually flush on the 40ms
   timeout and the 64-char threshold (not buffer until DONE), so slow
   streams stay visible and short answers don't land all at once.
2. A slow subscriber that overflows the bounded live queue is terminated
   (None sentinel + removal) instead of silently dropping frames — the
   client reconnects and resumes from its last seq, which heals the hole.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deeptutor.api.routers.unified_ws import _Inbox, _forward_with_content_batching
from deeptutor.core.stream import StreamEvent, StreamEventType
from deeptutor.services.session.turn_runtime import (
    _LiveSubscriber,
    _TurnExecution,
    TurnRuntimeManager,
)

pytestmark = pytest.mark.asyncio


async def _run_stream(events) -> tuple[bool, list[dict[str, Any]]]:
    """Feed ``events`` through a pump + inbox, exactly like the WS layer."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1024)

    async def pump() -> None:
        try:
            async for event in events:
                await queue.put(event)
        finally:
            await queue.put(None)

    pump_task = asyncio.create_task(pump())
    out: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        out.append(event)

    try:
        seen_done = await _forward_with_content_batching(_Inbox(queue), send)
    finally:
        pump_task.cancel()
        with pytest.MonkeyPatch.context() as _:  # keep linters quiet about ctx
            pass
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
    return seen_done, out


async def _content(text: str, seq: int, source: str = "chat") -> dict[str, Any]:
    return {"type": "content", "source": source, "stage": "responding", "content": text, "seq": seq}


class TestContentBatching:
    async def test_slow_stream_flushes_on_timeout(self):
        """A token every 0.2s must stream out on the 40ms timeout, not wait for DONE."""

        async def slow_stream():
            for i in range(5):
                yield await _content(f"tok{i}", i)
                await asyncio.sleep(0.2)
            yield {"type": "done", "source": "chat", "seq": 100}

        seen_done, out = await _run_stream(slow_stream())
        assert seen_done
        frames = [e for e in out if e["type"] == "content"]
        assert len(frames) >= 3, f"timeout flush should emit intermediate frames, got {len(frames)}"
        assert "".join(e["content"] for e in frames) == "tok0tok1tok2tok3tok4"

    async def test_fast_stream_flushes_at_char_threshold(self):
        """A fast stream coalesces into ~64-char frames, not one giant frame."""

        async def fast_stream():
            for i in range(200):
                yield await _content(f"t{i}", i + 1)
            yield {"type": "done", "source": "chat", "seq": 999}

        seen_done, out = await _run_stream(fast_stream())
        assert seen_done
        frames = [e for e in out if e["type"] == "content"]
        assert 3 <= len(frames) <= 15, f"expected ~64-char frames, got {len(frames)}"
        assert "".join(e["content"] for e in frames) == "".join(f"t{i}" for i in range(200))
        # 每帧保持最新 seq 以便 resume_from 重放
        assert frames[-1]["seq"] == 200

    async def test_no_done_returns_false_and_flushes_tail(self):
        """A dropped subscription (no DONE) returns False and ships the tail."""

        async def dropped_stream():
            yield await _content("abc", 1)

        seen_done, out = await _run_stream(dropped_stream())
        assert not seen_done
        assert out[-1]["content"] == "abc"

    async def test_non_content_events_preserve_order_and_flush(self):
        async def mixed_stream():
            for i in range(10):
                yield await _content(f"c{i}", i)
            yield {"type": "tool_start", "source": "rag", "seq": 100}
            for i in range(5):
                yield await _content(f"d{i}", 200 + i)

        _, out = await _run_stream(mixed_stream())
        assert [e["type"] for e in out] == ["content", "tool_start", "content"]


class TestSlowConsumerTermination:
    async def test_overflow_terminates_subscriber_instead_of_silent_drop(self):
        mgr = TurnRuntimeManager.__new__(TurnRuntimeManager)
        mgr._lock = asyncio.Lock()
        mgr._executions = {}
        execution = _TurnExecution(turn_id="t1", session_id="s1", capability="chat", payload={})
        mgr._executions["t1"] = execution

        slow_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
        slow_subscriber = _LiveSubscriber(queue=slow_queue)
        execution.subscribers.append(slow_subscriber)

        for i in range(10):
            event = StreamEvent(
                type=StreamEventType.CONTENT, source="chat", content=f"tok{i}"
            )
            await mgr._publish_live_event(execution, event)

        # 终止信号：最后一个条目是 None；订阅者立即从列表移除
        items = []
        while not slow_queue.empty():
            items.append(slow_queue.get_nowait())
        assert items[-1] is None, "subscriber must be terminated with the None sentinel"
        assert slow_subscriber not in execution.subscribers

        # 终止后不再投递任何事件（内容由客户端重连 + resume_from 从 store 补全）
        await mgr._publish_live_event(
            execution,
            StreamEvent(type=StreamEventType.CONTENT, source="chat", content="after"),
        )
        assert slow_queue.empty()

    async def test_healthy_subscriber_unaffected(self):
        mgr = TurnRuntimeManager.__new__(TurnRuntimeManager)
        mgr._lock = asyncio.Lock()
        mgr._executions = {}
        execution = _TurnExecution(turn_id="t2", session_id="s1", capability="chat", payload={})
        mgr._executions["t2"] = execution

        ok_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        ok_subscriber = _LiveSubscriber(queue=ok_queue)
        execution.subscribers.append(ok_subscriber)

        for i in range(3):
            await mgr._publish_live_event(
                execution,
                StreamEvent(type=StreamEventType.CONTENT, source="chat", content=f"x{i}"),
            )
        received = [ok_queue.get_nowait() for _ in range(3)]
        assert all(item is not None for item in received)
        assert ok_subscriber in execution.subscribers
