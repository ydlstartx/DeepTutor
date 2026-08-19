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

from deeptutor.api.routers.unified_ws import (
    _forward_subscription,
    _forward_with_content_batching,
    _Inbox,
)
from deeptutor.core.stream import StreamEvent, StreamEventType
from deeptutor.services.session.sqlite_store import SQLiteSessionStore
from deeptutor.services.session.turn_runtime import (
    TurnRuntimeManager,
    _LiveSubscriber,
    _TurnExecution,
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
            event = StreamEvent(type=StreamEventType.CONTENT, source="chat", content=f"tok{i}")
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


class TestForwardSubscription:
    """Integration-level: the WS _forward glue around batching + healing.

    These drive ``_forward_subscription`` (the module-level function the WS
    endpoint closures delegate to) end to end, covering the reconnect
    decision that a closure-only bug previously broke (UnboundLocalError on
    the no-DONE path left the socket open forever).
    """

    async def test_no_done_ends_stream_triggers_on_interrupted(self):
        async def events():
            yield await _content("abc", 1)
            # No DONE: subscription was dropped / turn vanished mid-stream.

        interrupted: list[bool] = []

        async def on_interrupted() -> None:
            interrupted.append(True)

        await _forward_subscription(events, lambda e: asyncio.sleep(0), on_interrupted)
        assert interrupted == [True], "no-DONE end must trigger the reconnect path"

    async def test_done_does_not_trigger_on_interrupted(self):
        async def events():
            yield await _content("abc", 1)
            yield {"type": "done", "source": "chat", "seq": 2}

        interrupted: list[bool] = []

        async def on_interrupted() -> None:
            interrupted.append(True)

        await _forward_subscription(events, lambda e: asyncio.sleep(0), on_interrupted)
        assert interrupted == [], "normal DONE end must not close the socket"

    async def test_slow_consumer_overflow_chain_reaches_on_interrupted(self, tmp_path):
        """End-to-end backpressure: slow client -> inbox full -> pump stalls
        -> runtime subscriber queue overflows -> subscription terminated
        without DONE -> on_interrupted invoked (the WS layer closes the
        socket, the client reconnects and resumes)."""
        store = SQLiteSessionStore(tmp_path / "chat_history.db")
        runtime = TurnRuntimeManager(store)
        session = await runtime.store.create_session("overflow")
        turn = await runtime.store.create_turn(session["id"], "chat")
        execution = _TurnExecution(
            turn_id=turn["id"], session_id=session["id"], capability="chat", payload={}
        )
        runtime._executions[turn["id"]] = execution

        async def events():
            async for event in runtime.subscribe_turn(turn["id"], after_seq=0):
                yield event

        send_gate = asyncio.Event()
        send_started = asyncio.Event()
        forwarded: list[dict[str, Any]] = []

        async def slow_send(event: dict[str, Any]) -> None:
            # A stalled client: never consumes until the test releases it.
            send_started.set()
            await send_gate.wait()
            forwarded.append(event)

        interrupted: list[bool] = []

        async def on_interrupted() -> None:
            interrupted.append(True)

        forward_task = asyncio.create_task(_forward_subscription(events, slow_send, on_interrupted))

        # Synchronize on observable milestones with wall-clock budgets. The
        # subscription path does real SQLite reads (backlog fetch, turn fetch),
        # so bare sleep(0) loops don't give them time to run — which is what
        # made the original fixed-1600-events version race: depending on
        # scheduling it either never overflowed (hang on await forward_task)
        # or overflowed before a single frame reached the WS layer.
        async def wait_until(pred, what: str, timeout: float = 5.0) -> None:
            deadline = asyncio.get_running_loop().time() + timeout
            while not pred():
                if asyncio.get_running_loop().time() > deadline:
                    pytest.fail(f"timed out waiting for {what}")
                await asyncio.sleep(0.01)

        # 1. The overflow can only fire after the runtime subscriber exists.
        await wait_until(
            lambda: bool(runtime._executions[turn["id"]].subscribers),
            "subscriber registration",
        )

        # 2. Get the pipeline flowing: publish until one frame is in-flight
        # (blocked on the gate), proving events reached the WS layer — the
        # drain below then provably delivers something to the client.
        async def publish(i: int) -> None:
            await runtime._publish_live_event(
                execution,
                StreamEvent(type=StreamEventType.CONTENT, source="chat", content=f"tok{i}"),
            )

        for i in range(1000):
            if send_started.is_set():
                break
            await publish(i)
            await asyncio.sleep(0.005)
        else:
            pytest.fail("pipeline never delivered a frame to the stalled client")

        # 3. Overflow the chain: publish until the runtime terminates the
        # subscriber (visible immediately via execution.subscribers), not a
        # fixed count — the batching layer absorbs a scheduling-dependent
        # number of in-flight events on top of inbox (1024) + subscriber
        # queue (500). The cap turns "never overflows" into a clear failure
        # instead of a stuck suite.
        for j in range(10000):
            if not runtime._executions[turn["id"]].subscribers:
                break
            await publish(1000 + j)
            await asyncio.sleep(0)
        else:
            pytest.fail("overflow never terminated the slow subscriber")

        # Release the client, drain, and confirm the reconnect path fired.
        send_gate.set()
        await asyncio.wait_for(forward_task, timeout=10)
        assert interrupted == [True], "overflow must end in the reconnect path"
        assert len(forwarded) > 0, "client received frames before the stall"
        # 订阅者已从 execution 移除，turn 仍在运行（未合成假 done）
        assert runtime._executions[turn["id"]].subscribers == []
