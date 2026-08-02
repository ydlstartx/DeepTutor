"""
Progress Broadcaster - Manages WebSocket broadcasting of knowledge base progress
"""

import asyncio
import logging
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ProgressBroadcaster:
    """Manages WebSocket broadcasting of knowledge base progress"""

    _instance: Optional["ProgressBroadcaster"] = None
    _connections: dict[str, set[WebSocket]] = {}  # kb_name -> Set[WebSocket]
    _lock = asyncio.Lock()
    # Coalescing state: while a send loop for a KB is in flight, later
    # broadcast() calls only replace the pending payload (latest wins), so
    # high-frequency progress bursts collapse into one WS write per loop.
    _sending: set[str] = set()
    _pending: dict[str, dict] = {}

    @classmethod
    def get_instance(cls) -> "ProgressBroadcaster":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self, kb_name: str, websocket: WebSocket):
        """Connect WebSocket to specified knowledge base"""
        async with self._lock:
            if kb_name not in self._connections:
                self._connections[kb_name] = set()
            self._connections[kb_name].add(websocket)
            logger.debug(
                f"Connected WebSocket for KB '{kb_name}' (total: {len(self._connections[kb_name])})"
            )

    async def disconnect(self, kb_name: str, websocket: WebSocket):
        """Disconnect WebSocket connection"""
        async with self._lock:
            if kb_name in self._connections:
                self._connections[kb_name].discard(websocket)
                if not self._connections[kb_name]:
                    del self._connections[kb_name]
                logger.debug(f"Disconnected WebSocket for KB '{kb_name}'")

    async def broadcast(self, kb_name: str, progress: dict):
        """Broadcast progress update to all WebSocket connections for specified knowledge base.

        Sends happen outside the lock and in parallel, so one slow or broken
        client can no longer stall every KB's progress channel.
        """
        async with self._lock:
            if kb_name not in self._connections:
                return
            if kb_name in self._sending:
                self._pending[kb_name] = progress
                return
            self._sending.add(kb_name)
            current = progress

        while True:
            dead = await self._send_to_all(kb_name, current)
            if dead:
                async with self._lock:
                    conns = self._connections.get(kb_name)
                    if conns:
                        for websocket in dead:
                            conns.discard(websocket)
                        if not conns:
                            del self._connections[kb_name]
            async with self._lock:
                current = self._pending.pop(kb_name, None)
                if current is None:
                    self._sending.discard(kb_name)
                    return

    async def _send_to_all(self, kb_name: str, progress: dict) -> list[WebSocket]:
        """Send one payload to a snapshot of the KB's connections, in parallel.

        Returns the websockets that failed so the caller can prune them.
        """
        async with self._lock:
            if kb_name not in self._connections:
                return []
            connections = list(self._connections[kb_name])

        async def _send(websocket: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(
                    websocket.send_json({"type": "progress", "data": progress}),
                    timeout=5.0,
                )
                return None
            except Exception as e:  # noqa: BLE001 — any ws error means drop it
                logger.debug(f"Error sending to WebSocket for KB '{kb_name}': {e}")
                return websocket

        results = await asyncio.gather(*(_send(ws) for ws in connections))
        return [ws for ws in results if ws is not None]

    def get_connection_count(self, kb_name: str) -> int:
        """Get connection count for specified knowledge base"""
        return len(self._connections.get(kb_name, set()))
