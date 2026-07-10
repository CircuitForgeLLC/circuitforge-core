"""SignalBus — generic SSE event publisher for real-time signal streams.

MIT licensed.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator

from fastapi import Request
from fastapi.responses import StreamingResponse

from circuitforge_core.signal_bus.models import SignalEvent

logger = logging.getLogger(__name__)

_DEFAULT_KEEPALIVE = 15.0  # seconds between SSE keepalive comments


def _enqueue_drop_oldest(q: asyncio.Queue, item: str) -> None:
    """Put item on queue; if full, drop the oldest entry to make room.

    Designed to run on the event loop thread via call_soon_threadsafe.
    """
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass  # another thread won the race; drop the new item


class SignalBus:
    """Publish-subscribe bus for real-time signal events over SSE.

    One bus instance per service. Producers call publish() (sync-safe);
    consumers mount subscribe() as a FastAPI endpoint.

    Args:
        queue_size: Max buffered events per subscriber before oldest is dropped.
    """

    def __init__(self, queue_size: int = 100, keepalive_interval: float = _DEFAULT_KEEPALIVE) -> None:
        self._queue_size = queue_size
        self._keepalive_interval = keepalive_interval
        # int(id(queue)) → queue; guarded by _lock for publish() from sync threads
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._lock = threading.Lock()
        # Captured on first subscribe() call; set from the FastAPI event loop
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def publish(self, event: SignalEvent) -> None:
        """Publish an event to all active subscribers.

        Thread-safe — may be called from sync threads (OpenCV, Meshtastic,
        PyPubSub callbacks, etc.). No-op when no subscribers are connected.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        sse = event.to_sse()
        with self._lock:
            subs = list(self._subscribers.values())

        for q in subs:
            loop.call_soon_threadsafe(_enqueue_drop_oldest, q, sse)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def subscribe(self, request: Request) -> StreamingResponse:
        """Return an SSE StreamingResponse for a FastAPI endpoint.

        Each caller gets an independent queue. The subscriber is removed
        automatically when the client disconnects.

        Usage::

            @app.get("/events")
            async def events(request: Request):
                return bus.subscribe(request)
        """
        loop = asyncio.get_event_loop()
        self._loop = loop

        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        sub_id = id(q)
        with self._lock:
            self._subscribers[sub_id] = q

        logger.debug("signal_bus: subscriber %d connected (%d total)", sub_id, len(self._subscribers))

        async def _generate() -> AsyncIterator[str]:
            # Starlette calls aclose() on this generator when the client disconnects,
            # injecting GeneratorExit. Polling request.is_disconnected() is unreliable
            # in ASGI test transports and adds latency in production.
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=self._keepalive_interval)
                        yield chunk
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                with self._lock:
                    self._subscribers.pop(sub_id, None)
                logger.debug(
                    "signal_bus: subscriber %d disconnected (%d remaining)",
                    sub_id,
                    len(self._subscribers),
                )

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
