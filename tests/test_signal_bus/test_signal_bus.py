"""Tests for SignalBus and SignalEvent."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from circuitforge_core.signal_bus import SignalBus, SignalEvent


# ---------------------------------------------------------------------------
# SignalEvent
# ---------------------------------------------------------------------------

class TestSignalEvent:
    def test_sse_format(self) -> None:
        ev = SignalEvent(source="merlin", kind="gesture", payload={"name": "open_palm"}, timestamp="2026-01-01T00:00:00Z")
        sse = ev.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

    def test_sse_contains_all_fields(self) -> None:
        ev = SignalEvent(source="linnet", kind="tone", payload={"label": "sarcasm"}, timestamp="2026-01-01T00:00:00Z")
        data = json.loads(ev.to_sse()[len("data: "):].strip())
        assert data["source"] == "linnet"
        assert data["kind"] == "tone"
        assert data["payload"] == {"label": "sarcasm"}
        assert data["timestamp"] == "2026-01-01T00:00:00Z"

    def test_timestamp_auto_populated(self) -> None:
        ev = SignalEvent(source="s", kind="k", payload={})
        assert ev.timestamp  # not empty
        assert "T" in ev.timestamp  # ISO format

    def test_frozen(self) -> None:
        ev = SignalEvent(source="s", kind="k", payload={})
        with pytest.raises((AttributeError, TypeError)):
            ev.source = "other"  # type: ignore


# ---------------------------------------------------------------------------
# SignalBus — unit (direct queue inspection)
# ---------------------------------------------------------------------------

class TestSignalBusUnit:
    def test_publish_noop_before_subscribe(self) -> None:
        bus = SignalBus()
        ev = SignalEvent(source="s", kind="k", payload={})
        bus.publish(ev)  # must not raise

    def test_subscriber_count_zero_initially(self) -> None:
        bus = SignalBus()
        assert bus.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_enqueue_drop_oldest_on_overflow(self) -> None:
        from circuitforge_core.signal_bus.bus import _enqueue_drop_oldest
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        _enqueue_drop_oldest(q, "a")
        _enqueue_drop_oldest(q, "b")
        _enqueue_drop_oldest(q, "c")  # drops "a"
        assert q.qsize() == 2
        assert q.get_nowait() == "b"
        assert q.get_nowait() == "c"

    @pytest.mark.asyncio
    async def test_enqueue_drop_oldest_empty_queue(self) -> None:
        from circuitforge_core.signal_bus.bus import _enqueue_drop_oldest
        q: asyncio.Queue = asyncio.Queue(maxsize=5)
        _enqueue_drop_oldest(q, "x")
        assert q.get_nowait() == "x"


# ---------------------------------------------------------------------------
# SignalBus — integration via FastAPI TestClient
# ---------------------------------------------------------------------------

def _make_bus() -> SignalBus:
    return SignalBus(keepalive_interval=0.1)


def _inject_subscriber(bus: SignalBus, q: "asyncio.Queue[str]") -> None:
    """Register a pre-built queue as a subscriber (test helper)."""
    bus._loop = asyncio.get_event_loop()
    with bus._lock:
        bus._subscribers[id(q)] = q


class TestSignalBusIntegration:
    """Integration tests verifying publish/subscribe queue mechanics directly.

    httpx.ASGITransport cannot cancel an unbounded SSE generator — the transport
    buffers outbound chunks but never delivers a disconnect signal to the ASGI
    generator, so any test that exits a client.stream() block before the generator
    finishes hangs forever. These tests bypass ASGI and inspect the subscriber
    queues directly, which is both faster and more reliable.

    The ASGI wiring is a one-liner (StreamingResponse) verified by
    test_subscribe_returns_streaming_response below.
    """

    @pytest.mark.asyncio
    async def test_event_delivered_to_subscriber_queue(self) -> None:
        """publish() enqueues an SSE-formatted string to all subscriber queues."""
        bus = _make_bus()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _inject_subscriber(bus, q)

        ev = SignalEvent(
            source="merlin", kind="gesture",
            payload={"name": "open_palm", "confidence": 0.94},
            timestamp="2026-01-01T00:00:00Z",
        )
        bus.publish(ev)
        await asyncio.sleep(0)  # allow call_soon_threadsafe callback to run

        assert not q.empty()
        sse = q.get_nowait()
        data = json.loads(sse[len("data: "):].strip())
        assert data["source"] == "merlin"
        assert data["kind"] == "gesture"
        assert data["payload"]["name"] == "open_palm"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_each_receive_event(self) -> None:
        """Every connected subscriber queue receives its own copy of the event."""
        bus = _make_bus()
        q1: asyncio.Queue = asyncio.Queue(maxsize=10)
        q2: asyncio.Queue = asyncio.Queue(maxsize=10)
        _inject_subscriber(bus, q1)
        bus._loop = asyncio.get_event_loop()
        with bus._lock:
            bus._subscribers[id(q2)] = q2

        ev = SignalEvent(source="linnet", kind="tone", payload={}, timestamp="T")
        bus.publish(ev)
        await asyncio.sleep(0)

        assert not q1.empty()
        assert not q2.empty()
        d1 = json.loads(q1.get_nowait()[len("data: "):].strip())
        d2 = json.loads(q2.get_nowait()[len("data: "):].strip())
        assert d1["kind"] == "tone"
        assert d2["kind"] == "tone"

    @pytest.mark.asyncio
    async def test_publish_safe_while_subscribed(self) -> None:
        """Rapid publish() calls with active subscribers must not raise."""
        bus = _make_bus()
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        _inject_subscriber(bus, q)

        errors: list[Exception] = []
        try:
            for i in range(20):
                bus.publish(SignalEvent(source="s", kind="k", payload={"i": i}, timestamp="T"))
        except Exception as exc:
            errors.append(exc)

        await asyncio.sleep(0)
        assert not errors
        assert q.qsize() > 0

    @pytest.mark.asyncio
    async def test_overflow_does_not_block_producer(self) -> None:
        """A slow consumer with a tiny queue must not block publish()."""
        bus = SignalBus(queue_size=2, keepalive_interval=0.1)
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        _inject_subscriber(bus, q)

        for i in range(50):
            bus.publish(SignalEvent(source="s", kind="k", payload={"i": i}, timestamp="T"))

        await asyncio.sleep(0)
        # Drop-oldest: only the last 2 events survive
        assert q.qsize() == 2

    @pytest.mark.asyncio
    async def test_subscribe_returns_streaming_response(self) -> None:
        """subscribe() returns a StreamingResponse with text/event-stream media type."""
        bus = _make_bus()
        mock_req = MagicMock(spec=Request)

        resp = bus.subscribe(mock_req)

        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "text/event-stream"
        assert bus.subscriber_count == 1
