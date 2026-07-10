"""cf-core signal_bus — generic SSE event publisher for real-time signal streams.

MIT licensed. Ticket: cf-core #58.

## Quick start

    from circuitforge_core.signal_bus import SignalBus, SignalEvent
    from fastapi import Request

    bus = SignalBus()

    # Producer — safe to call from sync threads (OpenCV, Meshtastic, PyPubSub)
    bus.publish(SignalEvent(
        source="merlin",
        kind="gesture",
        payload={"name": "open_palm", "confidence": 0.94},
    ))

    # Consumer — SSE endpoint
    @app.get("/events")
    async def events(request: Request):
        return bus.subscribe(request)

## Notes

- Each subscriber gets an independent bounded asyncio.Queue (default: 100 events).
- When full, the oldest event is dropped to make room for the newest.
- publish() is thread-safe via loop.call_soon_threadsafe().
- Keepalive comment sent every 15 s to keep proxies from closing idle connections.
- Install: no extra dependencies beyond fastapi (already a core dep for service products).
"""

from circuitforge_core.signal_bus.bus import SignalBus
from circuitforge_core.signal_bus.models import SignalEvent

__all__ = ["SignalBus", "SignalEvent"]
