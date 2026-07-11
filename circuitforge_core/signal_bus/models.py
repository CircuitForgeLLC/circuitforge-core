"""Data models for the cf-core signal_bus module.

MIT licensed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SignalEvent:
    """An event published to the signal bus.

    Args:
        source:    Originating service, e.g. "merlin", "linnet".
        kind:      Event type, e.g. "gesture", "tone", "alpha_rising".
        payload:   Arbitrary dict — contents are opaque to the bus.
        timestamp: ISO-8601 UTC. Auto-populated if not supplied.
    """

    source: str
    kind: str
    payload: dict
    timestamp: str = field(default_factory=_utcnow)

    def to_sse(self) -> str:
        """Serialize to SSE wire format: ``data: {...}\\n\\n``."""
        data = json.dumps(
            {
                "source": self.source,
                "kind": self.kind,
                "payload": self.payload,
                "timestamp": self.timestamp,
            }
        )
        return f"data: {data}\n\n"
