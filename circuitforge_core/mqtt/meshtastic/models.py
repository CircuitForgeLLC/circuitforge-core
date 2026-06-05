"""Data models for Meshtastic packets.

MIT licensed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# Meshtastic portnum → our label
PacketType = Literal[
    "text",
    "position",
    "nodeinfo",
    "telemetry",
    "routing",
    "admin",
    "unknown",
]


@dataclass(frozen=True)
class MeshtasticPosition:
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: int | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class MeshtasticTelemetry:
    battery_level: int | None = None     # 0-100 %
    voltage: float | None = None         # volts
    channel_util: float | None = None    # 0-100 %
    air_util_tx: float | None = None     # 0-100 %


@dataclass(frozen=True)
class MeshtasticPacket:
    """Normalized Meshtastic packet from any backend."""

    packet_type: PacketType
    from_id: str                          # hex node ID, e.g. "!deadbeef"
    from_num: int                         # numeric node ID
    to_num: int                           # 0xffffffff = broadcast
    channel: int
    received_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Type-specific payloads (only one is populated per packet type)
    text: str | None = None
    position: MeshtasticPosition | None = None
    telemetry: MeshtasticTelemetry | None = None
    node_longname: str | None = None
    node_shortname: str | None = None
    hardware: int | None = None

    # Original raw payload dict for downstream consumers that need all fields
    raw: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_broadcast(self) -> bool:
        return self.to_num == 0xFFFFFFFF

    def summary(self) -> str:
        """One-line human-readable description."""
        src = self.from_id or f"!{self.from_num:08x}"
        if self.packet_type == "text":
            return f"[{src}] {self.text}"
        if self.packet_type == "position" and self.position:
            p = self.position
            return f"[{src}] position {p.latitude:.5f},{p.longitude:.5f}"
        if self.packet_type == "nodeinfo":
            return f"[{src}] node info: {self.node_longname!r} ({self.node_shortname})"
        if self.packet_type == "telemetry" and self.telemetry:
            t = self.telemetry
            parts = []
            if t.battery_level is not None:
                parts.append(f"batt={t.battery_level}%")
            if t.voltage is not None:
                parts.append(f"v={t.voltage:.2f}V")
            return f"[{src}] telemetry {' '.join(parts)}"
        return f"[{src}] {self.packet_type} packet"
