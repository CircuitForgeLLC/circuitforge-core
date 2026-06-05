"""Data models for the MQTT client module.

MIT licensed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class MQTTConfig:
    """Connection config for an MQTT broker."""

    host: str
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str = ""
    keepalive: int = 60
    tls: bool = False
    reconnect_interval: float = 5.0


@dataclass(frozen=True)
class MQTTMessage:
    """A single received MQTT message."""

    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False
    received_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def text(self, encoding: str = "utf-8") -> str:
        return self.payload.decode(encoding, errors="replace")

    def json(self) -> dict:
        return json.loads(self.payload)

    @property
    def topic_parts(self) -> list[str]:
        return self.topic.split("/")
