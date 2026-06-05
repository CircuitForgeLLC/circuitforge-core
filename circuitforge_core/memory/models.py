"""Data models for the cf-core memory module.

MIT licensed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class MemoryConfig:
    """Connection config for a mnemo sidecar."""

    host: str = "localhost"
    port: int = 8080
    timeout: float = 10.0

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Read config from environment variables.

        Variables:
            MNEMO_HOST  — default: localhost
            MNEMO_PORT  — default: 8080
            MNEMO_TIMEOUT — default: 10.0
        """
        return cls(
            host=os.environ.get("MNEMO_HOST", "localhost"),
            port=int(os.environ.get("MNEMO_PORT", "8080")),
            timeout=float(os.environ.get("MNEMO_TIMEOUT", "10.0")),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class MemoryEntity:
    """A named entity extracted and stored by the mnemo knowledge graph."""

    entity_id: str
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source_count: int = 1

    @classmethod
    def from_mnemo(cls, obj) -> MemoryEntity:
        """Convert a mnemo-sdk Entity object to MemoryEntity."""
        return cls(
            entity_id=str(obj.id),
            name=obj.name,
            entity_type=obj.entity_type,
            aliases=list(obj.aliases or []),
            confidence=float(obj.confidence or 1.0),
            source_count=int(obj.source_count or 1),
        )


@dataclass(frozen=True)
class MemoryStats:
    """Snapshot of the mnemo knowledge graph state."""

    entity_count: int
    chunk_count: int
    node_count: int
    edge_count: int
    uptime_seconds: float
    available: bool
