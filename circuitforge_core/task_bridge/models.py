"""Data models for the cf-core task_bridge module.

MIT licensed. Ticket: cf-core #66. Design spec:
circuitforge-plans/shared/superpowers/specs/2026-07-05-focus-flow-task-bridge-design.md
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

SCHEMA_VERSION = 1

# v1 external tasks are always "flexible" — Focus Flow's calm task kind.
# External sources never set "inflexible" / "critical" / "locked" / "surprise",
# so no CF product can inject urgency into another product's UX.
TaskKind = Literal["flexible"]
TaskStatus = Literal["active", "cancelled"]

_VALID_STATUSES: frozenset[str] = frozenset({"active", "cancelled"})


@dataclass(frozen=True)
class ExternalTask:
    """
    A task pushed from a CF product into an external scheduler (e.g. Focus Flow).

    Pure data contract — no transport, no auth, no product-specific behavior.
    `external_id` must be stable and idempotent per source item (e.g.
    "kiwi:item:1234") so repeated pushes upsert rather than duplicate.
    """

    source_product: str
    external_id: str
    title: str
    due_at: str  # ISO 8601 UTC, e.g. "2026-07-07T00:00:00Z"
    notes: str | None = None
    kind: TaskKind = "flexible"
    status: TaskStatus = "active"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source_product:
            raise ValueError("source_product must not be empty")
        if not self.external_id:
            raise ValueError("external_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")
        if self.kind != "flexible":
            raise ValueError(
                f"kind must be 'flexible' in schema v{SCHEMA_VERSION}, got {self.kind!r}"
            )
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_STATUSES)}, got {self.status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the wire format expected by an importer's HTTP API."""
        return asdict(self)
