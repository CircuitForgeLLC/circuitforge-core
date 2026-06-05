"""MemoryClient — async wrapper around the mnemo persistent knowledge graph.

mnemo is an optional sidecar (https://github.com/zaydmulani09/mnemo).
When the sidecar is not running, all operations silently no-op so products
can call memory methods unconditionally without try/except.

MIT licensed.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from circuitforge_core.memory.models import MemoryConfig, MemoryEntity, MemoryStats

logger = logging.getLogger(__name__)

# Backoff schedule: 5 * 2^(failure-1), capped at _MAX_BACKOFF seconds.
# failure 1 →  5s, 2 → 10s, 3 → 20s, 4 → 40s, 5+ → 60s
_MAX_FAILURES: int = 3
_MAX_BACKOFF: float = 60.0


class MemoryUnavailableError(RuntimeError):
    """Raised only when strict=True and mnemo is not reachable."""


class MemoryClient:
    """Async interface to the mnemo knowledge graph sidecar.

    Resilience model:
    - If the sidecar is unreachable at connect(), logs once and enters no-op mode.
    - If a live call fails, the failure is counted. Each failure schedules an
      exponentially increasing cooldown before the next reconnect attempt.
    - After _MAX_FAILURES consecutive failures the client is marked unavailable;
      all calls no-op until the cooldown elapses and a reconnect succeeds.
    - Any successful call resets the failure counter.

    Usage (in a FastAPI lifespan)::

        from circuitforge_core.memory import MemoryClient, MemoryConfig

        memory = MemoryClient(MemoryConfig.from_env())

        @asynccontextmanager
        async def lifespan(app):
            await memory.connect()
            yield
            await memory.close()

    Then in handlers::

        await memory.remember("User prefers dark mode", source="settings")
        context = await memory.recall("What are the user's UI preferences?")
    """

    def __init__(self, config: MemoryConfig | None = None, *, strict: bool = False) -> None:
        """
        Args:
            config: connection settings; defaults to MemoryConfig.from_env()
            strict: if True, MemoryUnavailableError is raised on connect failure
                or after _MAX_FAILURES consecutive call failures
        """
        self._config = config or MemoryConfig.from_env()
        self._strict = strict
        self._available = False
        self._client: Any = None       # mnemo AsyncMnemoClient, set in connect()
        self._failure_count: int = 0
        self._retry_at: float | None = None  # monotonic timestamp; None = no retry pending

    @property
    def available(self) -> bool:
        """True if the mnemo sidecar was reachable at last health check."""
        return self._available

    @property
    def failure_count(self) -> int:
        """Consecutive call failures since the last success."""
        return self._failure_count

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Attempt to connect to the mnemo sidecar and run a health check.

        Safe to call multiple times (used internally for reconnect). If the
        sidecar is not reachable, logs a warning and enters no-op mode.
        Does NOT raise unless strict=True.
        """
        try:
            from mnemo import AsyncMnemoClient
        except ImportError:
            logger.debug(
                "mnemo-sdk not installed — memory module disabled. "
                "Install with: pip install circuitforge-core[memory]"
            )
            self._available = False
            return

        self._client = AsyncMnemoClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout,
        )
        try:
            health = await self._client.health()
            if health.status == "ok":
                self._available = True
                self._on_call_success()
                logger.info(
                    "mnemo memory sidecar connected at %s (LLM: %s/%s)",
                    self._config.base_url,
                    health.provider_type,
                    health.provider_model,
                )
            else:
                self._handle_unavailable("connect", reason=f"health status={health.status!r}")
        except Exception as exc:
            self._handle_unavailable("connect", reason=str(exc))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None
        self._available = False
        self._retry_at = None

    # ── Core API ──────────────────────────────────────────────────────────────

    async def remember(
        self,
        text: str,
        *,
        source: str = "cf-core",
        session_id: str | None = None,
    ) -> bool:
        """Store a text fragment in the knowledge graph.

        mnemo extracts named entities and relationships from the text and
        updates its graph. Large texts should be pre-chunked by the caller
        (mnemo stores each call as a single chunk with no sub-splitting).

        Args:
            text: the text to store (conversation turn, fact, note, etc.)
            source: label for the origin (e.g. "chat", "settings", "search")
            session_id: optional session grouping for multi-turn retrieval

        Returns:
            True if stored, False if sidecar unavailable.
        """
        if not await self._maybe_reconnect():
            return False
        try:
            await self._client.ingest(content=text, source=source, session_id=session_id)
            self._on_call_success()
            return True
        except Exception as exc:
            self._on_call_error("remember", exc)
            return False

    async def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
    ) -> str:
        """Retrieve a formatted context block relevant to query.

        Returns a prompt-ready string (or empty string if unavailable).
        Inject the result directly into a system prompt::

            context = await memory.recall("user dietary restrictions")
            system = f"You are a helpful assistant.\\n\\n{context}"

        Args:
            query: natural language question or topic to retrieve context for
            session_id: restrict retrieval to a specific session (optional)

        Returns:
            Formatted context string, or "" if sidecar unavailable.
        """
        if not await self._maybe_reconnect():
            return ""
        try:
            result = await self._client.get_context(text=query, session_id=session_id)
            self._failure_count = 0
            return result
        except Exception as exc:
            self._on_call_error("recall", exc)
            return ""

    async def entities(self, *, limit: int = 50) -> list[MemoryEntity]:
        """Return the most recent named entities in the knowledge graph.

        Args:
            limit: max entities to return (default 50)

        Returns:
            List of MemoryEntity objects, or [] if unavailable.
        """
        if not await self._maybe_reconnect():
            return []
        try:
            raw = await self._client.list_entities(limit=limit)
            self._on_call_success()
            return [MemoryEntity.from_mnemo(e) for e in raw]
        except Exception as exc:
            self._on_call_error("entities", exc)
            return []

    async def stats(self) -> MemoryStats | None:
        """Return knowledge graph statistics, or None if unavailable."""
        if not await self._maybe_reconnect():
            return None
        try:
            s = await self._client.stats()
            self._on_call_success()
            return MemoryStats(
                entity_count=s.entity_count,
                chunk_count=s.chunk_count,
                node_count=s.node_count,
                edge_count=s.edge_count,
                uptime_seconds=s.uptime_seconds,
                available=True,
            )
        except Exception as exc:
            self._on_call_error("stats", exc)
            return None

    async def wipe(self) -> bool:
        """Delete all stored memory. Irreversible.

        Returns True on success, False if unavailable or failed.
        """
        if not await self._maybe_reconnect():
            return False
        try:
            await self._client.wipe()
            self._on_call_success()
            logger.warning("mnemo memory wiped — all entities and chunks deleted")
            return True
        except Exception as exc:
            self._on_call_error("wipe", exc)
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _maybe_reconnect(self) -> bool:
        """Return True if the client is available (or just became available).

        Called at the top of every public method. If the client is unavailable
        but the retry cooldown has elapsed, silently attempts reconnect before
        answering. No-ops immediately if still within the cooldown window.
        """
        if self._available:
            return True
        if self._retry_at is not None and time.monotonic() >= self._retry_at:
            logger.info(
                "mnemo: cooldown elapsed after %d failure(s) — attempting reconnect",
                self._failure_count,
            )
            self._retry_at = None
            self._client = None
            await self.connect()
        return self._available

    def _on_call_success(self) -> None:
        """Reset failure state after a successful call."""
        self._failure_count = 0
        self._retry_at = None

    def _handle_unavailable(self, operation: str, reason: str = "") -> None:
        """Called when the sidecar is unreachable at connect() time."""
        self._available = False
        msg = f"mnemo memory sidecar unavailable (operation={operation!r})"
        if reason:
            msg += f": {reason}"
        if self._strict:
            raise MemoryUnavailableError(msg)
        logger.warning("%s — memory features disabled", msg)

    def _on_call_error(self, operation: str, exc: Exception) -> None:
        """Count consecutive failures and schedule exponential backoff retry.

        Backoff: 5 * 2^(failure-1) seconds, capped at 60s.
            failure 1 →  5s
            failure 2 → 10s
            failure 3 → 20s  ← _MAX_FAILURES default; client disabled here
            failure 4 → 40s
            failure 5+ → 60s

        After _MAX_FAILURES, _available is set to False and all calls no-op
        until _maybe_reconnect() fires after the cooldown elapses.
        """
        self._failure_count += 1
        backoff = min(5.0 * (2 ** (self._failure_count - 1)), _MAX_BACKOFF)
        self._retry_at = time.monotonic() + backoff

        if self._failure_count >= _MAX_FAILURES:
            self._available = False
            logger.warning(
                "mnemo %r failed %d consecutive times (%s) — disabled, reconnect in %.0fs",
                operation, self._failure_count, exc, backoff,
            )
            if self._strict:
                raise MemoryUnavailableError(
                    f"mnemo {operation!r} failed {self._failure_count} consecutive times: {exc}"
                )
        else:
            logger.warning(
                "mnemo %r failed (%d/%d): %s — retry in %.0fs",
                operation, self._failure_count, _MAX_FAILURES, exc, backoff,
            )
