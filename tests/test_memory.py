"""Tests for circuitforge_core.memory.

These tests mock the mnemo SDK so no live sidecar is required.
"""
from __future__ import annotations

import sys
import time
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from circuitforge_core.memory import MemoryClient, MemoryConfig, MemoryUnavailableError
from circuitforge_core.memory.client import _MAX_FAILURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_mnemo(health_ok: bool = True):
    """Return a (mock_module, mock_inner_client) pair."""
    mock_health = MagicMock(
        status="ok" if health_ok else "error",
        provider_type="ollama",
        provider_model="llama3",
    )
    mock_client = AsyncMock()
    mock_client.health = AsyncMock(return_value=mock_health)
    mock_client.ingest = AsyncMock(return_value=MagicMock(chunk_id="abc", entities_extracted=2))
    mock_client.get_context = AsyncMock(return_value="Relevant context: user prefers dark mode")
    mock_client.list_entities = AsyncMock(return_value=[])
    mock_client.stats = AsyncMock(return_value=MagicMock(
        entity_count=5, chunk_count=10, node_count=5, edge_count=3, uptime_seconds=120.0
    ))
    mock_client.wipe = AsyncMock(return_value=None)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    mock_module = ModuleType("mnemo")
    mock_module.AsyncMnemoClient = MagicMock(return_value=mock_client)
    return mock_module, mock_client


async def _connected(health_ok: bool = True):
    """Return a connected MemoryClient with mock inner client attached."""
    mock_module, mock_inner = _make_mock_mnemo(health_ok=health_ok)
    client = MemoryClient(MemoryConfig())
    with patch.dict(sys.modules, {"mnemo": mock_module}):
        await client.connect()
    client._mock_inner = mock_inner
    return client


# ── Config ────────────────────────────────────────────────────────────────────

class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 8080
        assert cfg.base_url == "http://localhost:8080"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MNEMO_HOST", "mnemo-sidecar")
        monkeypatch.setenv("MNEMO_PORT", "9090")
        monkeypatch.setenv("MNEMO_TIMEOUT", "30.0")
        cfg = MemoryConfig.from_env()
        assert cfg.host == "mnemo-sidecar"
        assert cfg.port == 9090
        assert cfg.timeout == 30.0

    def test_base_url(self):
        cfg = MemoryConfig(host="10.1.10.5", port=8080)
        assert cfg.base_url == "http://10.1.10.5:8080"


# ── connect() ─────────────────────────────────────────────────────────────────

class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        client = await _connected(health_ok=True)
        assert client.available is True
        assert client.failure_count == 0

    @pytest.mark.asyncio
    async def test_connect_bad_health_status(self):
        client = await _connected(health_ok=False)
        assert client.available is False

    @pytest.mark.asyncio
    async def test_connect_sidecar_unreachable(self):
        mock_module, mock_client = _make_mock_mnemo()
        mock_client.health.side_effect = ConnectionRefusedError("refused")
        client = MemoryClient(MemoryConfig())
        with patch.dict(sys.modules, {"mnemo": mock_module}):
            await client.connect()  # must not raise
        assert client.available is False

    @pytest.mark.asyncio
    async def test_connect_strict_raises(self):
        mock_module, mock_client = _make_mock_mnemo()
        mock_client.health.side_effect = ConnectionRefusedError("refused")
        client = MemoryClient(MemoryConfig(), strict=True)
        with patch.dict(sys.modules, {"mnemo": mock_module}):
            with pytest.raises(MemoryUnavailableError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connect_missing_sdk(self):
        client = MemoryClient(MemoryConfig())
        with patch.dict(sys.modules, {"mnemo": None}):
            await client.connect()
        assert client.available is False


# ── No-op when unavailable ────────────────────────────────────────────────────

class TestNoopWhenUnavailable:
    @pytest.fixture
    def unavailable(self):
        return MemoryClient(MemoryConfig())

    @pytest.mark.asyncio
    async def test_remember_noop(self, unavailable):
        assert await unavailable.remember("text") is False

    @pytest.mark.asyncio
    async def test_recall_noop(self, unavailable):
        assert await unavailable.recall("query") == ""

    @pytest.mark.asyncio
    async def test_entities_noop(self, unavailable):
        assert await unavailable.entities() == []

    @pytest.mark.asyncio
    async def test_stats_noop(self, unavailable):
        assert await unavailable.stats() is None

    @pytest.mark.asyncio
    async def test_wipe_noop(self, unavailable):
        assert await unavailable.wipe() is False


# ── Live calls when connected ─────────────────────────────────────────────────

class TestLiveCalls:
    @pytest.mark.asyncio
    async def test_remember_calls_ingest(self):
        client = await _connected()
        result = await client.remember("hello world", source="test")
        assert result is True
        client._mock_inner.ingest.assert_awaited_once_with(
            content="hello world", source="test", session_id=None
        )

    @pytest.mark.asyncio
    async def test_remember_resets_failure_count(self):
        client = await _connected()
        client._failure_count = 2  # simulate prior failures
        await client.remember("text")
        assert client.failure_count == 0

    @pytest.mark.asyncio
    async def test_recall_returns_context(self):
        client = await _connected()
        ctx = await client.recall("dark mode preference")
        assert "dark mode" in ctx

    @pytest.mark.asyncio
    async def test_recall_with_session(self):
        client = await _connected()
        await client.recall("query", session_id="user-123")
        client._mock_inner.get_context.assert_awaited_once_with(
            text="query", session_id="user-123"
        )

    @pytest.mark.asyncio
    async def test_stats_returns_memory_stats(self):
        from circuitforge_core.memory import MemoryStats
        client = await _connected()
        result = await client.stats()
        assert isinstance(result, MemoryStats)
        assert result.available is True
        assert result.entity_count == 5


# ── Backoff and reconnect ─────────────────────────────────────────────────────

class TestBackoffAndReconnect:
    @pytest.mark.asyncio
    async def test_failure_count_increments(self):
        client = await _connected()
        client._mock_inner.ingest.side_effect = ConnectionResetError("reset")
        await client.remember("text")
        assert client.failure_count == 1

    @pytest.mark.asyncio
    async def test_client_disabled_after_max_failures(self):
        client = await _connected()
        client._mock_inner.ingest.side_effect = ConnectionResetError("reset")
        # drive failures to the limit
        for _ in range(_MAX_FAILURES):
            await client.remember("text")
        assert client.available is False

    @pytest.mark.asyncio
    async def test_retry_at_set_after_failure(self):
        client = await _connected()
        client._mock_inner.ingest.side_effect = ConnectionResetError("reset")
        before = time.monotonic()
        await client.remember("text")
        assert client._retry_at is not None
        assert client._retry_at > before

    @pytest.mark.asyncio
    async def test_backoff_increases_with_failures(self):
        client = await _connected()
        client._mock_inner.ingest.side_effect = ConnectionResetError("reset")

        retry_times = []
        t0 = time.monotonic()
        for _ in range(3):
            await client.remember("text")
            retry_times.append(client._retry_at - t0)

        # Each cooldown should be longer than the previous
        assert retry_times[1] > retry_times[0]
        assert retry_times[2] > retry_times[1]

    @pytest.mark.asyncio
    async def test_reconnect_attempted_after_cooldown(self):
        """Once the retry window elapses, the next call triggers a reconnect."""
        client = await _connected()
        # Force unavailable with an expired retry window
        client._available = False
        client._retry_at = time.monotonic() - 1.0  # already elapsed

        mock_module, mock_inner = _make_mock_mnemo(health_ok=True)
        with patch.dict(sys.modules, {"mnemo": mock_module}):
            result = await client.remember("text after reconnect")

        # Reconnect should have restored availability
        assert client.available is True
        assert result is True

    @pytest.mark.asyncio
    async def test_no_reconnect_during_cooldown(self):
        """Within the cooldown window, calls no-op without attempting reconnect."""
        client = await _connected()
        client._available = False
        client._retry_at = time.monotonic() + 999.0  # far in the future

        mock_module, _ = _make_mock_mnemo(health_ok=True)
        with patch.dict(sys.modules, {"mnemo": mock_module}):
            result = await client.remember("text during cooldown")

        assert result is False
        assert client.available is False  # no reconnect fired

    @pytest.mark.asyncio
    async def test_success_resets_retry_state(self):
        """A successful call clears failure_count and retry_at."""
        client = await _connected()
        client._failure_count = 2
        client._retry_at = time.monotonic() + 30.0

        await client.remember("successful call")

        assert client.failure_count == 0
        assert client._retry_at is None

    @pytest.mark.asyncio
    async def test_strict_raises_after_max_failures(self):
        """strict=True raises MemoryUnavailableError once failure threshold is hit."""
        client = await _connected()
        client._strict = True
        client._mock_inner.ingest.side_effect = ConnectionResetError("reset")

        with pytest.raises(MemoryUnavailableError):
            for _ in range(_MAX_FAILURES):
                await client.remember("text")
