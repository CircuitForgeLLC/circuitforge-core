"""Tests for SyncPrefsStore — consent layer (cf-core #57)."""
from __future__ import annotations

from pathlib import Path

import pytest

from circuitforge_core.sync.db import SyncDB
from circuitforge_core.sync.store import SyncPrefsStore


@pytest.fixture
def db(tmp_path: Path) -> SyncDB:
    d = SyncDB(tmp_path / "sync.db")
    d.run_migrations()
    return d


@pytest.fixture
def prefs(db: SyncDB) -> SyncPrefsStore:
    return SyncPrefsStore(db)


class TestOptInByDefault:
    def test_absent_row_returns_disabled(self, prefs: SyncPrefsStore) -> None:
        assert prefs.is_enabled("u1", "kiwi", "cook_log") is False

    def test_get_sync_prefs_returns_empty_for_new_user(self, prefs: SyncPrefsStore) -> None:
        assert prefs.get_sync_prefs("u1", "kiwi") == {}

    def test_get_all_sync_prefs_empty_for_new_user(self, prefs: SyncPrefsStore) -> None:
        assert prefs.get_all_sync_prefs("u1") == {}


class TestSetPref:
    def test_enable_returns_pref_with_enabled_true(self, prefs: SyncPrefsStore) -> None:
        pref = prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        assert pref.enabled is True
        assert pref.data_class == "cook_log"

    def test_enable_persists(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        assert prefs.is_enabled("u1", "kiwi", "cook_log") is True

    def test_disable_after_enable(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "kiwi", "cook_log", False)
        assert prefs.is_enabled("u1", "kiwi", "cook_log") is False

    def test_idempotent_enable(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        assert prefs.is_enabled("u1", "kiwi", "cook_log") is True

    def test_prefs_isolated_by_user(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        assert prefs.is_enabled("u2", "kiwi", "cook_log") is False

    def test_prefs_isolated_by_product(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        assert prefs.is_enabled("u1", "peregrine", "cook_log") is False

    def test_prefs_isolated_by_data_class(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        assert prefs.is_enabled("u1", "kiwi", "bookmarks") is False


class TestGetSyncPrefs:
    def test_returns_all_known_classes(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "kiwi", "bookmarks", False)
        result = prefs.get_sync_prefs("u1", "kiwi")
        assert result == {"cook_log": True, "bookmarks": False}

    def test_scoped_to_product(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "peregrine", "dismissed", True)
        assert "dismissed" not in prefs.get_sync_prefs("u1", "kiwi")

    def test_get_all_spans_products(self, prefs: SyncPrefsStore) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "peregrine", "dismissed", True)
        result = prefs.get_all_sync_prefs("u1")
        assert result == {"kiwi": {"cook_log": True}, "peregrine": {"dismissed": True}}


class TestWipeSyncData:
    def test_wipe_specific_data_class(
        self, db: SyncDB, prefs: SyncPrefsStore
    ) -> None:
        from circuitforge_core.sync.store import SyncStore
        store = SyncStore(db, prefs)
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        store.push("u1", "kiwi", "cook_log", "data", "2026-01-01T00:00:00Z")
        prefs.wipe_sync_data("u1", "kiwi", "cook_log")
        assert store.pull("u1", "kiwi") == []
        assert prefs.is_enabled("u1", "kiwi", "cook_log") is False

    def test_wipe_product_clears_all_classes(
        self, db: SyncDB, prefs: SyncPrefsStore
    ) -> None:
        from circuitforge_core.sync.store import SyncStore
        store = SyncStore(db, prefs)
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "kiwi", "bookmarks", True)
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        store.push("u1", "kiwi", "bookmarks", "bm", "2026-01-01T00:00:00Z")
        prefs.wipe_sync_data("u1", "kiwi")
        assert store.pull("u1", "kiwi") == []
        assert prefs.get_sync_prefs("u1", "kiwi") == {}

    def test_wipe_does_not_touch_other_products(
        self, db: SyncDB, prefs: SyncPrefsStore
    ) -> None:
        from circuitforge_core.sync.store import SyncStore
        store = SyncStore(db, prefs)
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "peregrine", "dismissed", True)
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        store.push("u1", "peregrine", "dismissed", "dj", "2026-01-01T00:00:00Z")
        prefs.wipe_sync_data("u1", "kiwi")
        assert store.pull("u1", "peregrine") != []
        assert prefs.is_enabled("u1", "peregrine", "dismissed") is True

    def test_wipe_all_products(
        self, db: SyncDB, prefs: SyncPrefsStore
    ) -> None:
        from circuitforge_core.sync.store import SyncStore
        store = SyncStore(db, prefs)
        prefs.set_sync_pref("u1", "kiwi", "cook_log", True)
        prefs.set_sync_pref("u1", "peregrine", "dismissed", True)
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        prefs.wipe_sync_data("u1")
        assert store.pull("u1", "kiwi") == []
        assert prefs.get_all_sync_prefs("u1") == {}
