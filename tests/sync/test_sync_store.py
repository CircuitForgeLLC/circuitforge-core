"""Tests for SyncStore — opaque blob push/pull/delete (cf-core #56)."""
from __future__ import annotations

from pathlib import Path

import pytest

from circuitforge_core.sync.db import SyncDB
from circuitforge_core.sync.store import SyncPrefsStore, SyncStore


@pytest.fixture
def db(tmp_path: Path) -> SyncDB:
    d = SyncDB(tmp_path / "sync.db")
    d.run_migrations()
    return d


@pytest.fixture
def prefs(db: SyncDB) -> SyncPrefsStore:
    return SyncPrefsStore(db)


@pytest.fixture
def store(db: SyncDB, prefs: SyncPrefsStore) -> SyncStore:
    return SyncStore(db, prefs)


def _enable(prefs: SyncPrefsStore, user: str, product: str, dc: str) -> None:
    prefs.set_sync_pref(user, product, dc, True)


class TestPushRejectedWhenNotConsented:
    def test_push_returns_false_without_pref(self, store: SyncStore) -> None:
        written = store.push("u1", "kiwi", "cook_log", '{"x":1}', "2026-01-01T00:00:00Z")
        assert written is False

    def test_push_returns_false_when_pref_disabled(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        prefs.set_sync_pref("u1", "kiwi", "cook_log", False)
        written = store.push("u1", "kiwi", "cook_log", '{"x":1}', "2026-01-01T00:00:00Z")
        assert written is False


class TestPushAcceptedWhenConsented:
    def test_push_returns_true_when_enabled(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        written = store.push("u1", "kiwi", "cook_log", '{"x":1}', "2026-01-01T00:00:00Z")
        assert written is True

    def test_pulled_blob_matches_pushed_blob(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        store.push("u1", "kiwi", "cook_log", '{"meals":[]}', "2026-01-01T00:00:00Z")
        blobs = store.pull("u1", "kiwi")
        assert len(blobs) == 1
        assert blobs[0].blob == '{"meals":[]}'
        assert blobs[0].data_class == "cook_log"


class TestLastWriteWins:
    def test_newer_write_overwrites(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        store.push("u1", "kiwi", "cook_log", "old", "2026-01-01T00:00:00Z")
        store.push("u1", "kiwi", "cook_log", "new", "2026-01-02T00:00:00Z")
        blobs = store.pull("u1", "kiwi")
        assert blobs[0].blob == "new"

    def test_older_write_rejected(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        store.push("u1", "kiwi", "cook_log", "current", "2026-01-02T00:00:00Z")
        store.push("u1", "kiwi", "cook_log", "stale", "2026-01-01T00:00:00Z")
        blobs = store.pull("u1", "kiwi")
        assert blobs[0].blob == "current"

    def test_equal_timestamp_does_not_regress(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        store.push("u1", "kiwi", "cook_log", "v1", "2026-01-01T12:00:00Z")
        store.push("u1", "kiwi", "cook_log", "v2", "2026-01-01T12:00:00Z")
        blobs = store.pull("u1", "kiwi")
        assert blobs[0].blob == "v1"


class TestPullFiltering:
    def test_pull_returns_only_consented_classes(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        # bookmarks not consented
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        blobs = store.pull("u1", "kiwi")
        assert len(blobs) == 1
        assert blobs[0].data_class == "cook_log"

    def test_pull_empty_when_nothing_consented(self, store: SyncStore) -> None:
        blobs = store.pull("u1", "kiwi")
        assert blobs == []

    def test_pull_specific_data_classes(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        _enable(prefs, "u1", "kiwi", "bookmarks")
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        store.push("u1", "kiwi", "bookmarks", "bm", "2026-01-01T00:00:00Z")
        blobs = store.pull("u1", "kiwi", data_classes=["cook_log"])
        assert len(blobs) == 1
        assert blobs[0].data_class == "cook_log"


class TestBlobOpacity:
    def test_arbitrary_json_round_trips_untouched(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        raw = '{"nested":{"a":1},"arr":[1,2,3],"unicode":"héllo"}'
        store.push("u1", "kiwi", "cook_log", raw, "2026-01-01T00:00:00Z")
        blobs = store.pull("u1", "kiwi")
        assert blobs[0].blob == raw


class TestDelete:
    def test_delete_single_blob(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        _enable(prefs, "u1", "kiwi", "bookmarks")
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        store.push("u1", "kiwi", "bookmarks", "bm", "2026-01-01T00:00:00Z")
        store.delete("u1", "kiwi", "cook_log")
        blobs = store.pull("u1", "kiwi")
        assert len(blobs) == 1
        assert blobs[0].data_class == "bookmarks"

    def test_delete_all_blobs(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        store.push("u1", "kiwi", "cook_log", "cl", "2026-01-01T00:00:00Z")
        store.delete_all("u1", "kiwi")
        assert store.pull("u1", "kiwi") == []

    def test_delete_does_not_touch_other_users(
        self, store: SyncStore, prefs: SyncPrefsStore
    ) -> None:
        _enable(prefs, "u1", "kiwi", "cook_log")
        _enable(prefs, "u2", "kiwi", "cook_log")
        store.push("u1", "kiwi", "cook_log", "u1data", "2026-01-01T00:00:00Z")
        store.push("u2", "kiwi", "cook_log", "u2data", "2026-01-01T00:00:00Z")
        store.delete_all("u1", "kiwi")
        blobs = store.pull("u2", "kiwi")
        assert len(blobs) == 1
        assert blobs[0].blob == "u2data"
