"""Tests for make_sync_router — FastAPI endpoint behaviour."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from circuitforge_core.sync import make_sync_router, SyncConfig


def _make_user(user_id: str) -> Any:
    u = MagicMock()
    u.user_id = user_id
    return u


def _make_app(tmp_path: Path, user_id: str = "u1", paid: bool = True) -> TestClient:
    user = _make_user(user_id)

    def get_session():
        return user

    def require_paid():
        if not paid:
            raise HTTPException(status_code=403, detail="Paid tier required")
        return user

    app = FastAPI()
    router = make_sync_router(
        product="kiwi",
        get_session=get_session,
        require_paid=require_paid,
        config=SyncConfig(db_path=tmp_path / "sync.db", product="kiwi"),
    )
    app.include_router(router, prefix="/sync")
    return TestClient(app)


class TestPrefsEndpoints:
    def test_get_prefs_empty_on_new_user(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        resp = client.get("/sync/prefs")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_patch_pref_enables(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        resp = client.patch(
            "/sync/prefs", json={"data_class": "cook_log", "enabled": True}
        )
        assert resp.status_code == 200
        assert resp.json() == {"data_class": "cook_log", "enabled": True}

    def test_get_prefs_reflects_patch(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": True})
        resp = client.get("/sync/prefs")
        assert resp.json() == {"cook_log": True}

    def test_patch_disable_after_enable(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": True})
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": False})
        resp = client.get("/sync/prefs")
        assert resp.json()["cook_log"] is False


class TestPushPullPaidGating:
    def test_push_requires_paid(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path, paid=False)
        resp = client.post(
            "/sync/push",
            json={"data_class": "cook_log", "blob": "{}", "updated_at": "2026-01-01T00:00:00Z"},
        )
        assert resp.status_code == 403

    def test_pull_requires_paid(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path, paid=False)
        assert client.get("/sync/pull").status_code == 403

    def test_push_returns_false_when_not_consented(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        resp = client.post(
            "/sync/push",
            json={"data_class": "cook_log", "blob": "{}", "updated_at": "2026-01-01T00:00:00Z"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"written": False}

    def test_push_returns_true_when_consented(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": True})
        resp = client.post(
            "/sync/push",
            json={"data_class": "cook_log", "blob": '{"x":1}', "updated_at": "2026-01-01T00:00:00Z"},
        )
        assert resp.json() == {"written": True}

    def test_pull_returns_consented_blob(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": True})
        client.post(
            "/sync/push",
            json={"data_class": "cook_log", "blob": '{"meals":[]}', "updated_at": "2026-01-01T00:00:00Z"},
        )
        resp = client.get("/sync/pull")
        assert resp.status_code == 200
        blobs = resp.json()
        assert len(blobs) == 1
        assert blobs[0]["blob"] == '{"meals":[]}'


class TestDeleteEndpoints:
    def test_delete_blob_is_tier_free(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path, paid=False)
        resp = client.delete("/sync/blob/cook_log")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "cook_log"

    def test_wipe_data_is_tier_free(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path, paid=False)
        assert client.delete("/sync/data").status_code == 200

    def test_wipe_all_is_tier_free(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path, paid=False)
        assert client.delete("/sync/all").status_code == 200

    def test_wipe_all_clears_blobs_and_prefs(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": True})
        client.post(
            "/sync/push",
            json={"data_class": "cook_log", "blob": "data", "updated_at": "2026-01-01T00:00:00Z"},
        )
        client.delete("/sync/all")
        assert client.get("/sync/pull").json() == []
        assert client.get("/sync/prefs").json() == {}

    def test_wipe_data_preserves_prefs(self, tmp_path: Path) -> None:
        client = _make_app(tmp_path)
        client.patch("/sync/prefs", json={"data_class": "cook_log", "enabled": True})
        client.post(
            "/sync/push",
            json={"data_class": "cook_log", "blob": "data", "updated_at": "2026-01-01T00:00:00Z"},
        )
        client.delete("/sync/data")
        assert client.get("/sync/pull").json() == []
        assert client.get("/sync/prefs").json() == {"cook_log": True}
