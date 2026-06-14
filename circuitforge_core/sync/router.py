"""FastAPI router factory for the sync module.

MIT licensed.

Usage:
    from circuitforge_core.sync import make_sync_router, SyncConfig

    sync_router = make_sync_router(
        product="kiwi",
        get_session=_sessions.dependency(),
        require_paid=_sessions.require_tier("paid"),
        config=SyncConfig.from_env("kiwi"),
    )
    app.include_router(sync_router, prefix="/sync", tags=["sync"])
"""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from circuitforge_core.sync.db import SyncDB
from circuitforge_core.sync.models import SyncConfig
from circuitforge_core.sync.store import SyncPrefsStore, SyncStore


class _PushRequest(BaseModel):
    data_class: str
    blob: str
    updated_at: str


class _PrefPatch(BaseModel):
    data_class: str
    enabled: bool


def make_sync_router(
    *,
    product: str,
    get_session: Callable,
    require_paid: Callable,
    config: SyncConfig | None = None,
    encryption_key: str = "",
) -> APIRouter:
    """Return a configured sync APIRouter for the given product.

    Args:
        product:        Product slug, e.g. "kiwi". Used as the partition key.
        get_session:    FastAPI dependency yielding a user object with a
                        ``user_id`` attribute (e.g. CloudSessionFactory.dependency()).
        require_paid:   FastAPI dependency that raises 403 for free-tier users
                        (e.g. CloudSessionFactory.require_tier("paid")).
        config:         SyncConfig; defaults to SyncConfig.from_env(product).
        encryption_key: SQLCipher key for at-rest encryption. Defaults to the
                        SYNC_DB_KEY env var, or unencrypted if absent.
    """
    cfg = config or SyncConfig.from_env(product)
    key = encryption_key or os.environ.get("SYNC_DB_KEY", "")
    db = SyncDB(cfg.db_path, key=key)
    db.run_migrations()

    prefs_store = SyncPrefsStore(db)
    blob_store = SyncStore(db, prefs_store)

    router = APIRouter()

    # ------------------------------------------------------------------
    # Consent / preferences (#57) — no tier gate; prefs are always accessible
    # ------------------------------------------------------------------

    @router.get("/prefs")
    def get_prefs(user: Any = Depends(get_session)) -> dict:
        """Return all sync preferences for this user+product."""
        return prefs_store.get_sync_prefs(user.user_id, product)

    @router.patch("/prefs")
    def patch_pref(body: _PrefPatch, user: Any = Depends(get_session)) -> dict:
        """Enable or disable sync for a single data_class."""
        pref = prefs_store.set_sync_pref(
            user.user_id, product, body.data_class, body.enabled
        )
        return {"data_class": pref.data_class, "enabled": pref.enabled}

    # ------------------------------------------------------------------
    # Blob storage (#56) — Paid+ required for push/pull
    # ------------------------------------------------------------------

    @router.post("/push")
    def push_blob(
        body: _PushRequest,
        user: Any = Depends(require_paid),
    ) -> dict:
        """Push a localStorage blob to the server (last-write-wins).

        Returns {"written": false} when the data_class is not consented or the
        client timestamp is older than the stored value.
        """
        written = blob_store.push(
            user.user_id, product, body.data_class, body.blob, body.updated_at
        )
        return {"written": written}

    @router.get("/pull")
    def pull_blobs(
        user: Any = Depends(require_paid),
    ) -> list[dict]:
        """Return all consented blobs for this user+product."""
        blobs = blob_store.pull(user.user_id, product)
        return [
            {
                "data_class": b.data_class,
                "blob": b.blob,
                "updated_at": b.updated_at,
            }
            for b in blobs
        ]

    @router.delete("/blob/{data_class}")
    def delete_blob(data_class: str, user: Any = Depends(get_session)) -> dict:
        """Delete a single blob. Tier-free — always available, even after downgrade."""
        blob_store.delete(user.user_id, product, data_class)
        return {"deleted": data_class}

    # ------------------------------------------------------------------
    # Wipe endpoints (#57) — tier-free; users must always be able to delete
    # ------------------------------------------------------------------

    @router.delete("/data")
    def wipe_data(user: Any = Depends(get_session)) -> dict:
        """Delete all blobs for this user+product. Prefs are preserved."""
        blob_store.delete_all(user.user_id, product)
        return {"wiped": "data", "product": product}

    @router.delete("/all")
    def wipe_all(user: Any = Depends(get_session)) -> dict:
        """Delete all blobs and reset all prefs for this user+product. Immediate."""
        prefs_store.wipe_sync_data(user.user_id, product)
        return {"wiped": "all", "product": product}

    return router
