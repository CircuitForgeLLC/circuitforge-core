"""Data access layer for the sync module.

MIT licensed.

SyncStore  — #56: opaque blob push/pull/delete
SyncPrefsStore — #57: per-data-class opt-in consent
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from circuitforge_core.sync.db import SyncDB
from circuitforge_core.sync.models import SyncBlob, SyncConfig, SyncPref

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# #57 — consent layer (checked before writes in SyncStore)
# ---------------------------------------------------------------------------


class SyncPrefsStore:
    """Read and update per-user, per-data-class sync consent preferences.

    Absence of a row always means disabled — this is a schema invariant.
    No code in this module inserts a row with enabled=1 except set_sync_pref().
    """

    def __init__(self, db: SyncDB) -> None:
        self._db = db

    def get_sync_prefs(self, user_id: str, product: str) -> dict[str, bool]:
        """Return {data_class: enabled} for all known prefs for user+product.

        Missing data_classes are not included (absence = disabled).
        """
        conn = self._db.connect()
        try:
            rows = conn.execute(
                "SELECT data_class, enabled FROM sync_prefs "
                "WHERE user_id = ? AND product = ?",
                (user_id, product),
            ).fetchall()
            return {row[0]: bool(row[1]) for row in rows}
        finally:
            conn.close()

    def get_all_sync_prefs(self, user_id: str) -> dict[str, dict[str, bool]]:
        """Return {product: {data_class: enabled}} across all products for a user."""
        conn = self._db.connect()
        try:
            rows = conn.execute(
                "SELECT product, data_class, enabled FROM sync_prefs WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            result: dict[str, dict[str, bool]] = {}
            for product, data_class, enabled in rows:
                result.setdefault(product, {})[data_class] = bool(enabled)
            return result
        finally:
            conn.close()

    def is_enabled(self, user_id: str, product: str, data_class: str) -> bool:
        """Return True only if the user has explicitly opted in to this data_class."""
        conn = self._db.connect()
        try:
            row = conn.execute(
                "SELECT enabled FROM sync_prefs "
                "WHERE user_id = ? AND product = ? AND data_class = ?",
                (user_id, product, data_class),
            ).fetchone()
            return bool(row[0]) if row else False
        finally:
            conn.close()

    def set_sync_pref(
        self, user_id: str, product: str, data_class: str, enabled: bool
    ) -> SyncPref:
        """Set enabled/disabled for a single data_class. Returns the updated pref."""
        now = _utcnow()
        conn = self._db.connect()
        try:
            conn.execute(
                """
                INSERT INTO sync_prefs (user_id, product, data_class, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (user_id, product, data_class)
                DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at
                """,
                (user_id, product, data_class, int(enabled), now),
            )
            conn.commit()
        finally:
            conn.close()
        return SyncPref(
            product=product, data_class=data_class, enabled=enabled, updated_at=now
        )

    def wipe_sync_data(
        self, user_id: str, product: str | None = None, data_class: str | None = None
    ) -> None:
        """Delete synced blobs and reset prefs. Deletion is immediate and irrevocable.

        - product=None: wipe everything for the user across all products
        - data_class=None: wipe all data_classes for the given product
        """
        conn = self._db.connect()
        try:
            if product is None:
                conn.execute("DELETE FROM sync_blobs WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM sync_prefs WHERE user_id = ?", (user_id,))
            elif data_class is None:
                conn.execute(
                    "DELETE FROM sync_blobs WHERE user_id = ? AND product = ?",
                    (user_id, product),
                )
                conn.execute(
                    "DELETE FROM sync_prefs WHERE user_id = ? AND product = ?",
                    (user_id, product),
                )
            else:
                conn.execute(
                    "DELETE FROM sync_blobs WHERE user_id = ? AND product = ? AND data_class = ?",
                    (user_id, product, data_class),
                )
                conn.execute(
                    "DELETE FROM sync_prefs WHERE user_id = ? AND product = ? AND data_class = ?",
                    (user_id, product, data_class),
                )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# #56 — blob storage layer
# ---------------------------------------------------------------------------


class SyncStore:
    """Push, pull, and delete opaque localStorage blobs.

    Consent is checked before every write — a disabled data_class is silently
    rejected rather than raising, keeping callers simple.
    """

    def __init__(self, db: SyncDB, prefs: SyncPrefsStore) -> None:
        self._db = db
        self._prefs = prefs

    def push(
        self, user_id: str, product: str, data_class: str, blob: str, updated_at: str
    ) -> bool:
        """Store or update a blob. Returns True if written, False if rejected.

        Rejected when:
        - data_class consent is disabled for this user/product
        - client's updated_at is older than the stored updated_at (last-write-wins)
        """
        if not self._prefs.is_enabled(user_id, product, data_class):
            logger.debug("sync.push rejected: %s/%s not consented", product, data_class)
            return False

        conn = self._db.connect()
        try:
            conn.execute(
                """
                INSERT INTO sync_blobs (user_id, product, data_class, blob, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (user_id, product, data_class)
                DO UPDATE SET blob = excluded.blob, updated_at = excluded.updated_at
                WHERE excluded.updated_at > sync_blobs.updated_at
                """,
                (user_id, product, data_class, blob, updated_at),
            )
            conn.commit()
        finally:
            conn.close()
        return True

    def pull(
        self, user_id: str, product: str, data_classes: list[str] | None = None
    ) -> list[SyncBlob]:
        """Return blobs for consented data_classes.

        data_classes=None returns all consented classes for this user/product.
        """
        prefs = self._prefs.get_sync_prefs(user_id, product)
        enabled = {dc for dc, on in prefs.items() if on}
        if data_classes is not None:
            enabled = enabled & set(data_classes)

        if not enabled:
            return []

        placeholders = ",".join("?" * len(enabled))
        conn = self._db.connect()
        try:
            rows = conn.execute(
                f"SELECT product, data_class, blob, updated_at FROM sync_blobs "
                f"WHERE user_id = ? AND product = ? AND data_class IN ({placeholders})",
                (user_id, product, *enabled),
            ).fetchall()
            return [
                SyncBlob(
                    product=row[0],
                    data_class=row[1],
                    blob=row[2],
                    updated_at=row[3],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def delete(self, user_id: str, product: str, data_class: str) -> None:
        """Delete a single blob. Tier-free — always allowed regardless of consent state."""
        conn = self._db.connect()
        try:
            conn.execute(
                "DELETE FROM sync_blobs WHERE user_id = ? AND product = ? AND data_class = ?",
                (user_id, product, data_class),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_all(self, user_id: str, product: str) -> None:
        """Delete all blobs for user+product. Does not touch prefs."""
        conn = self._db.connect()
        try:
            conn.execute(
                "DELETE FROM sync_blobs WHERE user_id = ? AND product = ?",
                (user_id, product),
            )
            conn.commit()
        finally:
            conn.close()
