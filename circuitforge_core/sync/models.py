"""Data models for the cf-core sync module.

MIT licensed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncConfig:
    """Config for the sync module's SQLite DB."""

    db_path: Path
    product: str

    @classmethod
    def from_env(cls, product: str) -> SyncConfig:
        """Build config from environment.

        Variables:
            SYNC_DB_PATH    — full path to sync.db (default: data dir / sync.db)
            CLOUD_DATA_ROOT — base dir for per-user cloud data (used when SYNC_DB_PATH unset)
        """
        explicit = os.environ.get("SYNC_DB_PATH")
        if explicit:
            db_path = Path(explicit)
        else:
            base = Path(os.environ.get("CLOUD_DATA_ROOT", "/devl/cf-data"))
            db_path = base / "sync.db"
        return cls(db_path=db_path, product=product)


@dataclass(frozen=True)
class SyncBlob:
    """An opaque sync blob returned from the store."""

    product: str
    data_class: str
    blob: str
    updated_at: str


@dataclass(frozen=True)
class SyncPref:
    """A single sync preference entry."""

    product: str
    data_class: str
    enabled: bool
    updated_at: str
