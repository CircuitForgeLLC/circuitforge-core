"""SQLite connection + migration runner for the sync module.

MIT licensed.
"""
from __future__ import annotations

import importlib.resources
import logging
import sqlite3
from pathlib import Path

from circuitforge_core.db.base import get_connection
from circuitforge_core.db.migrations import run_migrations

logger = logging.getLogger(__name__)


class SyncDB:
    """Manages the sync SQLite database: connection + migrations.

    Usage:
        db = SyncDB(Path("/devl/cf-data/sync.db"))
        db.run_migrations()
        conn = db.connect()
        ...
        conn.close()
    """

    def __init__(self, db_path: Path, key: str = "") -> None:
        self._db_path = db_path
        self._key = key
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Open a connection to the sync database."""
        return get_connection(self._db_path, self._key)

    def run_migrations(self) -> None:
        """Apply any unapplied sync migrations. Safe to call on every startup."""
        conn = self.connect()
        try:
            migrations_dir = Path(
                str(importlib.resources.files("circuitforge_core.sync.migrations"))
            )
            run_migrations(conn, migrations_dir)
            logger.debug("Sync migrations applied from %s", migrations_dir)
        finally:
            conn.close()
