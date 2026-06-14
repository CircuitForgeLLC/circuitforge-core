"""cf-core sync module — opt-in localStorage sync for CircuitForge products.

MIT licensed. Tickets: cf-core #56 (storage) + #57 (consent layer).

## Quick start

### 1. Mount the router (product side)

    from circuitforge_core.sync import make_sync_router, SyncConfig

    sync_router = make_sync_router(
        product="kiwi",
        get_session=_sessions.dependency(),
        require_paid=_sessions.require_tier("paid"),
        config=SyncConfig.from_env("kiwi"),
    )
    app.include_router(sync_router, prefix="/sync", tags=["sync"])

### 2. Endpoints

    GET  /sync/prefs                    — return consent preferences for this user+product
    PATCH /sync/prefs                   — {data_class, enabled} — opt in/out
    POST  /sync/push                    — {data_class, blob, updated_at} — push blob (Paid+)
    GET   /sync/pull                    — return all consented blobs (Paid+)
    DELETE /sync/blob/{data_class}      — delete one blob (any tier)
    DELETE /sync/data                   — wipe all blobs, keep prefs (any tier)
    DELETE /sync/all                    — wipe blobs + prefs (any tier)

### 3. Privacy invariants
    - All sync prefs default to disabled; server never writes enabled=True except on
      explicit PATCH from the authenticated user.
    - Server stores blobs as opaque TEXT — content is never deserialized server-side.
    - Delete endpoints are tier-free — users can always delete their data.
    - Deletion is immediate and irrevocable.

### 4. Install

    pip install -e ../circuitforge-core[sync]

### 5. Env vars

    SYNC_DB_PATH        — path to sync.db (default: CLOUD_DATA_ROOT/sync.db)
    CLOUD_DATA_ROOT     — base dir for per-user cloud data (default: /devl/cf-data)
    SYNC_DB_KEY         — SQLCipher encryption key (optional; enables at-rest encryption)
"""

from circuitforge_core.sync.db import SyncDB
from circuitforge_core.sync.models import SyncBlob, SyncConfig, SyncPref
from circuitforge_core.sync.router import make_sync_router
from circuitforge_core.sync.store import SyncPrefsStore, SyncStore

__all__ = [
    "make_sync_router",
    "SyncDB",
    "SyncStore",
    "SyncPrefsStore",
    "SyncConfig",
    "SyncBlob",
    "SyncPref",
]
