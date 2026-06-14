-- 001_sync.sql
-- MIT License — data layer, no inference
--
-- Opt-in localStorage sync: opaque blob store + per-data-class consent prefs.
-- Server never parses blob content.

-- #56: opaque blob store keyed by (user_id, product, data_class)
CREATE TABLE IF NOT EXISTS sync_blobs (
    user_id     TEXT NOT NULL,
    product     TEXT NOT NULL,
    data_class  TEXT NOT NULL,
    blob        TEXT NOT NULL,          -- raw JSON string; server never deserializes
    updated_at  TEXT NOT NULL,          -- ISO-8601 UTC; basis for last-write-wins
    PRIMARY KEY (user_id, product, data_class)
);
CREATE INDEX IF NOT EXISTS idx_sync_blobs_user ON sync_blobs (user_id, product);

-- #57: consent prefs — absence of a row means disabled (opt-in by default)
CREATE TABLE IF NOT EXISTS sync_prefs (
    user_id     TEXT NOT NULL,
    product     TEXT NOT NULL,
    data_class  TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 0,  -- 0 = off; rows are never inserted enabled
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, product, data_class)
);
