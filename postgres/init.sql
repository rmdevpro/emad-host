-- State 4 pMAD — PostgreSQL schema
-- Loaded automatically by postgres entrypoint on first run.
-- Requires: pgvector extension (pre-installed in pgvector/pgvector:pg16 image)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- Schema versioning
-- Applied and checked at application startup.
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_migrations (version, description)
VALUES (1, 'Initial schema')
ON CONFLICT (version) DO NOTHING;

-- ============================================================
-- System logs (log shipper container writes here)
-- Enables Imperator to query logs from all MAD containers.
-- The log shipper discovers containers via Docker API on
-- emad-host-net and writes their logs with resolved names.
-- ============================================================

CREATE TABLE IF NOT EXISTS system_logs (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    container_name  VARCHAR(255) NOT NULL,
    log_timestamp   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    message         TEXT,
    data            JSONB
);

CREATE INDEX IF NOT EXISTS idx_system_logs_container_time
    ON system_logs (container_name, log_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_logs_time
    ON system_logs (log_timestamp DESC);
