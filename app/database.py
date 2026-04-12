"""
Database connection management for the Context Broker.

Manages asyncpg (PostgreSQL) connection pool. PostgreSQL is the only
external database — it handles data storage, vector search (pgvector),
job state (DB-driven workers), and advisory locks.
"""

import base64
import logging
import os
from typing import Optional

import asyncpg
import httpx

_log = logging.getLogger("emad_host.database")

_pg_pool: Optional[asyncpg.Pool] = None


async def init_postgres(config: dict) -> asyncpg.Pool:
    """Create the asyncpg connection pool using config and environment variables."""
    global _pg_pool

    # CR-M06: Close existing pool to prevent connection leak on re-init
    if _pg_pool is not None:
        _log.info("Closing existing PostgreSQL pool before re-initialization")
        await _pg_pool.close()

    db_config = config.get("database", {})
    password = os.environ.get("POSTGRES_PASSWORD", "")

    _pg_pool = await asyncpg.create_pool(
        host=os.environ.get("POSTGRES_HOST", "emad-host-postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "emad_host"),
        user=os.environ.get("POSTGRES_USER", "emad_host"),
        password=password,
        min_size=db_config.get("pool_min_size", 2),
        max_size=db_config.get("pool_max_size", 10),
        command_timeout=db_config.get("command_timeout", 30),
    )
    _log.info("PostgreSQL connection pool initialized")
    return _pg_pool


def get_pg_pool() -> asyncpg.Pool:
    """Return the initialized asyncpg pool. Raises if not initialized."""
    if _pg_pool is None:
        raise RuntimeError(
            "PostgreSQL pool not initialized — call init_postgres() first"
        )
    return _pg_pool


async def close_all_connections() -> None:
    """Gracefully close all database connections."""
    global _pg_pool

    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None
        _log.info("PostgreSQL pool closed")


async def check_postgres_health() -> bool:
    """Check PostgreSQL connectivity for health endpoint."""
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except (asyncpg.PostgresError, OSError, RuntimeError) as exc:
        _log.warning("PostgreSQL health check failed: %s", exc)
        return False


async def check_neo4j_health(config: dict | None = None) -> bool:
    """Check Neo4j connectivity for health endpoint.

    Probes the HTTP endpoint (port 7474) — Neo4j's built-in health endpoint.
    """
    neo4j_host = os.environ.get("NEO4J_HOST", "emad-host-neo4j")
    neo4j_http_port = os.environ.get("NEO4J_HTTP_PORT", "7474")
    url = f"http://{neo4j_host}:{neo4j_http_port}/"

    headers = {}
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
    if neo4j_password:
        credentials = base64.b64encode(f"neo4j:{neo4j_password}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url, headers=headers)
            return response.status_code == 200
    except (httpx.HTTPError, OSError) as exc:
        _log.warning("Neo4j health check failed: %s", exc)
        return False
