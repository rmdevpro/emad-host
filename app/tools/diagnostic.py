"""Diagnostic tools — log querying.

AE tool for querying container logs collected by the log shipper.
"""

import logging

import asyncpg
from app.database import get_pg_pool
from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.diagnostic")


@tool
async def log_query(
    container: str = "", level: str = "", search: str = "", limit: int = 50
) -> str:
    """Query MAD container logs from the system_logs table.

    Logs are collected by the log shipper from all containers on
    the pMAD's network and stored in Postgres with resolved names.

    Args:
        container: Filter by container name (e.g., "langgraph", "postgres"). Empty = all.
        level: Filter by log level in the structured data (e.g., "ERROR"). Empty = all.
        search: Text search in message content. Empty = no filter.
        limit: Maximum entries to return (default 50, max 200).
    """
    try:
        pool = get_pg_pool()
        conditions = []
        args: list = []
        idx = 1

        if container:
            conditions.append(f"container_name ILIKE ${idx}")
            args.append(f"%{container}%")
            idx += 1
        if level:
            conditions.append(f"data->>'level' = ${idx}")
            args.append(level.upper())
            idx += 1
        if search:
            conditions.append(f"message ILIKE ${idx}")
            args.append(f"%{search}%")
            idx += 1

        where = " AND ".join(conditions) if conditions else "1=1"
        args.append(min(limit, 200))

        rows = await pool.fetch(
            f"""
            SELECT container_name, log_timestamp, message, data
            FROM system_logs
            WHERE {where}
            ORDER BY log_timestamp DESC
            LIMIT ${idx}
            """,
            *args,
        )
        if not rows:
            return "No log entries found matching the filters."
        lines = []
        for row in rows:
            ts = row["log_timestamp"].isoformat() if row["log_timestamp"] else "?"
            data = row["data"] if isinstance(row["data"], dict) else {}
            lvl = data.get("level", "?")
            msg = row["message"] or str(row["data"] or "")[:200]
            lines.append(f"[{ts}] [{row['container_name']}] [{lvl}] {msg}")
        return "\n".join(lines)
    except (asyncpg.PostgresError, OSError, KeyError) as exc:
        return f"Log query error: {exc}"


def get_tools() -> list:
    """Return all diagnostic tools."""
    return [log_query]
