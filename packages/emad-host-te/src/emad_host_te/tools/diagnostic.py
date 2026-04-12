"""Diagnostic tools — always available to the Imperator.

Read-only observation tools for logs, context, pipeline status.
"""

import logging

import asyncpg
from langchain_core.tools import tool

from emad_host_te._ctx import get_ctx

_log = logging.getLogger("emad_host.tools.diagnostic")


@tool
async def log_query(
    container: str = "", level: str = "", search: str = "", limit: int = 50
) -> str:
    """Query MAD container logs from the system_logs table.

    Logs are collected by the log shipper from all containers on
    emad-host-net and stored in Postgres with resolved names.

    Args:
        container: Filter by container name (e.g., "langgraph", "postgres"). Empty = all.
        level: Filter by log level in the structured data (e.g., "ERROR"). Empty = all.
        search: Text search in message content. Empty = no filter.
        limit: Maximum entries to return (default 50, max 200).
    """
    try:
        pool = get_ctx().get_pool()
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


@tool
async def context_introspection(
    conversation_id: str, build_type: str = "tiered-summary"
) -> str:
    """Show the assembled context breakdown for a conversation.

    Displays tier allocation, token usage, summary counts, and last
    assembly time for the specified conversation and build type.

    Args:
        conversation_id: The conversation to inspect.
        build_type: The build type to inspect (default: tiered-summary).
    """
    import uuid as _uuid

    try:
        pool = get_ctx().get_pool()

        window = await pool.fetchrow(
            """
            SELECT id, build_type, max_token_budget, last_assembled_at, created_at
            FROM context_windows
            WHERE conversation_id = $1 AND build_type = $2
            ORDER BY created_at DESC LIMIT 1
            """,
            _uuid.UUID(conversation_id),
            build_type,
        )
        if not window:
            return (
                f"No context window found for conversation {conversation_id} "
                f"with build type {build_type}"
            )

        summary_count = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_summaries WHERE context_window_id = $1",
            window["id"],
        )

        msg_count = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = $1",
            _uuid.UUID(conversation_id),
        )

        total_tokens = await pool.fetchval(
            "SELECT COALESCE(SUM(token_count), 0) FROM conversation_messages "
            "WHERE conversation_id = $1",
            _uuid.UUID(conversation_id),
        )

        eu_default = get_ctx().effective_utilization_default

        effective = int(window["max_token_budget"] * eu_default)

        lines = [
            f"Context Window: {window['id']}",
            f"Build Type: {window['build_type']}",
            f"Raw Budget: {window['max_token_budget']} tokens",
            f"Effective Budget ({int(eu_default * 100)}%): {effective} tokens",
            f"Total Messages: {msg_count}",
            f"Total Message Tokens: {total_tokens}",
            f"Summaries: {summary_count}",
            f"Last Assembled: {window['last_assembled_at'].isoformat() if window['last_assembled_at'] else 'never'}",
            f"Created: {window['created_at'].isoformat() if window['created_at'] else '?'}",
        ]
        return "\n".join(lines)
    except (asyncpg.PostgresError, OSError, KeyError) as exc:
        return f"Introspection error: {exc}"


@tool
async def pipeline_status() -> str:
    """Show the status of background processing pipelines.

    Displays queue depths for embedding, assembly, and extraction jobs,
    plus recent activity.
    """
    try:
        pool = get_ctx().get_pool()

        pending_embed = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages "
            "WHERE embedding IS NULL AND content IS NOT NULL"
        )
        pending_extract = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages "
            "WHERE memory_extracted IS NOT TRUE"
        )

        recent_assembly = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_summaries "
            "WHERE created_at > NOW() - INTERVAL '1 hour'"
        )
        recent_embeddings = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages "
            "WHERE embedding IS NOT NULL AND created_at > NOW() - INTERVAL '1 hour'"
        )

        total_messages = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages"
        )
        total_embedded = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages WHERE embedding IS NOT NULL"
        )

        lines = [
            "Pipeline Status (DB-driven):",
            f"  Pending embedding: {pending_embed} messages",
            f"  Pending extraction: {pending_extract} messages",
            f"  Total messages: {total_messages}",
            f"  Total embedded: {total_embedded}",
            "",
            "Recent Activity (last hour):",
            f"  Summaries created: {recent_assembly}",
            f"  Messages embedded: {recent_embeddings}",
        ]
        return "\n".join(lines)
    except (asyncpg.PostgresError, OSError, KeyError) as exc:
        return f"Pipeline status error: {exc}"


def get_tools() -> list:
    """Return all diagnostic tools."""
    return [log_query, context_introspection, pipeline_status]
