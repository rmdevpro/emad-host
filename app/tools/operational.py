"""Operational tools — domain information management.

AE tools for storing and searching domain information.
Embedding is handled automatically by the AE via Postgres trigger + LISTEN/NOTIFY.
The TE just calls these tools — it doesn't handle embedding.
"""

import logging

import asyncpg
from app.config import async_load_config, get_embeddings_model
from app.database import get_pg_pool
from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.operational")


@tool
async def store_domain_info(content: str, source: str = "host") -> str:
    """Store a piece of domain information — a learned fact, procedure, or preference.

    Use this when you learn something operationally important that should
    persist across conversations. Examples: deployment procedures, user
    preferences, configuration patterns, performance characteristics.

    The embedding is generated automatically by the AE — just store the text.

    Args:
        content: The information to store. Be specific and actionable.
        source: Who is storing this (default: "host"). Use the agent name
                for agent-specific knowledge, or a tag like "<policy>" for
                shared knowledge visible to all agents.
    """
    try:
        pool = get_pg_pool()
        await pool.execute(
            """
            INSERT INTO domain_information (content, source)
            VALUES ($1, $2)
            """,
            content,
            source,
        )
        return f"Stored domain information: {content[:100]}..."
    except (asyncpg.PostgresError, OSError, RuntimeError) as exc:
        return f"Failed to store domain information: {exc}"


@tool
async def search_domain_info(query: str, source: str = "host", limit: int = 5) -> str:
    """Search stored domain information using semantic similarity.

    Use this to recall learned procedures, preferences, or operational
    facts from past conversations.

    Args:
        query: What to search for (natural language).
        source: Agent name to scope results. Also includes any knowledge
                tags (e.g., "<policy>") configured for this agent.
        limit: Maximum results to return (default 5).
    """
    try:
        config = await async_load_config()
        pool = get_pg_pool()

        # Embed the query
        emb_model = get_embeddings_model(config)
        query_vec = await emb_model.aembed_query(query)
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        # Search scoped to this agent + shared tags (source starting with <)
        rows = await pool.fetch(
            """
            SELECT id, content, source, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM domain_information
            WHERE embedding IS NOT NULL
              AND (source = $2 OR source LIKE '<%>')
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vec_str,
            source,
            limit,
        )

        if not rows:
            return "No domain information found matching that query."

        lines = [f"Found {len(rows)} relevant domain information entries:"]
        for row in rows:
            sim = round(float(row["similarity"]), 3)
            ts = row["created_at"].strftime("%Y-%m-%d") if row["created_at"] else "?"
            lines.append(f"- [{sim}] ({ts}, {row['source']}) {row['content']}")
        return "\n".join(lines)
    except (asyncpg.PostgresError, OSError, RuntimeError, ValueError) as exc:
        return f"Domain information search error: {exc}"
