"""Operational tools — domain information management.

Available when domain_information.enabled: true in TE config.
These tools are TE-owned: the Imperator decides what to store.
"""

import logging

import asyncpg
from langchain_core.tools import tool

from emad_host_te._ctx import get_ctx

_log = logging.getLogger("emad_host.tools.operational")


@tool
async def store_domain_info(content: str, source: str = "imperator") -> str:
    """Store a piece of domain information — a learned fact, procedure, or preference.

    Use this when you learn something operationally important that should
    persist across conversations. Examples: deployment procedures, user
    preferences, configuration patterns, performance characteristics.

    Args:
        content: The information to store. Be specific and actionable.
        source: How this was learned (default: "imperator").
    """
    try:
        ctx = get_ctx()
        config = await ctx.async_load_config()
        pool = ctx.get_pool()

        # Embed the content
        emb_model = ctx.get_embeddings_model(config)
        vector = await emb_model.aembed_query(content)
        vec_str = "[" + ",".join(str(v) for v in vector) + "]"

        await pool.execute(
            """
            INSERT INTO domain_information (content, embedding, source)
            VALUES ($1, $2::vector, $3)
            """,
            content,
            vec_str,
            source,
        )
        return f"Stored domain information: {content[:100]}..."
    except (asyncpg.PostgresError, OSError, RuntimeError, ValueError) as exc:
        return f"Failed to store domain information: {exc}"


@tool
async def search_domain_info(query: str, limit: int = 5) -> str:
    """Search stored domain information using semantic similarity.

    Use this to recall learned procedures, preferences, or operational
    facts from past conversations.

    Args:
        query: What to search for (natural language).
        limit: Maximum results to return (default 5).
    """
    try:
        ctx = get_ctx()
        config = await ctx.async_load_config()
        pool = ctx.get_pool()

        # Embed the query
        emb_model = ctx.get_embeddings_model(config)
        query_vec = await emb_model.aembed_query(query)
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        rows = await pool.fetch(
            """
            SELECT content, source, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM domain_information
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vec_str,
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


@tool
async def extract_domain_knowledge(content: str = "") -> str:
    """Extract structured knowledge from recent domain information into the knowledge graph.

    Processes unextracted domain information entries through Mem0, which
    extracts entities and relationships into Neo4j. Call without arguments
    to process all pending entries, or pass specific content to extract.

    Args:
        content: Optional specific content to extract. Empty = process pending entries.
    """
    try:
        ctx = get_ctx()
        config = await ctx.async_load_config()
        from emad_host_te.domain_mem0 import get_domain_mem0

        mem0 = await get_domain_mem0(config)
        if mem0 is None:
            return "Domain Mem0 client not available."

        pool = ctx.get_pool()

        if content:
            # Extract specific content
            import asyncio

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: mem0.add(
                    content, user_id="domain", metadata={"source": "manual"}
                ),
            )
            return f"Extracted knowledge from provided content. Result: {result}"

        # Process unextracted domain_information entries.
        # domain_memories table is created dynamically by Mem0 on first insert —
        # check if it exists before querying.
        table_exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'domain_memories')"
        )
        if table_exists:
            rows = await pool.fetch("""
                SELECT id, content FROM domain_information
                WHERE id NOT IN (
                    SELECT DISTINCT (metadata->>'domain_info_id')::uuid
                    FROM domain_memories
                    WHERE metadata->>'domain_info_id' IS NOT NULL
                )
                LIMIT 10
                """)
        else:
            # No domain_memories yet — all domain_information entries are pending
            rows = await pool.fetch(
                "SELECT id, content FROM domain_information LIMIT 10"
            )

        if not rows:
            return "No pending domain information entries to extract."

        import asyncio

        extracted = 0
        for row in rows:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda r=row: mem0.add(
                        r["content"],
                        user_id="domain",
                        metadata={"domain_info_id": str(r["id"])},
                    ),
                )
                extracted += 1
            except (RuntimeError, OSError, ValueError) as exc:
                _log.warning("Failed to extract domain info %s: %s", row["id"], exc)

        return f"Extracted knowledge from {extracted}/{len(rows)} domain information entries."
    except (asyncpg.PostgresError, OSError, RuntimeError, ValueError) as exc:
        return f"Domain knowledge extraction error: {exc}"


@tool
async def search_domain_knowledge(query: str, limit: int = 5) -> str:
    """Search the domain knowledge graph for structured facts and relationships.

    Use this for relational queries — "what depends on X?", "what is related to Y?".
    For semantic search over raw domain info text, use search_domain_info instead.

    Args:
        query: What to search for (natural language).
        limit: Maximum results to return (default 5).
    """
    try:
        ctx = get_ctx()
        config = await ctx.async_load_config()
        from emad_host_te.domain_mem0 import get_domain_mem0

        mem0 = await get_domain_mem0(config)
        if mem0 is None:
            return "Domain Mem0 client not available."

        import asyncio

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: mem0.search(query, user_id="domain", limit=limit),
        )

        memories = results.get("results", []) if isinstance(results, dict) else results
        if not memories:
            return "No domain knowledge found matching that query."

        lines = [f"Found {len(memories)} domain knowledge entries:"]
        for mem in memories:
            fact = mem.get("memory") or mem.get("content") or str(mem)
            lines.append(f"- {fact}")
        return "\n".join(lines)
    except (RuntimeError, OSError, ValueError) as exc:
        return f"Domain knowledge search error: {exc}"


def get_tools(te_config: dict | None = None) -> list:
    """Return operational tools based on TE config.

    Args:
        te_config: TE configuration dict. Tools are conditionally included
                   based on feature flags.
    """
    tools = []
    if te_config and te_config.get("domain_information", {}).get("enabled", True):
        tools.extend([store_domain_info, search_domain_info])
    if te_config and te_config.get("domain_knowledge", {}).get("enabled", False):
        tools.extend([extract_domain_knowledge, search_domain_knowledge])
    return tools
