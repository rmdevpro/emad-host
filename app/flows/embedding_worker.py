"""
Embedding worker — auto-embeds domain_information rows.

Small stategraph: fetch_row → embed → store_vector.
Invoked by the LISTEN callback when Postgres fires NOTIFY domain_info_new.
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

_log = logging.getLogger("emad_host.flows.embedding_worker")


class EmbedState(TypedDict):
    row_id: str
    content: Optional[str]
    error: Optional[str]


async def fetch_row(state: EmbedState) -> dict:
    """Fetch the domain_information row by ID."""
    from app.database import get_pg_pool

    try:
        pool = get_pg_pool()
        row = await pool.fetchrow(
            "SELECT id, content FROM domain_information WHERE id = $1::uuid",
            state["row_id"],
        )
        if row is None:
            return {"error": f"Row {state['row_id']} not found"}
        return {"content": row["content"]}
    except (OSError, RuntimeError) as exc:
        return {"error": str(exc)}


async def embed_and_store(state: EmbedState) -> dict:
    """Generate embedding and write it back to the row."""
    if state.get("error"):
        return {}

    from app.config import async_load_config, get_embeddings_model

    try:
        config = await async_load_config()
        emb_model = get_embeddings_model(config)
        vector = await emb_model.aembed_query(state["content"])
        vec_str = "[" + ",".join(str(v) for v in vector) + "]"

        from app.database import get_pg_pool

        pool = get_pg_pool()
        await pool.execute(
            "UPDATE domain_information SET embedding = $1::vector WHERE id = $2::uuid",
            vec_str,
            state["row_id"],
        )
        _log.info("Embedded domain_information row %s", state["row_id"])
        return {}
    except (OSError, RuntimeError, ValueError) as exc:
        _log.error("Embedding failed for row %s: %s", state["row_id"], exc)
        return {"error": str(exc)}


def build_embedding_worker_flow():
    """Build the embedding worker stategraph."""
    g = StateGraph(EmbedState)
    g.add_node("fetch_row", fetch_row)
    g.add_node("embed_and_store", embed_and_store)

    g.set_entry_point("fetch_row")
    g.add_edge("fetch_row", "embed_and_store")
    g.add_edge("embed_and_store", END)

    return g.compile()


# Singleton compiled flow
_flow = None


async def embed_row(row_id: str) -> None:
    """Embed a single domain_information row. Called by LISTEN callback."""
    global _flow
    if _flow is None:
        _flow = build_embedding_worker_flow()

    await _flow.ainvoke({"row_id": row_id, "content": None, "error": None})
