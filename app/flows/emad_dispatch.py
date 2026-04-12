"""
eMAD chat dispatch flow.

StateGraph: lookup_instance -> build_and_invoke -> extract_response

Looks up a named eMAD instance in Postgres, builds its StateGraph from
the registered package, invokes it, and extracts the response.

Adapted from kaiser-langgraph/flows/dispatch.py for the emad-host template.
Uses app.database for DB access, app.emad_registry for package lookup.
"""

import asyncio
import logging
from typing import TypedDict

import asyncpg
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app import emad_registry
from app.database import get_pg_pool

_log = logging.getLogger("emad_host.flows.emad_dispatch")

# Coordinated with nginx proxy_read_timeout
_INVOKE_TIMEOUT = 1800.0


class EmadChatState(TypedDict):
    emad_name: str
    conversation_id: str
    message: str
    instance_row: dict | None
    emad_result: dict | None
    response: str | None
    rogers_conversation_id: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def lookup_instance(state: EmadChatState) -> EmadChatState:
    """Query emad_instances by emad_name. Validate package is registered.

    Sets error on failure.
    """
    emad_name = state["emad_name"]
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT emad_name, package_name, parameters, status "
                "FROM emad_instances WHERE emad_name = $1",
                emad_name,
            )
    except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError) as exc:
        _log.error("lookup_instance: DB error: %s", exc)
        return {**state, "error": f"DB_ERROR: {exc}"}

    if row is None:
        return {**state, "error": f"EMAD_NOT_FOUND: {emad_name}"}
    if row["status"] != "active":
        return {**state, "error": f"EMAD_DISABLED: {emad_name}"}

    package_name = row["package_name"]
    if emad_registry.get_build_func(package_name) is None:
        return {
            **state,
            "error": f"PACKAGE_NOT_LOADED: {package_name} — "
            "install it with emad_install_package",
        }

    return {**state, "instance_row": dict(row), "error": None}


async def build_and_invoke(state: EmadChatState) -> EmadChatState:
    """Build the eMAD StateGraph from the registered package, then invoke it.

    Uses asyncio.wait_for to enforce the timeout.
    Validates build_graph output before invoking (ERQ-004 §1.2).
    """
    row = state["instance_row"]
    package_name = row["package_name"]
    parameters = dict(row["parameters"]) if row["parameters"] else {}

    build_func = emad_registry.get_build_func(package_name)

    # Validate host contract: build_graph must return a compiled StateGraph
    try:
        graph = build_func(parameters)
    except (TypeError, ImportError, RuntimeError) as exc:
        _log.error("build_and_invoke: build_graph(%s) failed: %s", package_name, exc)
        return {**state, "error": f"BUILD_ERROR: {exc}"}

    if not hasattr(graph, "ainvoke"):
        _log.error(
            "build_and_invoke: build_graph(%s) returned %s — not a compiled StateGraph",
            package_name,
            type(graph).__name__,
        )
        return {
            **state,
            "error": f"BUILD_CONTRACT_VIOLATION: {package_name} build_graph() "
            "must return a compiled StateGraph",
        }

    # Pass AE config to eMAD so it can resolve inference providers
    # without importing host internals (ERQ-002 §13.2, ERQ-004 §4.2)
    from app.config import load_merged_config

    ae_config = load_merged_config()

    # Initial state per ERQ-004 §2.1
    initial_state = {
        "messages": [HumanMessage(content=state["message"])],
        "rogers_conversation_id": state["conversation_id"],
        "rogers_context_window_id": None,
        "config": ae_config,
    }

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=_INVOKE_TIMEOUT,
        )
        return {**state, "emad_result": result}
    except asyncio.TimeoutError:
        _log.error(
            "build_and_invoke: eMAD %s timed out after %.0fs",
            package_name,
            _INVOKE_TIMEOUT,
        )
        return {
            **state,
            "error": f"EMAD_TIMEOUT: {package_name} exceeded {_INVOKE_TIMEOUT}s",
        }
    except (RuntimeError, ValueError, TypeError) as exc:
        _log.error("build_and_invoke: eMAD %s raised: %s", package_name, exc)
        return {**state, "error": f"EMAD_EXECUTION_ERROR: {exc}"}


def extract_response(state: EmadChatState) -> EmadChatState:
    """Extract final_response from eMAD result (ERQ-004 §2.2), or surface the error."""
    if state.get("error"):
        return {
            **state,
            "response": f"[ERROR: {state['error']}]",
            "rogers_conversation_id": None,
        }

    result = state.get("emad_result") or {}
    # ERQ-004 §2.2: eMAD must include final_response in output state
    response = (
        result.get("final_response")
        or result.get("last_observation")
        or "[No response from eMAD]"
    )
    # Surface the conversation_id created by the eMAD so the caller
    # can pass it on subsequent turns for session continuity.
    rogers_conv_id = (
        result.get("rogers_conversation_id") or state.get("conversation_id") or None
    )
    return {**state, "response": response, "rogers_conversation_id": rogers_conv_id}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_lookup(state: EmadChatState) -> str:
    return "extract_response" if state.get("error") else "build_and_invoke"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_emad_chat_flow() -> StateGraph:
    """Build and compile the eMAD chat dispatch StateGraph."""
    g = StateGraph(EmadChatState)
    g.add_node("lookup_instance", lookup_instance)
    g.add_node("build_and_invoke", build_and_invoke)
    g.add_node("extract_response", extract_response)

    g.set_entry_point("lookup_instance")
    g.add_conditional_edges(
        "lookup_instance",
        _route_after_lookup,
        {
            "extract_response": "extract_response",
            "build_and_invoke": "build_and_invoke",
        },
    )
    g.add_edge("build_and_invoke", "extract_response")
    g.add_edge("extract_response", END)

    return g.compile()
