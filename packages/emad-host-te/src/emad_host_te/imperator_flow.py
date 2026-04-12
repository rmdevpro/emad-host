"""
Imperator — LangGraph ReAct-style conversational agent flow.

The Imperator is the pMAD's built-in conversational agent.
It uses a proper LangGraph ReAct graph (agent_node -> tool_node loop).

Supports two modes controlled by emad_host config in te.yml:
- CB mode: loads context via external Context Broker (get_context, store_message, CEAc)
- No-CB mode: uses LangGraph trim_messages for context window management

ARCH-05: ReAct loop is graph edges, not a while loop inside a node.
ARCH-06: In-memory MemorySaver is used as the LangGraph checkpointer for
         the ReAct loop's multi-step tool calling cycle.
"""

import logging
import socket
import uuid
from typing import Annotated, Optional

import httpx
import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from emad_host_te._ctx import get_ctx

from emad_host_te.tools.admin import get_tools as get_admin_tools
from emad_host_te.tools.alerting import get_tools as get_alerting_tools
from emad_host_te.tools.diagnostic import get_tools as get_diagnostic_tools
from emad_host_te.tools.filesystem import get_tools as get_filesystem_tools
from emad_host_te.tools.notify import get_tools as get_notify_tools
from emad_host_te.tools.operational import get_tools as get_operational_tools
from emad_host_te.tools.system import get_tools as get_system_tools
from emad_host_te.tools.web import get_tools as get_web_tools

_log = logging.getLogger("emad_host.flows.imperator")

# MAD identity — hostname is the Docker container name
_MAD_HOSTNAME = socket.gethostname()


# ── State ────────────────────────────────────────────────────────────────


class ImperatorState(TypedDict):
    """State for the Imperator ReAct agent."""

    messages: Annotated[list[AnyMessage], add_messages]
    context_window_id: Optional[str]
    config: dict
    response_text: Optional[str]
    error: Optional[str]
    iteration_count: int
    _user_message_stored: Optional[bool]
    _streaming: Optional[bool]


# ── Tool assembly ──────────────────────────────────────────────────────


# R7-m14: Pre-bound LLM with tools — set at graph compilation time
_prebound_llm = None


def _collect_tools(imperator_config: dict) -> list:
    """Collect all active tools based on config."""
    active = []
    active.extend(get_diagnostic_tools())
    active.extend(get_web_tools())
    active.extend(get_filesystem_tools())
    active.extend(get_system_tools())
    active.extend(get_notify_tools(imperator_config))
    if imperator_config.get("admin_tools", False):
        active.extend(get_admin_tools())
    active.extend(get_operational_tools(imperator_config))
    active.extend(get_alerting_tools(imperator_config))
    return active


# ── CB integration helpers ─────────────────────────────────────────────


def _get_cb_url(config: dict) -> Optional[str]:
    """Return the Context Broker URL from config, or None if not configured."""
    cb_config = config.get("context_broker", {})
    if not cb_config:
        return None
    url = cb_config.get("url")
    if not url or url == "null":
        return None
    return url.rstrip("/")


async def _cb_call(cb_url: str, tool_name: str, arguments: dict) -> dict:
    """Call a Context Broker MCP tool via HTTP."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        response = await client.post(f"{cb_url}/mcp", json=payload)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(f"CB tool {tool_name} failed: {result['error']}")
        return result.get("result", {})


# ── Graph nodes ──────────────────────────────────────────────────────────


async def init_context_node(state: ImperatorState) -> dict:
    """First-call setup: load system prompt and conversation history.

    CB mode: calls get_context on the configured Context Broker.
    No-CB mode: uses trim_messages with 8192 fallback for context management.
    """
    config = state["config"]
    ctx = get_ctx()

    imperator_cfg = config.get("imperator", {})
    prompt_name = imperator_cfg.get("system_prompt", "imperator_identity")
    try:
        system_content = await ctx.async_load_prompt(prompt_name)
    except RuntimeError as exc:
        _log.error("Failed to load system prompt '%s': %s", prompt_name, exc)
        return {
            "messages": [AIMessage(content="I encountered a configuration error.")],
            "response_text": "I encountered a configuration error.",
            "error": f"Prompt loading failed: {exc}",
        }

    system_msg = SystemMessage(content=system_content)
    messages = list(state["messages"])
    history_messages = []

    cb_url = _get_cb_url(config)

    if cb_url:
        # CB mode: load context via Context Broker
        conversation_id = state.get("context_window_id")
        if conversation_id:
            try:
                build_type = imperator_cfg.get("build_type", "tiered-summary")
                budget = imperator_cfg.get("max_context_tokens", 8192)
                if not isinstance(budget, int):
                    budget = 8192

                get_context_args = {
                    "build_type": build_type,
                    "budget": budget,
                    "conversation_id": str(conversation_id),
                }

                # Extract user query for retrieval
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        get_context_args["user_prompt"] = msg.content
                        break

                ctx_result = await _cb_call(cb_url, "get_context", get_context_args)
                context_messages = ctx_result.get("context", [])

                for msg in context_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "assistant":
                        history_messages.append(
                            AIMessage(content=content, tool_calls=msg.get("tool_calls", []))
                        )
                    elif role == "tool":
                        history_messages.append(
                            ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "unknown"))
                        )
                    elif role == "system":
                        if content and content.strip() == system_msg.content.strip():
                            continue
                        history_messages.append(SystemMessage(content=content))
                    else:
                        history_messages.append(HumanMessage(content=content))

            except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
                _log.warning("Failed to load context from CB: %s", exc)
    else:
        # No-CB mode: use trim_messages for context management
        from langchain_core.messages import trim_messages as _trim_messages

        max_tokens = imperator_cfg.get("max_context_tokens", 8192)
        if not isinstance(max_tokens, int):
            max_tokens = 8192

        messages = _trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=len,  # Approximate: count messages, not tokens
            allow_partial=False,
        )

    # Reconcile DB history with HTTP messages
    if history_messages and messages:
        current_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                current_user_msg = msg
                break
        if current_user_msg is not None:
            messages = [current_user_msg]

        if messages:
            last_hist = history_messages[-1] if history_messages else None
            last_msg = messages[-1]
            if (isinstance(last_hist, HumanMessage)
                    and isinstance(last_msg, HumanMessage)
                    and last_hist.content == last_msg.content):
                history_messages = history_messages[:-1]

    assembled = [system_msg] + history_messages + messages

    return {"messages": assembled}


async def ceac_enrichment_node(state: ImperatorState) -> dict:
    """CEAc enrichment — pass-through when no CB is configured.

    In CB mode, this would run the CEAc enrichment subgraph.
    In no-CB mode, this is a no-op.
    """
    config = state.get("config", {})
    cb_url = _get_cb_url(config)

    if not cb_url:
        return {}

    cea_config = config.get("cea", {}).get("ceac", {})
    if not cea_config.get("enabled"):
        return {}

    # CB mode with CEAc enabled — enrichment would be implemented here
    # when the pMAD is connected to a Context Broker that supports CEAc.
    # For now, pass through.
    return {}


async def llm_call_node(state: ImperatorState) -> dict:
    """Call the LLM with bound tools and return the response.

    ARCH-05: This node contains NO loop. Flow control (tool-call vs
    final answer) is handled by the conditional edge after this node.
    """
    config = state["config"]
    ctx = get_ctx()

    if state.get("_streaming"):
        imperator_config = config.get("imperator", {})
        active_tools = _collect_tools(imperator_config)
        llm = ctx.get_chat_model(config, role="imperator", streaming=True)
        llm_with_tools = llm.bind_tools(active_tools)
    elif _prebound_llm is not None:
        llm_with_tools = _prebound_llm
    else:
        imperator_config = config.get("imperator", {})
        active_tools = _collect_tools(imperator_config)
        llm = ctx.get_chat_model(config, role="imperator")
        llm_with_tools = llm.bind_tools(active_tools)

    messages = list(state["messages"])

    # Truncate older messages if the list exceeds the limit
    max_react_messages = ctx.get_tuning(config, "imperator_max_react_messages", 40)
    if len(messages) > max_react_messages:
        cut_index = len(messages) - (max_react_messages - 1)
        while cut_index < len(messages) and isinstance(
            messages[cut_index], ToolMessage
        ):
            cut_index += 1
        messages = [messages[0]] + messages[cut_index:]

    # Sanitize message sequence: remove consecutive HumanMessages (keep last),
    # consecutive AIMessages (keep last).
    sanitized = []
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage) and i > 0 and sanitized and not isinstance(sanitized[-1], SystemMessage):
            sanitized.append(msg)
        elif isinstance(msg, SystemMessage) and i == 0:
            sanitized.append(msg)
        elif isinstance(msg, HumanMessage):
            if sanitized and isinstance(sanitized[-1], HumanMessage):
                sanitized[-1] = msg
            else:
                sanitized.append(msg)
        elif isinstance(msg, AIMessage):
            if sanitized and isinstance(sanitized[-1], AIMessage):
                sanitized[-1] = msg
            else:
                sanitized.append(msg)
        else:
            sanitized.append(msg)
    if len(sanitized) != len(messages):
        _log.info(
            "Sanitized message sequence: %d → %d messages",
            len(messages), len(sanitized),
        )
    messages = sanitized

    _log.info(
        "Imperator LLM call: %d messages, types=%s",
        len(messages),
        [type(m).__name__ for m in messages[-5:]],
    )
    max_retries = 2
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = await llm_with_tools.ainvoke(messages)
        except (openai.APIError, httpx.HTTPError, ValueError, RuntimeError) as exc:
            _log.error("Imperator LLM call failed: %s", exc, exc_info=True)
            return {
                "messages": [
                    AIMessage(
                        content="I encountered an error processing your request."
                    )
                ],
                "response_text": "I encountered an error processing your request.",
                "error": str(exc),
            }

        if (response.content and response.content.strip()) or response.tool_calls:
            break

        if attempt < max_retries:
            _log.warning(
                "Imperator LLM returned empty response (attempt %d/%d) — retrying",
                attempt + 1,
                max_retries + 1,
            )
        else:
            _log.error(
                "Imperator LLM returned empty response after %d attempts",
                max_retries + 1,
            )

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def needs_init(state: ImperatorState) -> str:
    """Conditional edge at entry: route to init or llm_call."""
    IDENTITY_MARKER = "Imperator"
    for m in state.get("messages", []):
        if isinstance(m, SystemMessage) and IDENTITY_MARKER in (m.content or ""):
            return "llm_call_node"
    return "init_context_node"


def should_continue(state: ImperatorState) -> str:
    """Conditional edge: route to tool_node if tool calls, else store nodes."""
    if state.get("error"):
        return "store_user_message"

    messages = state["messages"]
    if not messages:
        return "store_user_message"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        max_iterations = get_ctx().get_tuning(
            state.get("config", {}), "imperator_max_iterations", 10
        )
        if state.get("iteration_count", 0) >= max_iterations:
            _log.warning(
                "Imperator hit max iterations (%d) — forcing end",
                max_iterations,
            )
            return "max_iterations_fallback"
        return "tool_node"

    return "store_user_message"


async def max_iterations_fallback(state: ImperatorState) -> dict:
    """Inject a fallback text response when max iterations is reached."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "I was unable to complete that request within the allowed "
                    "number of steps. Please try again, or break the request "
                    "into smaller parts."
                )
            )
        ]
    }


async def store_user_message(state: ImperatorState) -> dict:
    """Persist the user's message via the Context Broker (if configured).

    No-op when no CB is configured.
    """
    config = state.get("config", {})
    cb_url = _get_cb_url(config)
    conversation_id = state.get("context_window_id")

    if not cb_url or not conversation_id:
        return {}

    if state.get("_user_message_stored"):
        return {}

    user_content = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_content = msg.content

    if not user_content:
        return {}

    user_identity = (
        state.get("config", {}).get("imperator", {}).get("_request_user", "unknown")
    )

    try:
        await _cb_call(cb_url, "store_message", {
            "conversation_id": str(conversation_id),
            "role": "user",
            "sender": user_identity,
            "recipient": _MAD_HOSTNAME,
            "content": user_content,
        })
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
        _log.warning("Failed to store user message via CB: %s", exc)

    return {}


async def store_assistant_message(state: ImperatorState) -> dict:
    """Persist the assistant's response via the Context Broker (if configured).

    No-op when no CB is configured. Always extracts and returns response_text.
    """
    last_ai = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai = msg
            break

    response_text = last_ai.content if last_ai else ""

    if not response_text.strip():
        _log.warning("Imperator produced empty response")
        response_text = (
            "I was unable to generate a response. This may be due to a "
            "temporary provider issue. Please try again."
        )

    config = state.get("config", {})
    cb_url = _get_cb_url(config)
    conversation_id = state.get("context_window_id")
    user_identity = config.get("imperator", {}).get("_request_user", "unknown")

    is_fallback = response_text.startswith("I was unable to generate")

    if cb_url and conversation_id and response_text and not is_fallback:
        try:
            await _cb_call(cb_url, "store_message", {
                "conversation_id": str(conversation_id),
                "role": "assistant",
                "sender": _MAD_HOSTNAME,
                "recipient": user_identity,
                "content": response_text,
            })
        except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
            _log.warning("Failed to store assistant message via CB: %s", exc)

    return {"response_text": response_text}


# ── Build the graph ──────────────────────────────────────────────────────


def build_imperator_flow(config: dict | None = None) -> StateGraph:
    """Build and compile the Imperator StateGraph.

    ARCH-05: Proper graph structure with agent_node <-> tool_node loop
             via conditional edges. No while loops inside nodes.
    """
    ctx = get_ctx()
    if config is None:
        config = ctx.load_merged_config()
    imperator_config = config.get("imperator", {})
    active_tools = _collect_tools(imperator_config)
    tool_node_instance = ToolNode(active_tools)

    global _prebound_llm
    llm = ctx.get_chat_model(config, role="imperator")
    _prebound_llm = llm.bind_tools(active_tools)

    workflow = StateGraph(ImperatorState)

    workflow.add_node("init_context_node", init_context_node)
    workflow.add_node("ceac_enrichment_node", ceac_enrichment_node)
    workflow.add_node("llm_call_node", llm_call_node)
    workflow.add_node("tool_node", tool_node_instance)
    workflow.add_node("max_iterations_fallback", max_iterations_fallback)
    workflow.add_node("store_user_message", store_user_message)
    workflow.add_node("store_assistant_message", store_assistant_message)

    workflow.set_conditional_entry_point(
        needs_init,
        {
            "init_context_node": "init_context_node",
            "llm_call_node": "llm_call_node",
        },
    )

    workflow.add_edge("init_context_node", "ceac_enrichment_node")
    workflow.add_edge("ceac_enrichment_node", "llm_call_node")

    workflow.add_conditional_edges(
        "llm_call_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "max_iterations_fallback": "max_iterations_fallback",
            "store_user_message": "store_user_message",
        },
    )

    workflow.add_edge("tool_node", "llm_call_node")
    workflow.add_edge("max_iterations_fallback", "store_user_message")
    workflow.add_edge("store_user_message", "store_assistant_message")
    workflow.add_edge("store_assistant_message", END)

    from langgraph.checkpoint.memory import MemorySaver

    return workflow.compile(checkpointer=MemorySaver())
