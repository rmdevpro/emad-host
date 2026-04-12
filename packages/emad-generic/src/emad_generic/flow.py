"""
emad-generic: Imperator flow — generic conversational agent.

StateGraph: init_node -> llm_call_node -> extract_response

Uses the host pMAD's Imperator inference settings (ERQ-004 §4.2).
No admin tools — pure conversational agent.

ERQ-004 §3.1: Imperator with Identity and Purpose.
ERQ-002 §2.1: All logic as LangGraph StateGraphs.
ERQ-002 §11.4: ReAct pattern (agent node -> conditional edge -> tool node).
  Note: this generic eMAD has no tools, so the ReAct loop is a single pass.
"""

import logging
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

_log = logging.getLogger("emad_generic.flow")

_MAX_ITERATIONS = 8

# Default system prompt — overridden by instance parameters
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class GenericImperatorState(TypedDict):
    """State for the generic eMAD Imperator (ERQ-004 §3.2)."""

    messages: Annotated[list[AnyMessage], add_messages]
    rogers_conversation_id: Optional[str]
    rogers_context_window_id: Optional[str]
    config: dict
    final_response: Optional[str]


def build_graph(params: dict):
    """Build and compile the generic eMAD StateGraph.

    Args:
        params: Instance parameters from emad_instances.parameters.
            - system_prompt: Custom system prompt text (optional).

    Returns:
        Compiled StateGraph ready for ainvoke().
    """
    system_prompt = params.get("system_prompt") or _DEFAULT_SYSTEM_PROMPT

    async def init_node(state: GenericImperatorState) -> dict:
        """Prepend system prompt to messages if not already present."""
        messages = list(state["messages"])

        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages = [SystemMessage(content=system_prompt)] + messages

        return {"messages": messages}

    async def llm_call_node(state: GenericImperatorState) -> dict:
        """Call the LLM using the host's Imperator inference settings.

        ERQ-004 §4.2: Uses provided inference interface via config,
        not direct LLM instantiation.
        """
        config = state.get("config", {})

        # Use host AE's get_chat_model (available in-process)
        from app.config import get_chat_model

        llm = get_chat_model(config, role="imperator")

        messages = list(state["messages"])

        _log.info(
            "Generic eMAD LLM call: %d messages",
            len(messages),
        )

        try:
            response = await llm.ainvoke(messages)
        except (ValueError, RuntimeError, OSError) as exc:
            _log.error("Generic eMAD LLM call failed: %s", exc)
            return {
                "messages": [
                    AIMessage(
                        content="I encountered an error processing your request."
                    )
                ],
            }

        return {"messages": [response]}

    def extract_response(state: GenericImperatorState) -> dict:
        """Extract final_response from last AIMessage (ERQ-004 §2.2)."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return {"final_response": str(msg.content)}
        return {"final_response": "[No response generated]"}

    # Build the graph (ERQ-002 §2.1, §11.4)
    g = StateGraph(GenericImperatorState)
    g.add_node("init_node", init_node)
    g.add_node("llm_call_node", llm_call_node)
    g.add_node("extract_response", extract_response)

    g.set_entry_point("init_node")
    g.add_edge("init_node", "llm_call_node")
    g.add_edge("llm_call_node", "extract_response")
    g.add_edge("extract_response", END)

    return g.compile()
