"""
Autoprompter dispatcher — StateGraph flow for Dkron callbacks.

When Dkron fires a job, it sends an HTTP POST to the langgraph container.
This flow reads the referenced runbook file and POSTs its contents as a
prompt to the Imperator's /v1/chat/completions endpoint.

The dispatcher has zero intelligence — it only reads and delivers.
"""

import logging
from pathlib import Path
from typing import Optional

import httpx
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

_log = logging.getLogger("emad_host.flows.autoprompt_dispatcher")

RUNBOOK_DIR = Path("/config/runbooks")


class DispatcherState(TypedDict):
    """State for the autoprompter dispatcher flow."""

    job_name: str
    runbook_path: str
    target_url: str
    runbook_content: Optional[str]
    delivery_status: Optional[str]
    error: Optional[str]


async def load_runbook(state: DispatcherState) -> dict:
    """Read the runbook file from disk."""
    runbook_path = state.get("runbook_path", "")
    if not runbook_path:
        return {"error": "No runbook_path specified"}

    full_path = RUNBOOK_DIR / runbook_path
    try:
        content = full_path.read_text(encoding="utf-8").strip()
        if not content:
            return {"error": f"Runbook is empty: {runbook_path}"}
        _log.info("Loaded runbook: %s (%d chars)", runbook_path, len(content))
        return {"runbook_content": content}
    except (FileNotFoundError, OSError) as exc:
        _log.error("Failed to load runbook %s: %s", runbook_path, exc)
        return {"error": f"Failed to load runbook: {exc}"}


async def deliver_prompt(state: DispatcherState) -> dict:
    """POST the runbook content to the Imperator's chat endpoint."""
    if state.get("error"):
        return {}

    content = state.get("runbook_content", "")
    target_url = state.get("target_url", "http://emad-host-langgraph:8000/v1/chat/completions")

    payload = {
        "model": "emad-host",
        "messages": [
            {"role": "user", "content": content},
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(target_url, json=payload)
            response.raise_for_status()
        _log.info(
            "Autoprompter delivered job '%s' to %s",
            state.get("job_name", "unknown"),
            target_url,
        )
        return {"delivery_status": "delivered"}
    except (httpx.HTTPError, OSError) as exc:
        _log.error(
            "Autoprompter delivery failed for job '%s': %s",
            state.get("job_name", "unknown"),
            exc,
        )
        return {"delivery_status": "failed", "error": str(exc)}


def build_autoprompt_dispatcher_flow() -> StateGraph:
    """Build and compile the autoprompter dispatcher StateGraph."""
    workflow = StateGraph(DispatcherState)
    workflow.add_node("load_runbook", load_runbook)
    workflow.add_node("deliver_prompt", deliver_prompt)
    workflow.set_entry_point("load_runbook")
    workflow.add_edge("load_runbook", "deliver_prompt")
    workflow.add_edge("deliver_prompt", END)
    return workflow.compile()
