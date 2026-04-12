"""
Imperator flow wrapper — kernel-side metrics and lifecycle management.

Looks up the TE's Imperator from the package_registry and invokes
it with metrics recording. This is the kernel's interface to the TE.

Per REQ-001 §6.4: metrics produced inside the flow layer, not route handlers.
"""

import logging
import time
import uuid
from typing import AsyncGenerator

from app.metrics_registry import CHAT_REQUESTS, CHAT_REQUEST_DURATION

_log = logging.getLogger("emad_host.flows.imperator_wrapper")

# Lazy singleton — rebuilt on install_stategraph() or TE config change.
_imperator_flow = None
_te_config_mtime: float = 0.0


def _get_flow():
    """Get the compiled Imperator flow from the TE registry.

    Recompiles when TE config changes (detects admin_tools toggle,
    model changes, etc.) without requiring container restart.
    """
    global _imperator_flow, _te_config_mtime
    import os

    from app.config import TE_CONFIG_PATH

    # Check if TE config changed since last compilation
    try:
        current_mtime = os.stat(TE_CONFIG_PATH).st_mtime
    except (OSError, FileNotFoundError):
        current_mtime = 0.0

    if _imperator_flow is not None and current_mtime != _te_config_mtime:
        _log.info("TE config changed — recompiling Imperator flow")
        _imperator_flow = None

    if _imperator_flow is None:
        from app.package_registry import get_imperator_builder

        builder = get_imperator_builder()
        if builder is None:
            raise RuntimeError(
                "No TE package registered. Install a TE package with "
                "install_stategraph or ensure one is installed at startup."
            )
        _imperator_flow = builder()
        _te_config_mtime = current_mtime
    return _imperator_flow


def invalidate():
    """Clear the cached flow. Called after install_stategraph()."""
    global _imperator_flow
    _imperator_flow = None
    _log.info("Imperator flow cache invalidated")


async def invoke_with_metrics(initial_state: dict, config: dict | None = None) -> dict:
    """Invoke the Imperator flow and record chat metrics.

    Per REQ-001 §6.4: metrics produced inside the flow layer, not route handlers.
    """
    start_time = time.monotonic()
    status = "error"
    try:
        # Each invocation gets a unique thread_id. MemorySaver persists state
        # WITHIN a single ReAct execution (agent → tool → agent → response),
        # not across separate user turns. Cross-turn conversation memory comes
        # from the Context Broker via get_context, not from MemorySaver.
        thread_id = str(uuid.uuid4())
        result = await _get_flow().ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        status = "success"
        return result
    finally:
        duration = time.monotonic() - start_time
        CHAT_REQUESTS.labels(status=status).inc()
        CHAT_REQUEST_DURATION.observe(duration)


async def astream_events_with_metrics(
    initial_state: dict,
    config: dict | None = None,
) -> AsyncGenerator:
    """Stream Imperator events and record chat metrics.

    Per REQ-001 §6.4: metrics produced inside the flow layer, not route handlers.
    """
    start_time = time.monotonic()
    status = "error"
    try:
        thread_id = str(uuid.uuid4())
        async for event in _get_flow().astream_events(
            initial_state,
            version="v2",
            config={"configurable": {"thread_id": thread_id}},
        ):
            yield event
        status = "success"
    finally:
        duration = time.monotonic() - start_time
        CHAT_REQUESTS.labels(status=status).inc()
        CHAT_REQUEST_DURATION.observe(duration)
