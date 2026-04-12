"""
Tool dispatch — routes MCP tool calls to compiled StateGraph flows.

All tool logic lives in StateGraph flows loaded dynamically from AE/TE
packages via entry_points (REQ-001 §10). This module is the thin
kernel-side routing layer that maps tool names to their flows using
the package_registry.
"""

import logging
import time
from typing import Any

from app.metrics_registry import MCP_REQUESTS, MCP_REQUEST_DURATION
from app.package_registry import get_flow_builder, get_imperator_builder
from app.models import (
    EmadChatInput,
    EmadCreateInput,
    EmadDeleteInput,
    EmadInstallPackageInput,
    EmadUpdateInput,
    ImperatorChatInput,
    MetricsGetInput,
)

_log = logging.getLogger("emad_host.flows.tool_dispatch")

# Lazy-initialized flow singletons — compiled on first use from
# dynamically loaded packages via the package_registry.
_flow_cache: dict[str, Any] = {}


def _get_flow(name: str) -> Any:
    """Get a compiled flow by registry name (lazy singleton)."""
    if name not in _flow_cache:
        builder = get_flow_builder(name)
        if builder is None:
            raise RuntimeError(
                f"Flow '{name}' not available. Is the AE package installed?"
            )
        _flow_cache[name] = builder()
    return _flow_cache[name]


def _get_imperator_flow() -> Any:
    """Get the compiled Imperator flow from the TE registry."""
    if "imperator" not in _flow_cache:
        builder = get_imperator_builder()
        if builder is None:
            raise RuntimeError(
                "No TE package registered. Install a TE package with "
                "install_stategraph or ensure one is installed at startup."
            )
        _flow_cache["imperator"] = builder()
    return _flow_cache["imperator"]


_emad_flow_builders = {
    "emad_chat": "app.flows.emad_dispatch:build_emad_chat_flow",
    "emad_install_package": "app.flows.emad_management:build_install_emad_package_flow",
    "emad_create": "app.flows.emad_management:build_create_emad_flow",
    "emad_update": "app.flows.emad_management:build_update_emad_flow",
    "emad_delete": "app.flows.emad_management:build_delete_emad_flow",
    "emad_list": "app.flows.emad_management:build_list_emads_flow",
}

_emad_flow_cache: dict[str, Any] = {}


def _get_emad_flow(name: str) -> Any:
    """Get a compiled eMAD management flow by name (lazy singleton)."""
    if name not in _emad_flow_cache:
        ref = _emad_flow_builders.get(name)
        if ref is None:
            raise RuntimeError(f"Unknown eMAD flow: {name}")
        module_path, func_name = ref.rsplit(":", 1)
        import importlib

        module = importlib.import_module(module_path)
        builder = getattr(module, func_name)
        _emad_flow_cache[name] = builder()
    return _emad_flow_cache[name]


def invalidate_flow_cache() -> None:
    """Clear all cached flows. Called after install_stategraph()."""
    _flow_cache.clear()
    _emad_flow_cache.clear()
    _log.info("Flow dispatch cache cleared")


async def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    config: dict[str, Any],
    app_state: Any,
) -> dict[str, Any]:
    """Route a tool call to its StateGraph flow.

    Validates inputs using Pydantic models before invoking flows.
    Raises ValueError for unknown tools or validation errors.
    """
    _log.info("Dispatching tool: %s", tool_name)
    _start_time = time.monotonic()
    _status = "error"

    try:
        result = await _dispatch_tool_inner(tool_name, arguments, config, app_state)
        _status = "success"
        return result
    finally:
        _duration = time.monotonic() - _start_time
        MCP_REQUESTS.labels(tool=tool_name, status=_status).inc()
        MCP_REQUEST_DURATION.labels(tool=tool_name).observe(_duration)


async def _dispatch_tool_inner(
    tool_name: str,
    arguments: dict[str, Any],
    config: dict[str, Any],
    app_state: Any,
) -> dict[str, Any]:
    """Inner dispatch — routes tool calls to their StateGraph flows."""

    if tool_name == "imperator_chat":
        validated = ImperatorChatInput(**arguments)
        from langchain_core.messages import HumanMessage
        import uuid as _uuid

        thread_id = str(_uuid.uuid4())

        result = await _get_imperator_flow().ainvoke(
            {
                "messages": [HumanMessage(content=validated.message)],
                "context_window_id": (
                    str(validated.context_window_id)
                    if validated.context_window_id
                    else None
                ),
                "config": config,
                "response_text": None,
                "error": None,
                "iteration_count": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        if result.get("error"):
            raise ValueError(result["error"])
        return {
            "response": result.get("response_text", ""),
        }

    elif tool_name == "metrics_get":
        MetricsGetInput(**arguments)
        result = await _get_flow("metrics").ainvoke(
            {
                "action": "collect",
                "metrics_output": "",
                "error": None,
            }
        )
        if result.get("error"):
            raise ValueError(result["error"])
        return {"metrics": result.get("metrics_output", "")}

    elif tool_name == "install_stategraph":
        package_name = arguments.get("package_name", "")
        version = arguments.get("version")
        if not package_name:
            raise ValueError("package_name is required")
        from app.flows.install_stategraph import install_stategraph

        result = await install_stategraph(package_name, version)
        # Invalidate all cached flows so next call uses new package
        invalidate_flow_cache()
        from app.flows.imperator_wrapper import invalidate as invalidate_imperator

        invalidate_imperator()
        return result

    elif tool_name == "emad_chat":
        validated = EmadChatInput(**arguments)
        flow = _get_emad_flow("emad_chat")
        result = await flow.ainvoke(
            {
                "emad_name": validated.emad_name,
                "conversation_id": validated.conversation_id,
                "message": validated.message,
                "instance_row": None,
                "emad_result": None,
                "response": None,
                "rogers_conversation_id": None,
                "error": None,
            }
        )
        if result.get("error"):
            raise ValueError(result["error"])
        return {
            "response": result["response"],
            "conversation_id": result.get("rogers_conversation_id")
            or validated.conversation_id,
            "emad_name": validated.emad_name,
        }

    elif tool_name == "emad_install_package":
        validated = EmadInstallPackageInput(**arguments)
        flow = _get_emad_flow("emad_install_package")
        result = await flow.ainvoke(
            {
                "package_name": validated.package_name,
                "version": validated.version,
                "result": None,
            }
        )
        return result.get("result") or {"status": "error", "detail": "unknown"}

    elif tool_name == "emad_create":
        validated = EmadCreateInput(**arguments)
        flow = _get_emad_flow("emad_create")
        result = await flow.ainvoke(
            {
                "emad_name": validated.emad_name,
                "package_name": validated.package_name,
                "description": validated.description,
                "parameters": validated.parameters,
                "result": None,
            }
        )
        return result.get("result") or {"status": "error"}

    elif tool_name == "emad_update":
        validated = EmadUpdateInput(**arguments)
        flow = _get_emad_flow("emad_update")
        result = await flow.ainvoke(
            {
                "emad_name": validated.emad_name,
                "description": validated.description,
                "parameters": validated.parameters,
                "result": None,
            }
        )
        return result.get("result") or {"status": "error"}

    elif tool_name == "emad_delete":
        validated = EmadDeleteInput(**arguments)
        flow = _get_emad_flow("emad_delete")
        result = await flow.ainvoke(
            {"emad_name": validated.emad_name, "result": None}
        )
        return result.get("result") or {"status": "error"}

    elif tool_name == "emad_list":
        flow = _get_emad_flow("emad_list")
        result = await flow.ainvoke({"result": None})
        return result.get("result") or {"emads": []}

    else:
        raise ValueError(f"Unknown tool: {tool_name}")
