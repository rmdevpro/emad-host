"""
Tool dispatch — routes MCP tool calls to AE flows and tools.

All tools are AE-owned. This module dispatches MCP tool calls to
the appropriate handler: AE infrastructure flows (health, metrics)
or AE tools (file_read, web_search, etc.).
"""

import logging
import time
from typing import Any

from app.metrics_registry import MCP_REQUESTS, MCP_REQUEST_DURATION
from app.package_registry import get_flow_builder
from app.models import MetricsGetInput

_log = logging.getLogger("emad_host.flows.tool_dispatch")

# Lazy-initialized flow singletons
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


def invalidate_flow_cache() -> None:
    """Clear all cached flows. Called after install_package()."""
    _flow_cache.clear()
    _log.info("Flow dispatch cache cleared")


async def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    config: dict[str, Any],
    app_state: Any,
) -> dict[str, Any]:
    """Route a tool call to its handler.

    Validates inputs using Pydantic models before invoking.
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
    """Inner dispatch — routes tool calls to handlers."""

    if tool_name == "metrics_get":
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

    elif tool_name == "install_package":
        package_name = arguments.get("package_name", "")
        version = arguments.get("version")
        if not package_name:
            raise ValueError("package_name is required")
        from app.flows.install_stategraph import install_stategraph

        result = await install_stategraph(package_name, version)
        invalidate_flow_cache()
        return result

    # AE tools from the tool registry
    from app.tools import TOOL_REGISTRY

    if tool_name in TOOL_REGISTRY:
        tool_fn = TOOL_REGISTRY[tool_name]
        # Tools are LangChain @tool decorated — invoke with arguments
        result = await tool_fn.ainvoke(arguments)
        return {"result": result}

    raise ValueError(f"Unknown tool: {tool_name}")
