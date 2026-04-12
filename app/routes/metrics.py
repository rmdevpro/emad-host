"""
Prometheus metrics endpoint.

Exposes metrics collected from StateGraph executions.
Metrics are produced inside StateGraphs, not in route handlers.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

from app.stategraph_registry import get_flow_builder

_log = logging.getLogger("emad_host.routes.metrics")

router = APIRouter()

_metrics_flow = None


def _get_metrics_flow():
    global _metrics_flow
    if _metrics_flow is None:
        builder = get_flow_builder("metrics")
        if builder is None:
            raise RuntimeError("AE package not loaded: metrics flow unavailable")
        _metrics_flow = builder()
    return _metrics_flow


@router.get("/metrics")
async def get_metrics() -> Response:
    """Expose Prometheus metrics in exposition format.

    Metrics are collected inside the StateGraph flow.
    """
    initial_state = {
        "action": "collect",
        "metrics_output": "",
        "error": None,
    }
    result = await _get_metrics_flow().ainvoke(initial_state)

    # G5-30: Check for flow errors and return 500 instead of masking with 200.
    if result.get("error"):
        _log.error("Metrics flow error: %s", result["error"])
        return Response(
            content=f"# ERROR: metrics collection failed: {result['error']}\n",
            media_type="text/plain",
            status_code=500,
        )

    metrics_data = result.get("metrics_output", "")
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
