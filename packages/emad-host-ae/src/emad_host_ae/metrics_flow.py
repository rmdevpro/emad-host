"""
Metrics collection StateGraph flow.

Collects Prometheus metrics inside a StateGraph node,
as required by REQ §4.8.
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph
from prometheus_client import generate_latest, REGISTRY
from typing_extensions import TypedDict

_log = logging.getLogger("emad_host.flows.metrics")


class MetricsState(TypedDict):
    """State for the metrics collection flow."""

    action: str
    metrics_output: str
    error: Optional[str]


async def collect_metrics_node(state: MetricsState) -> dict:
    """Collect Prometheus metrics from the registry.

    Produces metrics output inside the StateGraph as required by REQ §4.8.
    """
    try:
        metrics_bytes = generate_latest(REGISTRY)
        metrics_text = metrics_bytes.decode("utf-8", errors="replace")
        return {"metrics_output": metrics_text, "error": None}
    except (ValueError, OSError) as exc:
        _log.error("Failed to collect metrics: %s", exc)
        return {"metrics_output": "", "error": str(exc)}


def build_metrics_flow() -> StateGraph:
    """Build and compile the metrics collection StateGraph."""
    workflow = StateGraph(MetricsState)
    workflow.add_node("collect_metrics", collect_metrics_node)
    workflow.set_entry_point("collect_metrics")
    workflow.add_edge("collect_metrics", END)
    return workflow.compile()
