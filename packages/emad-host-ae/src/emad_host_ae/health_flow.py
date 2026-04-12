"""
Health Check — LangGraph StateGraph flow.

Checks connectivity to backing services (PostgreSQL)
and returns aggregated health status. Invoked by the /health route.
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

_log = logging.getLogger("emad_host.flows.health")


class HealthCheckState(TypedDict):
    """State for the health check flow."""

    config: dict

    postgres_ok: bool
    all_healthy: bool
    status_detail: Optional[dict]
    http_status: int


async def check_dependencies(state: HealthCheckState) -> dict:
    """Check connectivity to all backing services."""
    from app.database import get_pg_pool

    postgres_ok = False
    try:
        pool = get_pg_pool()
        await pool.fetchval("SELECT 1")
        postgres_ok = True
    except (RuntimeError, OSError, Exception) as exc:
        _log.warning("PostgreSQL health check failed: %s", exc)

    all_healthy = postgres_ok

    if not all_healthy:
        status_label = "unhealthy"
        http_status = 503
    else:
        status_label = "healthy"
        http_status = 200

    status_detail = {
        "status": status_label,
        "database": "ok" if postgres_ok else "error",
    }

    if not all_healthy:
        _log.warning("Health check: unhealthy — %s", status_detail)

    return {
        "postgres_ok": postgres_ok,
        "all_healthy": all_healthy,
        "status_detail": status_detail,
        "http_status": http_status,
    }


def build_health_check_flow() -> StateGraph:
    """Build and compile the health check StateGraph."""
    workflow = StateGraph(HealthCheckState)
    workflow.add_node("check_dependencies", check_dependencies)
    workflow.set_entry_point("check_dependencies")
    workflow.add_edge("check_dependencies", END)
    return workflow.compile()
