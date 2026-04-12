"""
Health check endpoint.

Tests all backing service connections and returns aggregated status.
The LangGraph container performs the actual dependency checks —
nginx proxies the response without performing checks itself.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import async_load_config
from app.stategraph_registry import get_flow_builder

_log = logging.getLogger("emad_host.routes.health")

router = APIRouter()

_health_flow = None


def _get_health_flow():
    global _health_flow
    if _health_flow is None:
        builder = get_flow_builder("health_check")
        if builder is None:
            raise RuntimeError("AE package not loaded: health_check flow unavailable")
        _health_flow = builder()
    return _health_flow


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """Check connectivity to all backing services.

    Returns 200 if all critical services are healthy.
    Returns 503 if any critical service is unhealthy.
    """
    # R7-M8: Wrap config load in try/except — return degraded health instead of 500
    try:
        config = await async_load_config()
    except (OSError, RuntimeError, ValueError) as exc:
        _log.warning("Health check: config load failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "error": f"Config load failed: {exc}",
                "database": "unknown",
                "neo4j": "unknown",
            },
        )

    result = await _get_health_flow().ainvoke(
        {
            "config": config,
            "postgres_ok": False,
            "neo4j_ok": False,
            "all_healthy": False,
            "status_detail": None,
            "http_status": 503,
        }
    )

    return JSONResponse(
        status_code=result["http_status"],
        content=result["status_detail"],
    )
