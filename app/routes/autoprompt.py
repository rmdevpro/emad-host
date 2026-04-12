"""
Autoprompter callback endpoint.

Receives HTTP POST from Dkron when a job fires. Routes to the
autoprompt_dispatcher StateGraph flow which reads the runbook
and delivers it to the Imperator.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.package_registry import get_flow_builder

_log = logging.getLogger("emad_host.routes.autoprompt")

router = APIRouter()

_dispatcher_flow = None


def _get_dispatcher():
    global _dispatcher_flow
    if _dispatcher_flow is None:
        builder = get_flow_builder("autoprompt_dispatcher")
        if builder is None:
            raise RuntimeError(
                "autoprompt_dispatcher flow not available. Is the AE package installed?"
            )
        _dispatcher_flow = builder()
    return _dispatcher_flow


@router.post("/autoprompt")
async def autoprompt_callback(request: Request) -> JSONResponse:
    """Handle Dkron job callback.

    Expects JSON body with:
    - job_name: name of the fired job
    - runbook_path: relative path to runbook file in /config/runbooks/
    - target_url: (optional) override for the Imperator chat endpoint
    """
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        _log.warning("Autoprompt: invalid JSON: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON"},
        )

    job_name = body.get("job_name", "unknown")
    runbook_path = body.get("runbook_path", "")
    target_url = body.get(
        "target_url",
        "http://emad-host-langgraph:8000/v1/chat/completions",
    )

    if not runbook_path:
        return JSONResponse(
            status_code=400,
            content={"error": "runbook_path is required"},
        )

    _log.info("Autoprompt: job '%s' fired, runbook='%s'", job_name, runbook_path)

    try:
        dispatcher = _get_dispatcher()
        result = await dispatcher.ainvoke({
            "job_name": job_name,
            "runbook_path": runbook_path,
            "target_url": target_url,
            "runbook_content": None,
            "delivery_status": None,
            "error": None,
        })

        if result.get("error"):
            _log.error("Autoprompt job '%s' failed: %s", job_name, result["error"])
            return JSONResponse(
                status_code=500,
                content={"error": result["error"]},
            )

        return JSONResponse(
            content={
                "status": result.get("delivery_status", "unknown"),
                "job_name": job_name,
            }
        )
    except RuntimeError as exc:
        _log.error("Autoprompt dispatch failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )
