"""
MCP (Model Context Protocol) endpoint.

Implements HTTP/SSE transport for MCP tool access.
Routes tool calls to compiled StateGraph flows.

Endpoints:
  GET  /mcp          — Establish SSE session
  POST /mcp          — Sessionless tool call or route to session
  POST /mcp?sessionId=xxx — Route to existing session
"""

import asyncio
import decimal
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.config import async_load_config, get_tuning
from app.flows.tool_dispatch import dispatch_tool
from app.models import MCPToolCall
from app.routes.caller_identity import resolve_caller

_log = logging.getLogger("context_broker.routes.mcp")


def _json_default(obj: object) -> object:
    """Handle non-standard types in MCP JSON responses."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


router = APIRouter()

# Active SSE sessions: session_id -> {"queue": asyncio.Queue, "created_at": float}
# OrderedDict preserves insertion order for efficient eviction of oldest sessions.
_sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()

# R5-M25: Track total queued messages across all sessions to bound memory
_total_queued_messages: int = 0

# Configurable limits (defaults; overridden by config at request time)
_MAX_SESSIONS = 1000
_SESSION_TTL_SECONDS = 3600  # 1 hour
_MAX_TOTAL_QUEUED = 10000

# R6-M3: Lock for session dict mutations to prevent race conditions
_session_lock = asyncio.Lock()


def _evict_stale_sessions(
    session_ttl: int = _SESSION_TTL_SECONDS,
    max_sessions: int = _MAX_SESSIONS,
    max_total_queued: int = _MAX_TOTAL_QUEUED,
) -> None:
    """Remove sessions older than TTL and enforce the max sessions cap.

    R5-M25: Also evict oldest sessions when total queued messages exceed the cap.
    R7-m15: Accepts parameters instead of reading module globals on every call.
    """
    global _total_queued_messages
    now = time.monotonic()
    stale_ids = [
        sid for sid, info in _sessions.items() if now - info["created_at"] > session_ttl
    ]
    for sid in stale_ids:
        info = _sessions.pop(sid, None)
        if info is not None:
            _total_queued_messages -= info["queue"].qsize()
        _log.info("MCP SSE session evicted (TTL): %s", sid)

    # Evict oldest if over cap
    while len(_sessions) > max_sessions:
        evicted_id, info = _sessions.popitem(last=False)
        _total_queued_messages -= info["queue"].qsize()
        _log.info("MCP SSE session evicted (cap): %s", evicted_id)

    # R5-M25: Evict oldest sessions if total queued messages exceed threshold
    while _total_queued_messages > max_total_queued and _sessions:
        evicted_id, info = _sessions.popitem(last=False)
        _total_queued_messages -= info["queue"].qsize()
        _log.warning("MCP SSE session evicted (total queue pressure): %s", evicted_id)


@router.get("/mcp")
async def mcp_sse_session(request: Request) -> StreamingResponse:
    """Establish an SSE session for MCP communication.

    Returns an SSE stream. The client sends tool calls via
    POST /mcp?sessionId=<id>.
    """
    # R7-m15: Read config limits but don't mutate module globals on every request.
    # Use local variables for this request's session creation.
    config = await async_load_config()
    max_sessions = get_tuning(config, "mcp_max_sessions", _MAX_SESSIONS)
    session_ttl = get_tuning(config, "mcp_session_ttl_seconds", _SESSION_TTL_SECONDS)
    max_total_queued = get_tuning(config, "mcp_max_total_queued", _MAX_TOTAL_QUEUED)

    session_id = str(uuid.uuid4())

    # R6-M3: Protect session dict mutations with asyncio.Lock
    async with _session_lock:
        _evict_stale_sessions(session_ttl, max_sessions, max_total_queued)
        # G5-26: Bound the per-session queue to prevent memory growth from slow clients
        message_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _sessions[session_id] = {"queue": message_queue, "created_at": time.monotonic()}

    _log.info("MCP SSE session established: %s (active=%d)", session_id, len(_sessions))

    async def event_stream() -> AsyncGenerator[str, None]:
        # Send session ID as first event
        yield f"data: {json.dumps({'sessionId': session_id})}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(message_queue.get(), timeout=30.0)
                    # R5-M25: Decrement global counter when message is consumed
                    global _total_queued_messages
                    _total_queued_messages = max(0, _total_queued_messages - 1)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        finally:
            # R6-M3: Protect session dict mutations with asyncio.Lock
            async with _session_lock:
                # R6-M2: Decrement global counter by remaining queue size before removal
                removed = _sessions.pop(session_id, None)
                if removed is not None:
                    _total_queued_messages = max(
                        0, _total_queued_messages - removed["queue"].qsize()
                    )
            _log.info("MCP SSE session closed: %s", session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/mcp")
async def mcp_tool_call(
    request: Request,
    session_id: str = Query(None, alias="sessionId"),
) -> JSONResponse:
    """Handle an MCP tool call.

    Supports both sessionless mode (no sessionId) and session mode.
    All tool calls are routed through StateGraph flows.
    """
    tool_name = "unknown"

    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        _log.warning("MCP: failed to parse request body: %s", exc)
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            },
        )

    try:
        mcp_request = MCPToolCall(**body)
    except ValidationError as exc:
        _log.warning("MCP: invalid request structure: %s", exc)
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "Invalid Request"},
            },
        )

    if mcp_request.method == "initialize":
        tool_name = "initialize"
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "context-broker",
                        "version": "1.0.0",
                    },
                },
            }
        )

    if mcp_request.method == "tools/list":
        tool_name = "tools_list"
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "result": {"tools": _get_tool_list()},
            }
        )

    if mcp_request.method != "tools/call":
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {mcp_request.method}",
                },
            },
        )

    tool_name = mcp_request.params.get("name", "unknown")
    tool_arguments = mcp_request.params.get("arguments", {})

    # R7-M9: Wrap config load in try/except with error response
    try:
        config = await async_load_config()
    except (OSError, RuntimeError, ValueError) as exc:
        _log.error("MCP: config load failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "error": {
                    "code": -32000,
                    "message": f"Configuration unavailable: {exc}",
                },
            },
        )

    # Resolve caller identity from the HTTP request for sender/recipient
    caller = await resolve_caller(request)
    config = {
        **config,
        "imperator": {
            **config.get("imperator", {}),
            "_request_user": caller,
        },
    }

    try:
        result = await dispatch_tool(
            tool_name, tool_arguments, config, request.app.state
        )

        response_content = {
            "jsonrpc": "2.0",
            "id": mcp_request.id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            json.dumps(result, default=_json_default)
                            if isinstance(result, dict)
                            else str(result)
                        ),
                    }
                ]
            },
        }

        # If session mode, push to session queue and return acknowledgment
        if session_id:
            if session_id not in _sessions:
                # G5-25: Unknown sessionId — return error instead of falling through
                return JSONResponse(
                    status_code=404,
                    content={
                        "jsonrpc": "2.0",
                        "id": mcp_request.id,
                        "error": {
                            "code": -32001,
                            "message": f"Session not found: {session_id}",
                        },
                    },
                )
            try:
                async with _session_lock:
                    if session_id not in _sessions:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "jsonrpc": "2.0",
                                "id": mcp_request.id,
                                "error": {
                                    "code": -32001,
                                    "message": f"Session disconnected: {session_id}",
                                },
                            },
                        )
                    _sessions[session_id]["queue"].put_nowait(response_content)
                    global _total_queued_messages
                    _total_queued_messages += 1
            except asyncio.QueueFull:
                _log.warning(
                    "MCP SSE session queue full for session=%s; dropping response",
                    session_id,
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "jsonrpc": "2.0",
                        "id": mcp_request.id,
                        "error": {
                            "code": -32000,
                            "message": "Session queue full — client is not consuming events fast enough",
                        },
                    },
                )
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": mcp_request.id,
                    "result": "queued",
                }
            )

        return JSONResponse(content=response_content)

    except (ValueError, ValidationError) as exc:
        _log.warning("MCP tool '%s' validation error: %s", tool_name, exc)
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "error": {"code": -32602, "message": str(exc)},
            },
        )
    except (RuntimeError, ConnectionError, OSError) as exc:
        _log.error("MCP tool '%s' failed: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "error": {"code": -32000, "message": str(exc)},
            },
        )
    # Metrics (MCP_REQUESTS, MCP_REQUEST_DURATION) are recorded inside
    # dispatch_tool() per REQ-001 §6.4 (metrics in flows, not route handlers).


def _get_tool_list() -> list[dict]:
    """Return the MCP tool list with schemas."""
    return [
        {
            "name": "imperator_chat",
            "description": "Conversational interface to the Imperator",
            "inputSchema": {
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {"type": "string"},
                    "context_window_id": {"type": "string", "format": "uuid"},
                },
            },
        },
        {
            "name": "metrics_get",
            "description": "Retrieve Prometheus metrics",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "install_stategraph",
            "description": "Install or upgrade a StateGraph package at runtime without container restart (REQ-001 §10)",
            "inputSchema": {
                "type": "object",
                "required": ["package_name"],
                "properties": {
                    "package_name": {
                        "type": "string",
                        "description": "Python package name (e.g., 'context-broker-te', 'context-broker-ae')",
                    },
                    "version": {
                        "type": "string",
                        "description": "Specific version to install (e.g., '0.2.0'). Omit for latest.",
                    },
                },
            },
        },
        {
            "name": "emad_chat",
            "description": "Send a message to a named eMAD instance and receive a response",
            "inputSchema": {
                "type": "object",
                "required": ["emad_name", "message"],
                "properties": {
                    "emad_name": {"type": "string", "description": "Name of the eMAD instance"},
                    "conversation_id": {"type": "string", "description": "Conversation ID for session continuity"},
                    "message": {"type": "string", "description": "Message to send to the eMAD"},
                },
            },
        },
        {
            "name": "emad_install_package",
            "description": "Install an eMAD package and refresh the registry",
            "inputSchema": {
                "type": "object",
                "required": ["package_name"],
                "properties": {
                    "package_name": {"type": "string", "description": "Python package name of the eMAD"},
                    "version": {"type": "string", "description": "Specific version to install. Omit for latest."},
                },
            },
        },
        {
            "name": "emad_create",
            "description": "Register a new named eMAD instance backed by an installed package",
            "inputSchema": {
                "type": "object",
                "required": ["emad_name", "package_name"],
                "properties": {
                    "emad_name": {"type": "string", "description": "Unique name for the eMAD instance"},
                    "package_name": {"type": "string", "description": "Installed eMAD package to back this instance"},
                    "description": {"type": "string", "description": "Human-readable description"},
                    "parameters": {"type": "object", "description": "Instance parameters (e.g., system_prompt)"},
                },
            },
        },
        {
            "name": "emad_update",
            "description": "Update parameters or description of an existing eMAD instance",
            "inputSchema": {
                "type": "object",
                "required": ["emad_name"],
                "properties": {
                    "emad_name": {"type": "string", "description": "Name of the eMAD instance to update"},
                    "description": {"type": "string", "description": "New description (optional)"},
                    "parameters": {"type": "object", "description": "New parameters (optional)"},
                },
            },
        },
        {
            "name": "emad_delete",
            "description": "Remove a named eMAD instance. Does not uninstall the backing package.",
            "inputSchema": {
                "type": "object",
                "required": ["emad_name"],
                "properties": {
                    "emad_name": {"type": "string", "description": "Name of the eMAD instance to delete"},
                },
            },
        },
        {
            "name": "emad_list",
            "description": "List all registered eMAD instances with their package, parameters, and status",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]
