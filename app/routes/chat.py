"""
OpenAI-compatible chat completions endpoint.

Implements /v1/chat/completions following the OpenAI API specification.
Routes to the Imperator StateGraph.
Supports both streaming (SSE) and non-streaming responses.
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator

import asyncpg
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from app.config import async_load_config
from app.flows.imperator_wrapper import (
    astream_events_with_metrics,
    invoke_with_metrics,
)
from app.models import ChatCompletionRequest
from app.routes.caller_identity import resolve_caller

_log = logging.getLogger("emad_host.routes.chat")

router = APIRouter()


async def _lookup_emad_instance(model: str) -> dict | None:
    """Check if model name matches an active eMAD instance.

    Returns the instance row dict if found and active, None otherwise.
    Non-fatal — returns None on any DB error so the Imperator fallback works.
    """
    try:
        from app.database import get_pg_pool

        pool = get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT emad_name, package_name, parameters, status "
                "FROM emad_instances WHERE emad_name = $1",
                model,
            )
        if row is not None and row["status"] == "active":
            return dict(row)
    except (asyncpg.PostgresError, asyncpg.InterfaceError, RuntimeError, OSError):
        pass
    return None


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    """Handle OpenAI-compatible chat completion requests.

    Routes to the Imperator StateGraph. Supports streaming and non-streaming.
    Metrics are recorded inside the flow layer per REQ-001 §6.4.
    """
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        _log.warning("Chat: failed to parse request body: %s", exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": {"message": "Invalid JSON", "type": "invalid_request_error"}
            },
        )

    try:
        chat_request = ChatCompletionRequest(**body)
    except ValidationError as exc:
        _log.warning("Chat: request validation failed: %s", exc)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

    # R7-M9: Wrap config load in try/except with error response
    try:
        config = await async_load_config()
    except (OSError, RuntimeError, ValueError) as exc:
        _log.error("Chat: config load failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Configuration unavailable",
                    "type": "internal_error",
                }
            },
        )

    imperator_manager = getattr(request.app.state, "imperator_manager", None)

    # Extract the last user message as the primary input
    user_messages = [m for m in chat_request.messages if m.role == "user"]
    if not user_messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "At least one user message is required",
                    "type": "invalid_request_error",
                }
            },
        )

    # G5-27: Allow clients to specify a context_window_id for multi-client
    # isolation via x-context-window-id header or context_window_id in the body.
    # Also accepts the legacy x-conversation-id / conversation_id for compatibility.
    # Falls back to the default Imperator context window when not provided.
    context_window_id = (
        request.headers.get("x-context-window-id")
        or body.get("context_window_id")
        or request.headers.get("x-conversation-id")
        or body.get("conversation_id")
    )
    if not context_window_id and imperator_manager is not None:
        context_window_id = await imperator_manager.get_context_window_id()

    # Convert plain messages to LangChain message objects
    # G5-28: Include ToolMessage so tool-role messages are not coerced to HumanMessage.
    _role_map = {
        "user": HumanMessage,
        "system": SystemMessage,
        "assistant": AIMessage,
        "tool": ToolMessage,
    }
    lc_messages = []
    for m in chat_request.messages:
        cls = _role_map.get(m.role, HumanMessage)
        if cls is ToolMessage:
            # ToolMessage requires a tool_call_id; use the one from the
            # request body if available, otherwise fall back to a placeholder.
            tool_call_id = m.tool_call_id or "unknown"
            lc_messages.append(
                ToolMessage(content=m.content, tool_call_id=tool_call_id)
            )
        elif cls is AIMessage:
            # R7-M14: Pass tool_calls if present for AIMessage (G5-28)
            lc_messages.append(
                AIMessage(content=m.content, tool_calls=m.tool_calls or [])
            )
        else:
            lc_messages.append(cls(content=m.content))

    # Resolve caller identity for sender/recipient on stored messages.
    caller = await resolve_caller(request, chat_request.user)
    config = {
        **config,
        "imperator": {
            **config.get("imperator", {}),
            "_request_user": caller,
        },
    }

    # ── eMAD routing: if model matches a registered eMAD, dispatch to it ──
    emad_instance = await _lookup_emad_instance(chat_request.model)
    if emad_instance is not None:
        return await _handle_emad_request(
            emad_instance, chat_request, lc_messages, config
        )

    # ── Imperator (default) ──
    initial_state = {
        "messages": lc_messages,
        "context_window_id": str(context_window_id) if context_window_id else None,
        "config": config,
        "response_text": None,
        "error": None,
    }

    try:
        if chat_request.stream:
            return StreamingResponse(
                _stream_imperator_response(initial_state, chat_request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            # Metrics recorded inside invoke_with_metrics (flow layer)
            result = await invoke_with_metrics(initial_state)

            if result.get("error"):
                _log.error("Imperator flow error: %s", result["error"])
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": result["error"],
                            "type": "internal_error",
                        }
                    },
                )

            response_text = result.get("response_text", "")

            return JSONResponse(
                content=_build_completion_response(response_text, chat_request.model)
            )

    except (RuntimeError, ConnectionError, OSError) as exc:
        _log.error("Chat completion failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                }
            },
        )


async def _stream_imperator_response(
    initial_state: dict,
    chat_request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Stream the Imperator response as SSE tokens.

    M-22: astream_events(version="v2") captures on_chat_model_stream events
    from nested ainvoke() calls within the LangGraph runtime, so real token
    streaming works without requiring the agent to use astream() internally.
    Metrics are recorded inside astream_events_with_metrics (flow layer).
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        # G5-29: Known limitation — when the ReAct agent processes tool calls,
        # astream_events may emit no content tokens for those intermediate LLM
        # turns (only the final non-tool-call turn produces streamable tokens).
        # Metrics recorded inside the flow wrapper per REQ-001 §6.4.
        async for event in astream_events_with_metrics(initial_state):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": chat_request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except (RuntimeError, ConnectionError, OSError) as exc:
        _log.error("Streaming imperator response failed: %s", exc, exc_info=True)
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "An error occurred processing your request."},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _build_completion_response(response_text: str, model: str) -> dict:
    """Build an OpenAI-compatible non-streaming completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


# ── eMAD dispatch via OpenAI-compatible endpoint ──────────────────────────


async def _handle_emad_request(
    emad_instance: dict,
    chat_request: ChatCompletionRequest,
    lc_messages: list,
    config: dict,
) -> JSONResponse | StreamingResponse:
    """Dispatch a chat request to an eMAD instance.

    Builds the eMAD's StateGraph from the registered package, invokes it,
    and returns the response in OpenAI-compatible format. Supports streaming
    via astream_events (same pattern as kaiser-langgraph/server.py).
    """
    from app import package_registry

    model = emad_instance["emad_name"]
    package_name = emad_instance["package_name"]
    raw_params = emad_instance["parameters"]
    if isinstance(raw_params, str):
        parameters = json.loads(raw_params)
    elif raw_params:
        parameters = dict(raw_params)
    else:
        parameters = {}

    build_func = package_registry.get_build_func(package_name)
    if build_func is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Package not loaded: {package_name}",
                    "type": "server_error",
                }
            },
        )

    try:
        graph = build_func(parameters)
    except (TypeError, ImportError, RuntimeError) as exc:
        _log.error("eMAD build error for %s: %s", model, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"eMAD build error: {exc}",
                    "type": "server_error",
                }
            },
        )

    # Extract last user message content
    last_user = ""
    for msg in reversed(lc_messages):
        if isinstance(msg, HumanMessage):
            last_user = msg.content
            break

    # Initial state per ERQ-004 §2.1
    initial_state = {
        "messages": [HumanMessage(content=last_user)],
        "rogers_conversation_id": "",
        "rogers_context_window_id": None,
        "config": config,
    }

    if chat_request.stream:
        return StreamingResponse(
            _stream_emad_response(graph, initial_state, chat_request, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: invoke and return
    try:
        result = await graph.ainvoke(initial_state)
        response_text = (
            result.get("final_response")
            or result.get("response_text")
            or "[No response from eMAD]"
        )
        return JSONResponse(content=_build_completion_response(response_text, model))
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        _log.error("eMAD invocation error for %s: %s", model, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "eMAD execution error",
                    "type": "internal_error",
                }
            },
        )


async def _stream_emad_response(
    graph,
    initial_state: dict,
    chat_request: ChatCompletionRequest,
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream eMAD response as SSE tokens.

    Same pattern as kaiser-langgraph/server.py:openai_chat_completions().
    Uses astream_events(version="v2") to capture on_chat_model_stream events.
    """
    completion_id = f"chatcmpl-emad-{uuid.uuid4().hex}"
    created = int(time.time())
    first_content = True
    yielded_any = False

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk_data = event["data"].get("chunk")
                if chunk_data is None:
                    continue
                content = chunk_data.content if hasattr(chunk_data, "content") else ""
                if not content:
                    continue
                delta = {"content": content}
                if first_content:
                    delta["role"] = "assistant"
                    first_content = False
                sse_chunk = json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                yield f"data: {sse_chunk}\n\n"
                yielded_any = True
            elif kind == "on_chat_model_end":
                # Fallback for eMADs that disable streaming
                if not yielded_any:
                    output_msg = event.get("data", {}).get("output")
                    if (
                        output_msg is not None
                        and hasattr(output_msg, "content")
                        and output_msg.content
                    ):
                        has_tool_calls = bool(
                            getattr(output_msg, "tool_calls", None)
                            or getattr(output_msg, "additional_kwargs", {}).get(
                                "tool_calls"
                            )
                        )
                        if not has_tool_calls:
                            content = (
                                output_msg.content
                                if isinstance(output_msg.content, str)
                                else str(output_msg.content)
                            )
                            delta = {"role": "assistant", "content": content}
                            sse_chunk = json.dumps(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            yield f"data: {sse_chunk}\n\n"
                            yielded_any = True
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        _log.error("eMAD streaming error: %s", exc)

    # Final chunk
    final = json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield f"data: {final}\n\ndata: [DONE]\n\n"
