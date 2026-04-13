"""
OpenAI-compatible chat completions endpoint — dumb router.

Reads the model field from the OpenAI payload, looks up the stategraph
in the routing table, and forwards the full payload. Does not extract
messages, build state, or inject parameters. The stategraph handles
everything.

Contract with stategraphs:
  Input:  full OpenAI request body (dict) as initial state under "payload" key
  Output: {"response_text": str, "conversation_id": str | None}
  Streaming: astream_events emits on_chat_model_stream events
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.models import ChatCompletionRequest

_log = logging.getLogger("emad_host.routes.chat")

router = APIRouter()


_graph_cache: dict = {}


def _get_stategraph(model_name: str):
    """Look up and return the compiled stategraph for the given model name.

    Caches compiled graphs. Returns None if the model is not registered.

    Lookup order:
    1. eMAD directory (/emads/{model_name}/config.json exists) → runbook-emad-te
    2. Host Imperator (fallback for any model name when TE is loaded)
    """
    if model_name in _graph_cache:
        return _graph_cache[model_name]

    import os

    from app.package_registry import get_imperator_builder, get_build_func

    # Check if this model has an eMAD config directory
    emad_config_path = f"/emads/{model_name}/config.json"
    if os.path.isfile(emad_config_path):
        build_func = get_build_func("runbook-emad-te")
        if build_func is not None:
            graph = build_func({})
            _graph_cache[model_name] = graph
            return graph

    # Host Imperator — only for the "host" model name
    if model_name == "host":
        builder = get_imperator_builder()
        if builder is not None:
            graph = builder()
            _graph_cache[model_name] = graph
            return graph

    return None


def invalidate_graph_cache() -> None:
    """Clear cached graphs. Called after package install."""
    _graph_cache.clear()


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    """Route OpenAI-compatible chat requests to the appropriate stategraph.

    Pure router — reads model, looks up stategraph, forwards full payload.
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

    model = chat_request.model

    # Look up stategraph for this model name
    graph = _get_stategraph(model)
    if graph is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Model not found: {model}",
                    "type": "invalid_request_error",
                }
            },
        )

    _log.info("Routing model=%s", model)

    # Resolve conversation_id for the checkpointer's thread_id.
    # The router extracts this from the payload but doesn't interpret it —
    # the stategraph's init_node handles "new" and default logic.
    conv_id = body.get("conversation_id") or ""
    if conv_id == "new":
        import uuid as _uuid
        conv_id = str(_uuid.uuid4())
    elif not conv_id:
        # Let the stategraph handle default thread resolution
        conv_id = f"default-{model}"

    initial_state = {"payload": body}
    graph_config = {"configurable": {"thread_id": conv_id}}

    try:
        if chat_request.stream:
            return StreamingResponse(
                _stream_response(graph, initial_state, model, graph_config),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            result = await graph.ainvoke(initial_state, config=graph_config)

            if result.get("error"):
                _log.error("Stategraph error for model=%s: %s", model, result["error"])
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
            conversation_id = result.get("conversation_id") or conv_id

            return JSONResponse(
                content=_build_completion_response(
                    response_text, model, conversation_id
                )
            )

    except (RuntimeError, ConnectionError, OSError) as exc:
        _log.error("Chat completion failed for model=%s: %s", model, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                }
            },
        )


async def _stream_response(
    graph,
    initial_state: dict,
    model: str,
    graph_config: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Stream stategraph response as SSE tokens.

    Uses astream_events(version="v2") to capture on_chat_model_stream
    events from the stategraph's LLM calls.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yielded_any = False

    try:
        async for event in graph.astream_events(
            initial_state, version="v2", config=graph_config
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk_data = event["data"].get("chunk")
                if chunk_data is None:
                    continue
                content = chunk_data.content if hasattr(chunk_data, "content") else ""
                if not content:
                    continue
                delta = {"content": content}
                if not yielded_any:
                    delta["role"] = "assistant"
                    yielded_any = True
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
            elif kind == "on_chat_model_end":
                # Fallback for stategraphs that disable streaming
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
        _log.error("Streaming error for model=%s: %s", model, exc)

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


def _build_completion_response(
    response_text: str, model: str, conversation_id: str | None = None
) -> dict:
    """Build an OpenAI-compatible non-streaming completion response.

    Always includes conversation_id so clients can continue the conversation.
    """
    response = {
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
    if conversation_id:
        response["conversation_id"] = conversation_id
    return response
