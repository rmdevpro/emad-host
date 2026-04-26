"""
OpenAI-compatible chat completions endpoint — dumb router.

Reads the model field from the OpenAI payload, looks up the stategraph
in the routing table, and forwards the full payload. Does not extract
messages, build state, or inject parameters. The stategraph handles
everything.

Also exposes lightweight conversation listing/retrieval/deletion for
the UI sidebar. These read from the conversations + conversation_messages
tables (migration 006), which the endpoint writes to after each turn.

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

from app.database import get_pg_pool
from app.models import ChatCompletionRequest

_log = logging.getLogger("emad_host.routes.chat")

router = APIRouter()


_graph_cache: dict = {}


async def _get_stategraph(model_name: str):
    """Look up and return a compiled stategraph for the given model name."""
    if model_name in _graph_cache:
        return _graph_cache[model_name]

    from app.package_registry import get_imperator_builder, get_build_func

    if model_name == "host":
        builder = get_imperator_builder()
        if builder is not None:
            graph = builder()
            _graph_cache[model_name] = graph
            return graph
        return None

    try:
        pool = get_pg_pool()
        row = await pool.fetchrow(
            "SELECT package_name FROM emad_instances WHERE emad_name = $1 AND status = 'active'",
            model_name,
        )
        if row is None:
            return None

        package_name = row["package_name"]
        build_func = get_build_func(package_name)
        if build_func is None:
            from app.package_registry import load_emad
            try:
                load_emad(package_name)
                build_func = get_build_func(package_name)
            except (ImportError, AttributeError) as exc:
                _log.warning("Failed to load eMAD package '%s': %s", package_name, exc)
        if build_func is not None:
            graph = build_func({})
            _graph_cache[model_name] = graph
            return graph
    except (RuntimeError, OSError) as exc:
        _log.warning("Failed to look up eMAD '%s': %s", model_name, exc)

    return None


def invalidate_graph_cache() -> None:
    """Clear cached graphs. Called after package install."""
    _graph_cache.clear()


# ── Conversation tracking ────────────────────────────────────────────


def _last_user_message(body: dict) -> str:
    """Extract the last user message text from an OpenAI request body."""
    for msg in reversed(body.get("messages", []) or []):
        if msg.get("role") == "user":
            content = msg.get("content") or ""
            if isinstance(content, str):
                return content
            return str(content)
    return ""


async def _record_turn(
    conversation_id: str | None,
    model: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Upsert conversation row + insert user/assistant message pair.

    Best-effort: failures are logged but do not break the chat response.
    """
    if not conversation_id or not user_text:
        return
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except (ValueError, TypeError):
        return

    try:
        pool = get_pg_pool()
        title = user_text.strip().splitlines()[0][:80] if user_text.strip() else ""
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, model, title)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (conversation_id) DO UPDATE
                    SET updated_at = NOW()
                    """,
                    conv_uuid,
                    model,
                    title,
                )
                await conn.execute(
                    """
                    INSERT INTO conversation_messages (conversation_id, role, content)
                    VALUES ($1, 'user', $2), ($1, 'assistant', $3)
                    """,
                    conv_uuid,
                    user_text,
                    assistant_text or "",
                )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to record conversation turn %s: %s", conversation_id, exc)


# ── Chat completions ─────────────────────────────────────────────────


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    """Route OpenAI-compatible chat requests to the appropriate stategraph."""
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        _log.warning("Chat: failed to parse request body: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
        )

    try:
        chat_request = ChatCompletionRequest(**body)
    except ValidationError as exc:
        _log.warning("Chat: request validation failed: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    model = chat_request.model

    graph = await _get_stategraph(model)
    if graph is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Model not found: {model}", "type": "invalid_request_error"}},
        )

    _log.info("Routing model=%s", model)

    user_text = _last_user_message(body)
    initial_state = {"payload": body}

    try:
        if chat_request.stream:
            return StreamingResponse(
                _stream_response(graph, initial_state, model, user_text),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await graph.ainvoke(initial_state)

        if result.get("error"):
            _log.error("Stategraph error for model=%s: %s", model, result["error"])
            return JSONResponse(
                status_code=500,
                content={"error": {"message": result["error"], "type": "internal_error"}},
            )

        response_text = result.get("response_text", "")
        conversation_id = result.get("conversation_id")
        await _record_turn(conversation_id, model, user_text, response_text)

        return JSONResponse(
            content=_build_completion_response(response_text, model, conversation_id)
        )

    except (RuntimeError, ConnectionError, OSError) as exc:
        _log.error("Chat completion failed for model=%s: %s", model, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )


async def _stream_response(
    graph,
    initial_state: dict,
    model: str,
    user_text: str,
) -> AsyncGenerator[str, None]:
    """Stream stategraph response as SSE tokens.

    Emits a non-standard SSE chunk carrying conversation_id before [DONE]
    so the UI can track it without a separate request. Records the turn to
    the conversations tables after the stream finishes.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yielded_any = False
    accumulated = ""
    final_output = None

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
                accumulated += content
                delta = {"content": content}
                if not yielded_any:
                    delta["role"] = "assistant"
                    yielded_any = True
                sse_chunk = json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                })
                yield f"data: {sse_chunk}\n\n"
            elif kind == "on_chat_model_end":
                if not yielded_any:
                    output_msg = event.get("data", {}).get("output")
                    if (
                        output_msg is not None
                        and hasattr(output_msg, "content")
                        and output_msg.content
                    ):
                        has_tool_calls = bool(
                            getattr(output_msg, "tool_calls", None)
                            or getattr(output_msg, "additional_kwargs", {}).get("tool_calls")
                        )
                        if not has_tool_calls:
                            content = (
                                output_msg.content
                                if isinstance(output_msg.content, str)
                                else str(output_msg.content)
                            )
                            accumulated += content
                            delta = {"role": "assistant", "content": content}
                            sse_chunk = json.dumps({
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            })
                            yield f"data: {sse_chunk}\n\n"
                            yielded_any = True
            elif kind == "on_chain_end":
                if not event.get("parent_ids"):
                    final_output = event.get("data", {}).get("output")
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        _log.error("Streaming error for model=%s: %s", model, exc)

    conversation_id = None
    if isinstance(final_output, dict):
        conversation_id = final_output.get("conversation_id")

    if not yielded_any and isinstance(final_output, dict):
        response_text = final_output.get("response_text") or ""
        if response_text:
            accumulated += response_text
            delta = {"role": "assistant", "content": response_text}
            sse_chunk = json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            })
            yield f"data: {sse_chunk}\n\n"
            yielded_any = True

    await _record_turn(conversation_id, model, user_text, accumulated)

    if conversation_id:
        meta_chunk = json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "conversation_id": conversation_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        })
        yield f"data: {meta_chunk}\n\n"

    final = json.dumps({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield f"data: {final}\n\ndata: [DONE]\n\n"


def _build_completion_response(
    response_text: str, model: str, conversation_id: str | None = None
) -> dict:
    """Build an OpenAI-compatible non-streaming completion response."""
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }
    if conversation_id:
        response["conversation_id"] = conversation_id
    return response


# ── Conversation management ──────────────────────────────────────────


@router.get("/v1/conversations")
async def list_conversations(model: str | None = None, limit: int = 50):
    """List recent conversations, optionally filtered by model."""
    limit = max(1, min(limit, 500))
    try:
        pool = get_pg_pool()
        if model:
            rows = await pool.fetch(
                """
                SELECT conversation_id::text AS conversation_id,
                       model, title, created_at, updated_at
                FROM conversations
                WHERE model = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                model,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT conversation_id::text AS conversation_id,
                       model, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                limit,
            )
        return JSONResponse(content={
            "conversations": [
                {
                    "conversation_id": r["conversation_id"],
                    "model": r["model"],
                    "title": r["title"],
                    "created_at": r["created_at"].isoformat(),
                    "updated_at": r["updated_at"].isoformat(),
                }
                for r in rows
            ]
        })
    except (RuntimeError, OSError) as exc:
        _log.warning("Failed to list conversations: %s", exc)
        return JSONResponse(status_code=500, content={"error": "failed_to_list"})


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Return the full message history for a conversation."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "invalid_id"})

    try:
        pool = get_pg_pool()
        conv = await pool.fetchrow(
            """
            SELECT conversation_id::text AS conversation_id,
                   model, title, created_at, updated_at
            FROM conversations WHERE conversation_id = $1
            """,
            conv_uuid,
        )
        if conv is None:
            return JSONResponse(status_code=404, content={"error": "not_found"})

        rows = await pool.fetch(
            """
            SELECT role, content, created_at
            FROM conversation_messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC, id ASC
            """,
            conv_uuid,
        )
        return JSONResponse(content={
            "conversation_id": conv["conversation_id"],
            "model": conv["model"],
            "title": conv["title"],
            "messages": [
                {"role": r["role"], "content": r["content"]} for r in rows
            ],
        })
    except (RuntimeError, OSError) as exc:
        _log.warning("Failed to get conversation %s: %s", conversation_id, exc)
        return JSONResponse(status_code=500, content={"error": "failed_to_load"})


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "invalid_id"})

    try:
        pool = get_pg_pool()
        result = await pool.execute(
            "DELETE FROM conversations WHERE conversation_id = $1",
            conv_uuid,
        )
        deleted = result.split()[-1] if result else "0"
        return JSONResponse(content={"deleted": int(deleted)})
    except (RuntimeError, OSError, ValueError) as exc:
        _log.warning("Failed to delete conversation %s: %s", conversation_id, exc)
        return JSONResponse(status_code=500, content={"error": "failed_to_delete"})
