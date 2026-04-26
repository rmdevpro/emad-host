"""MAD client — talks to any State 4 MAD via standard endpoints.

Uses:
  - /v1/chat/completions (OpenAI-compatible) for chat, streaming
  - /v1/conversations(...) for session listing/retrieval/deletion
  - /health for status
"""

import json
import logging
from typing import AsyncGenerator

import httpx

_log = logging.getLogger("ui.mad_client")


class StreamEvent:
    """Small tagged union for items yielded by chat_stream."""
    __slots__ = ("kind", "text", "conversation_id")

    def __init__(self, kind: str, text: str = "", conversation_id: str | None = None):
        self.kind = kind  # "token" | "meta" | "error"
        self.text = text
        self.conversation_id = conversation_id


class MADClient:
    """Client for a single State 4 MAD. Holds a persistent httpx.AsyncClient."""

    def __init__(self, name: str, base_url: str, hostname: str = ""):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.hostname = hostname or name
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        try:
            resp = await self._client.get("/health", timeout=5)
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return {"status": "unreachable"}

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        conversation_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat completion. Yields StreamEvent('token', text=...) for
        each content delta and StreamEvent('meta', conversation_id=...) when
        the server emits the conversation id. On transport error yields
        StreamEvent('error', text=<message>).
        """
        payload = {"model": model, "messages": messages, "stream": True}
        payload["conversation_id"] = conversation_id or "new"

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    meta_conv = chunk.get("conversation_id")
                    if meta_conv:
                        yield StreamEvent("meta", conversation_id=meta_conv)
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        yield StreamEvent("token", text=content)
        except httpx.HTTPStatusError as exc:
            yield StreamEvent("error", text=f"HTTP {exc.response.status_code}")
        except (httpx.HTTPError, OSError) as exc:
            yield StreamEvent("error", text=str(exc))

    async def list_conversations(
        self, model: str | None = None, limit: int = 50
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if model:
            params["model"] = model
        try:
            resp = await self._client.get("/v1/conversations", params=params)
            resp.raise_for_status()
            return resp.json().get("conversations", [])
        except (httpx.HTTPError, ValueError):
            return []

    async def get_conversation(self, conversation_id: str) -> dict | None:
        try:
            resp = await self._client.get(f"/v1/conversations/{conversation_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def delete_conversation(self, conversation_id: str) -> bool:
        try:
            resp = await self._client.delete(f"/v1/conversations/{conversation_id}")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def query_logs(self, limit: int = 30) -> list[dict]:
        """Query recent logs via MCP."""
        try:
            result = await self._mcp_call("query_logs", {"limit": limit})
            return result.get("entries", [])
        except (RuntimeError, OSError):
            return []

    async def _mcp_call(self, tool_name: str, arguments: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        resp = await self._client.post("/mcp", json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"MCP error: {body['error']}")
        text = body.get("result", {}).get("content", [{}])[0].get("text", "{}")
        return json.loads(text)
