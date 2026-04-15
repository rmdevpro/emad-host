"""MAD client — talks to any State 4 MAD via standard endpoints.

Uses:
  - /v1/chat/completions (OpenAI-compatible) for chat
  - /health for health status
  - /mcp for MCP tool calls
"""

import json
import logging
from typing import AsyncGenerator

import httpx

_log = logging.getLogger("ui.mad_client")


class MADClient:
    """Client for a single State 4 MAD. Reuses a persistent httpx client."""

    def __init__(self, name: str, base_url: str, hostname: str = ""):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.hostname = hostname or name
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(300, connect=10),
        )

    async def health(self) -> dict:
        try:
            resp = await self._client.get("/health", timeout=5)
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return {"status": "unreachable"}

    async def chat(
        self,
        model: str,
        messages: list[dict],
        conversation_id: str | None = None,
    ) -> dict:
        """Send chat completion (non-streaming). Returns full response dict."""
        payload = {
            "model": model,
            "messages": messages,
            "conversation_id": conversation_id or "new",
        }
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        conversation_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions via SSE."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "conversation_id": conversation_id or "new",
        }
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def query_logs(self, limit: int = 30) -> list[dict]:
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
