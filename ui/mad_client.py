"""MAD client — talks to any State 4 MAD via standard endpoints.

Uses:
  - /v1/chat/completions (OpenAI-compatible) for chat
  - /health for health status
"""

import json
import logging
from typing import AsyncGenerator

import httpx

_log = logging.getLogger("ui.mad_client")


class MADClient:
    """Client for a single State 4 MAD."""

    def __init__(self, name: str, base_url: str, hostname: str = ""):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.hostname = hostname or name

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/health", timeout=5)
                return resp.json()
        except (httpx.HTTPError, ValueError):
            return {"status": "unreachable"}

    async def list_models(self) -> list[str]:
        """List available models by querying emad_instances + host."""
        models = ["host"]
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": "host",
                        "messages": [{"role": "user", "content": "List all installed eMADs. Just give me the model names, nothing else."}],
                    },
                    timeout=60,
                )
                # We can't easily parse model names from the response.
                # Instead, just return what we know from config.
        except (httpx.HTTPError, ValueError):
            pass
        return models

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
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        else:
            payload["conversation_id"] = "new"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        conversation_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions. Falls back to non-streaming if SSE not available."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        else:
            payload["conversation_id"] = "new"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
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
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/mcp", json=payload, timeout=60)
            resp.raise_for_status()
            body = resp.json()
            if "error" in body:
                raise RuntimeError(f"MCP error: {body['error']}")
            text = body.get("result", {}).get("content", [{}])[0].get("text", "{}")
            return json.loads(text)
