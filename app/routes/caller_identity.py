"""Resolve caller identity from HTTP requests.

Used by both the OpenAI chat and MCP endpoints to determine who is
sending messages. The resolved identity is used as the sender field
on stored messages.

Priority: user field from request body → reverse DNS on source IP → "unknown"
"""

import asyncio
import functools
import logging
import socket

from fastapi import Request

_log = logging.getLogger("emad_host.routes.caller_identity")


@functools.lru_cache(maxsize=256)
def _reverse_dns_sync(client_ip: str) -> str:
    """Cached reverse DNS lookup (blocking — run via executor)."""
    try:
        hostname, _, _ = socket.gethostbyaddr(client_ip)
        return hostname
    except (socket.herror, socket.gaierror, OSError):
        _log.debug("Reverse DNS failed for %s", client_ip)
        return client_ip


async def _reverse_dns(client_ip: str) -> str:
    """Non-blocking reverse DNS lookup using a thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _reverse_dns_sync, client_ip)


async def resolve_caller(request: Request, user_field: str | None = None) -> str:
    """Resolve the caller's identity from the HTTP request.

    Args:
        request: The FastAPI request object.
        user_field: The 'user' field from the request body (OpenAI standard).
                    Takes priority over reverse lookup.

    Returns:
        The caller's identity string.
    """
    # Priority 1: explicit user field from request body
    if user_field:
        return user_field

    # Priority 2: reverse DNS lookup on source IP (works in Docker networking)
    if request.client and request.client.host:
        return await _reverse_dns(request.client.host)

    return "unknown"
