"""
Token budget resolution for context windows.

Resolves the max_context_tokens setting for a build type:
- "auto": query the configured LLM provider's model list endpoint
- explicit integer: use that value directly
- caller override: takes precedence over build type default

Token budget is resolved once at window creation and stored.
"""

import logging
from typing import Optional

import httpx

from app.config import get_api_key

_log = logging.getLogger("emad_host.token_budget")


async def resolve_token_budget(
    config: dict,
    build_type_config: dict,
    caller_override: Optional[int] = None,
) -> int:
    """Resolve the token budget for a context window.

    Priority order:
    1. caller_override (explicit max_tokens from the caller)
    2. build_type_config["max_context_tokens"] if it's an integer
    3. Auto-query the LLM provider if max_context_tokens == "auto"
    4. fallback_tokens from build_type_config

    Args:
        config: Full application config (for LLM provider settings).
        build_type_config: The build type configuration dict.
        caller_override: Optional explicit token budget from the caller.

    Returns:
        Resolved token budget as an integer.
    """
    if caller_override is not None and caller_override > 0:
        _log.info("Token budget: using caller override %d", caller_override)
        return caller_override

    max_context_tokens = build_type_config.get("max_context_tokens", "auto")
    fallback_tokens = build_type_config.get("fallback_tokens", 8192)

    if isinstance(max_context_tokens, int) and max_context_tokens > 0:
        _log.info(
            "Token budget: using explicit build type value %d", max_context_tokens
        )
        return max_context_tokens

    if max_context_tokens == "auto":
        resolved = await _query_provider_context_length(config, fallback_tokens)
        _log.info("Token budget: auto-resolved to %d", resolved)
        return resolved

    _log.warning(
        "Token budget: unrecognized max_context_tokens value '%s', using fallback %d",
        max_context_tokens,
        fallback_tokens,
    )
    return fallback_tokens


async def _query_provider_context_length(config: dict, fallback: int) -> int:
    """Query the LLM provider's model list endpoint for context length.

    Returns fallback if the provider doesn't report context length or
    if the request fails.
    """
    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "")
    model = llm_config.get("model", "")
    api_key = get_api_key(llm_config)

    if not base_url or not model:
        _log.warning(
            "Token budget auto-resolution: LLM provider not configured, using fallback %d",
            fallback,
        )
        return fallback

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        models_url = base_url.rstrip("/") + "/models"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(models_url, headers=headers)
            response.raise_for_status()
            data = response.json()

        models = data.get("data") or []
        for model_info in models:
            if model_info.get("id") == model:
                context_length = model_info.get("context_length")
                if isinstance(context_length, int) and context_length > 0:
                    return context_length

        _log.info(
            "Token budget: model '%s' not found in provider model list, using fallback %d",
            model,
            fallback,
        )
        return fallback

    except httpx.HTTPError as exc:
        _log.warning(
            "Token budget: failed to query provider model list: %s, using fallback %d",
            exc,
            fallback,
        )
        return fallback
    except (ValueError, KeyError, OSError) as exc:
        _log.warning(
            "Token budget: unexpected error querying provider: %s, using fallback %d",
            exc,
            fallback,
        )
        return fallback
