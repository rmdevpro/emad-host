"""Admin tools — gated by admin_tools: true in TE config.

Configuration reading/writing, verbose toggle, database queries.
"""

import asyncio
import copy
import logging
import re

import asyncpg
import yaml
from app.config import CONFIG_PATH, TE_CONFIG_PATH, async_load_config, get_tuning, load_config
from app.database import get_pg_pool
from langchain_core.tools import tool


_log = logging.getLogger("emad_host.tools.admin")


def _redact_config(config: dict) -> dict:
    """Return a deep copy of *config* with sensitive values redacted (G5-16).

    Removes the top-level ``credentials`` section entirely and replaces any
    value whose key matches common secret patterns (api_key, secret, token,
    password) with ``"***REDACTED***"``.
    """
    redacted = copy.deepcopy(config)
    redacted.pop("credentials", None)

    _secret_key_re = re.compile(r"(api_key|secret|_token|password)", re.IGNORECASE)

    def _walk(obj: dict | list) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if _secret_key_re.search(key) and obj[key]:
                    obj[key] = "***REDACTED***"
                elif isinstance(obj[key], (dict, list)):
                    _walk(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _walk(item)

    _walk(redacted)
    return redacted


@tool
async def config_read() -> str:
    """Read the current config.yml contents (sensitive values are redacted).

    Admin-only tool. Returns the configuration as YAML text with credentials
    and API keys redacted for safety.

    R7-M2: File read wrapped in run_in_executor to avoid blocking the event loop.
    """
    import asyncio


    def _sync_read():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)

    try:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, _sync_read)
        sanitized = _redact_config(raw)
        return yaml.dump(sanitized, default_flow_style=False)
    except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
        return f"Error reading config: {exc}"


@tool
async def db_query(sql: str) -> str:
    """Execute a read-only SQL query against the Context Broker database.

    Admin-only tool. The transaction is set to READ ONLY mode, so any
    DML/DDL will be rejected by PostgreSQL regardless of query structure.
    A 5-second statement timeout prevents expensive queries.

    Args:
        sql: A SQL query to execute (enforced read-only at the DB level).
    """
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute("SET LOCAL statement_timeout = '5000'")
                rows = await conn.fetch(sql)
        if not rows:
            return "No results."
        columns = list(rows[0].keys())
        lines = [" | ".join(columns)]
        for row in rows[:50]:
            lines.append(" | ".join(str(row[c]) for c in columns))
        return "\n".join(lines)
    except (asyncpg.PostgresError, OSError, RuntimeError) as exc:
        return f"Query error: {exc}"


@tool
async def config_write(key: str, value: str) -> str:
    """Write a value to the AE configuration (config.yml).

    Changes are hot-reloaded — they take effect on the next operation
    without a container restart. Only AE config keys are writable;
    TE config (Identity, Purpose, system prompt) cannot be modified.

    Args:
        key: Dot-notation config path (e.g., "summarization.model", "tuning.verbose_logging").
        value: New value as a string. Numbers and booleans are auto-converted.
    """

    te_keys = ["imperator", "system_prompt", "identity", "purpose"]
    if any(key.startswith(k) for k in te_keys):
        return (
            f"Cannot modify TE config key '{key}'. "
            "TE configuration is the architect's domain."
        )

    def _sync_write():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                return f"Config path '{key}' not found."
            target = target[part]

        if parts[-1] not in target:
            return (
                f"Config key '{key}' not found. "
                f"Available keys at this level: {list(target.keys())}"
            )

        old_value = target[parts[-1]]
        if isinstance(old_value, bool):
            value_typed = value.lower() in ("true", "1", "yes")
        elif isinstance(old_value, int):
            value_typed = int(value)
        elif isinstance(old_value, float):
            value_typed = float(value)
        else:
            value_typed = value

        target[parts[-1]] = value_typed

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

        return (
            f"Updated '{key}': {old_value} → {value_typed}. "
            "Change will take effect on next operation (hot-reload)."
        )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _sync_write)
        # Invalidate config cache so next read picks up the new value
        # immediately, regardless of filesystem mtime resolution.
        from app.config import invalidate_config_cache
        invalidate_config_cache()
        return result
    except (FileNotFoundError, OSError, yaml.YAMLError, ValueError) as exc:
        return f"Config write error: {exc}"


@tool
async def verbose_toggle() -> str:
    """Toggle verbose pipeline logging on or off.

    Reads the current value of tuning.verbose_logging from config and
    writes the opposite. Changes take effect immediately (hot-reload).
    """
    current = get_tuning(load_config(), "verbose_logging", False)
    new_value = "false" if current else "true"
    return await config_write.ainvoke(
        {"key": "tuning.verbose_logging", "value": new_value}
    )


def _load_inference_models_sync() -> dict:
    """Load the inference-models.yml catalog (sync, for use in executor)."""
    import os
    from pathlib import Path

    catalog_path = Path(os.environ.get("CONFIG_DIR", "/config")) / "inference-models.yml"
    if not catalog_path.exists():
        return {}
    with open(catalog_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


async def _load_inference_models() -> dict:
    """Load the inference-models.yml catalog without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _load_inference_models_sync)


async def _test_endpoint(base_url: str, api_key_env: str, model: str) -> str | None:
    """Test that an inference endpoint is reachable. Returns None on success, error on failure."""
    import httpx
    import os

    headers = {}
    if api_key_env:
        key = os.environ.get(api_key_env, "")
        if not key:
            return f"API key env var '{api_key_env}' is not set"
        headers["Authorization"] = f"Bearer {key}"

    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=10)
            if resp.status_code in (200, 401):
                # 401 = auth works differently (e.g., Anthropic) — endpoint is reachable
                return None
            return f"Endpoint returned {resp.status_code}"
    except (httpx.HTTPError, OSError) as exc:
        return f"Cannot reach {url}: {exc}"


@tool
async def change_inference(slot: str, provider: str = "", model: str = "") -> str:
    """Change the inference model for a given slot, or list available models.

    Handles all inference slots: imperator, summarization, extraction, embeddings.

    Without provider/model arguments, lists available models for the slot from
    the inference-models.yml catalog.

    With provider and model, tests the endpoint and switches to the new model.
    For embeddings, this triggers a full re-embed (wipe all embeddings, reset
    extraction flags) — the user is warned and must call again with confirm.

    Args:
        slot: Inference slot to change. One of: imperator, summarization, extraction, embeddings.
        provider: Provider name (e.g., "openai", "google", "ollama"). Empty to list options.
        model: Model identifier (e.g., "gpt-4.1-mini"). Empty to list options.
    """
    valid_slots = ["imperator", "summarization", "extraction", "embeddings"]
    if slot not in valid_slots:
        return f"Invalid slot '{slot}'. Must be one of: {', '.join(valid_slots)}"

    catalog = await _load_inference_models()

    # List mode: show available models for this slot
    if not provider or not model:
        slot_models = catalog.get(slot, {})
        if not slot_models:
            return f"No models in catalog for slot '{slot}'. Check inference-models.yml."

        config = await async_load_config()
        if slot == "imperator":
            current = config.get("imperator", {}).get("model", "unknown")
        elif slot == "embeddings":
            current = config.get("embeddings", {}).get("model", "unknown")
        else:
            current = config.get(slot, {}).get("model", "unknown")

        lines = [f"Available models for '{slot}' (current: {current}):\n"]
        for prov, models in slot_models.items():
            lines.append(f"  {prov}:")
            for m in models:
                marker = " ← current" if m["model"] == current else ""
                dims = f", {m['embedding_dims']} dims" if "embedding_dims" in m else ""
                lines.append(f"    - {m['model']}{dims}{marker}")
                if m.get("notes"):
                    lines.append(f"      {m['notes']}")
        lines.append(
            f"\nTo switch, call: change_inference(slot='{slot}', "
            f"provider='<provider>', model='<model>')"
        )
        return "\n".join(lines)

    # Switch mode: find the model in the catalog
    slot_models = catalog.get(slot, {})
    provider_models = slot_models.get(provider, [])
    match = None
    for m in provider_models:
        if m["model"] == model:
            match = m
            break
    if not match:
        available = [m["model"] for models in slot_models.values() for m in models]
        return (
            f"Model '{model}' from provider '{provider}' not found in catalog for "
            f"slot '{slot}'. Available: {available}"
        )

    # Test the endpoint before switching
    err = await _test_endpoint(match["base_url"], match.get("api_key_env", ""), model)
    if err:
        return f"Endpoint test failed: {err}. Model not switched."

    # Determine which config file to modify
    if slot == "imperator":
        target_path = TE_CONFIG_PATH
        config_section = "imperator"
    else:
        target_path = CONFIG_PATH
        config_section = slot

    # For embeddings, warn about destructive migration
    if slot == "embeddings":
        config = await async_load_config()
        current_model = config.get("embeddings", {}).get("model", "unknown")
        current_dims = config.get("embeddings", {}).get("embedding_dims", "unknown")
        new_dims = match.get("embedding_dims", current_dims)

        pool = get_pg_pool()
        msg_count = await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_messages WHERE embedding IS NOT NULL"
        )

        return (
            f"EMBEDDING CHANGE requires full re-embed. This will:\n"
            f"  Current: {current_model} ({current_dims} dims)\n"
            f"  New: {model} ({new_dims} dims)\n"
            f"  - Wipe {msg_count} message embeddings\n"
            f"  - Wipe log and domain info embeddings\n"
            f"  - Reset memory extraction flags\n"
            f"  - Background workers will re-process everything\n"
            f"\n"
            f"To confirm, call: migrate_embeddings(new_model='{model}', "
            f"new_dims={new_dims}, confirm=true)"
        )

    # Apply the change
    def _sync_apply():
        with open(target_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        cfg.setdefault(config_section, {})["model"] = match["model"]
        cfg[config_section]["base_url"] = match["base_url"]
        if match.get("api_key_env"):
            cfg[config_section]["api_key_env"] = match["api_key_env"]
        elif "api_key_env" in cfg.get(config_section, {}):
            cfg[config_section]["api_key_env"] = ""

        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        file_name = "te.yml" if slot == "imperator" else "config.yml"
        return (
            f"Switched {slot} to {model} ({provider}).\n"
            f"Updated {file_name}. Change takes effect on next operation (hot-reload)."
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_apply)
    except (OSError, yaml.YAMLError) as exc:
        return f"Failed to update config: {exc}"


def get_tools() -> list:
    """Return all admin tools."""
    return [
        config_read,
        db_query,
        config_write,
        verbose_toggle,
        change_inference,
    ]
