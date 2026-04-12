"""
Configuration management for the Context Broker.

Two config files per REQ-002 §7 (TE Configuration Separation):
- AE config (/config/config.yml): infrastructure settings (database, workers, locks)
- TE config (/config/te.yml): cognitive settings (inference, build types, tuning)

AE config is read once at startup (cached, restart required for changes).
TE config is read on each operation (hot-reloadable, no restart needed).
"""

import asyncio
import hashlib
import logging
import os
import threading
from functools import lru_cache
from typing import Any

import yaml

_log = logging.getLogger("emad_host.config")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/config/config.yml")
TE_CONFIG_PATH = os.environ.get("TE_CONFIG_PATH", "/config/te.yml")

# Cached config with mtime check to avoid repeated file reads (M-11).
# os.stat() is near-instant and avoids synchronous file I/O on every call.
_config_cache: dict[str, Any] | None = None
_config_mtime: float = 0.0
_config_content_hash: str = ""

# G5-04: Lock for compound clear-and-set operations on caches.
# Individual dict ops are atomic under CPython's GIL, but the
# clear-LLM-cache + clear-embeddings-cache sequence in load_config
# must be atomic to prevent a concurrent reader from seeing a
# half-cleared state.
_cache_lock = threading.Lock()


def invalidate_config_cache() -> None:
    """Force the next load_config/async_load_config to re-read from disk.

    Called after config_write to ensure the new value is picked up
    immediately, regardless of filesystem mtime resolution.
    """
    global _config_cache, _config_mtime, _config_content_hash
    with _cache_lock:
        _config_cache = None
        _config_mtime = 0.0
        _config_content_hash = ""


def _read_and_parse_config() -> tuple[dict[str, Any], str]:
    """Read config.yml from disk and return (parsed_dict, raw_text).

    Separated from load_config() so that async_load_config() can
    offload only this blocking portion to run_in_executor.
    """
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = f.read()
        config = yaml.safe_load(raw)
        if not isinstance(config, dict):
            raise ValueError("config.yml must be a YAML mapping at the top level")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Configuration file not found at {CONFIG_PATH}. "
            "Mount /config/config.yml into the container."
        ) from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse {CONFIG_PATH}: {exc}") from exc
    return config, raw


def _apply_config(
    config: dict[str, Any], raw: str, current_mtime: float
) -> dict[str, Any]:
    """Update global cache state after a successful config read.

    Shared by both load_config() and async_load_config().
    R5-M24: All global state updates are performed inside _cache_lock
    to prevent concurrent readers from seeing a half-updated state.
    """
    global _config_cache, _config_mtime, _config_content_hash

    new_hash = hashlib.sha256(raw.encode()).hexdigest()
    with _cache_lock:
        if new_hash != _config_content_hash and _config_content_hash != "":
            _log.info(
                "Config file content changed — clearing LLM and embeddings caches"
            )
            _llm_cache.clear()
            _embeddings_cache.clear()

        _config_cache = config
        _config_mtime = current_mtime
        _config_content_hash = new_hash
    return config


def load_config() -> dict[str, Any]:
    """Load and return the AE configuration from /config/config.yml.

    Uses mtime-based caching with content hash invalidation.
    Infrastructure settings only — for cognitive/TE settings use
    load_te_config() or load_merged_config().

    Raises RuntimeError if the file cannot be read or parsed.
    """
    global _config_cache, _config_mtime, _config_content_hash

    try:
        current_mtime = os.stat(CONFIG_PATH).st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Configuration file not found at {CONFIG_PATH}. "
            "Mount /config/config.yml into the container."
        ) from exc

    if _config_cache is not None and current_mtime == _config_mtime:
        return _config_cache

    config, raw = _read_and_parse_config()
    return _apply_config(config, raw, current_mtime)


def load_merged_config() -> dict[str, Any]:
    """Load and return merged AE + TE configuration.

    Reads both config.yml (AE) and imperator.yml (TE), merging into a
    single dict. TE keys take precedence. This provides backward
    compatibility — callers get a single config dict containing both
    infrastructure and cognitive settings.

    If imperator.yml does not exist, returns AE config only (graceful
    fallback for legacy single-file deployments).
    """
    ae_config = load_config()
    try:
        te_config = load_te_config()
    except RuntimeError:
        # TE config not found — legacy deployment with single config.yml
        _log.info("No TE config found at %s — using AE config only", TE_CONFIG_PATH)
        return ae_config
    # Merge: AE is base, TE overlays
    merged = {**ae_config, **te_config}
    return merged


async def async_load_config() -> dict[str, Any]:
    """Async wrapper that returns merged AE + TE configuration.

    This is the primary config function for route handlers and flow nodes.
    Returns the merged config so callers get both infrastructure and
    cognitive settings in one dict.

    CR-M11: AE config read is offloaded to run_in_executor on cache miss
    to avoid blocking the event loop with synchronous file I/O.
    """
    global _config_cache, _config_mtime

    # AE config: fast-path if cached
    try:
        ae_mtime = os.stat(CONFIG_PATH).st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Configuration file not found at {CONFIG_PATH}. "
            "Mount /config/config.yml into the container."
        ) from exc

    if _config_cache is not None and ae_mtime == _config_mtime:
        ae_config = _config_cache
    else:
        loop = asyncio.get_running_loop()
        config, raw = await loop.run_in_executor(None, _read_and_parse_config)
        ae_config = _apply_config(config, raw, ae_mtime)

    # Fast-path: check TE config mtime without async overhead
    try:
        te_mtime = os.stat(TE_CONFIG_PATH).st_mtime
    except FileNotFoundError:
        return ae_config

    if _te_config_cache is not None and te_mtime == _te_config_mtime:
        return {**ae_config, **_te_config_cache}

    loop = asyncio.get_running_loop()
    te_config, raw = await loop.run_in_executor(None, _read_and_parse_te_config)
    te = _apply_te_config(te_config, raw, te_mtime)
    return {**ae_config, **te}


@lru_cache(maxsize=1)
def load_startup_config() -> dict[str, Any]:
    """Load AE configuration once at startup for infrastructure settings.

    Cached — changes require container restart.
    """
    return load_config()


# ============================================================
# TE Configuration (imperator.yml) — hot-reloadable
# ============================================================

_te_config_cache: dict[str, Any] | None = None
_te_config_mtime: float = 0.0
_te_config_content_hash: str = ""


def _read_and_parse_te_config() -> tuple[dict[str, Any], str]:
    """Read imperator.yml from disk and return (parsed_dict, raw_text)."""
    try:
        with open(TE_CONFIG_PATH, encoding="utf-8") as f:
            raw = f.read()
        config = yaml.safe_load(raw)
        if not isinstance(config, dict):
            raise ValueError("te.yml must be a YAML mapping at the top level")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"TE configuration file not found at {TE_CONFIG_PATH}. "
            "Mount /config/te.yml into the container."
        ) from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse {TE_CONFIG_PATH}: {exc}") from exc
    return config, raw


def _apply_te_config(
    config: dict[str, Any], raw: str, current_mtime: float
) -> dict[str, Any]:
    """Update TE config cache after a successful read."""
    global _te_config_cache, _te_config_mtime, _te_config_content_hash

    new_hash = hashlib.sha256(raw.encode()).hexdigest()
    with _cache_lock:
        if new_hash != _te_config_content_hash and _te_config_content_hash != "":
            _log.info(
                "TE config file content changed — clearing LLM and embeddings caches"
            )
            _llm_cache.clear()
            _embeddings_cache.clear()

        _te_config_cache = config
        _te_config_mtime = current_mtime
        _te_config_content_hash = new_hash
    return config


def load_te_config() -> dict[str, Any]:
    """Load and return the TE configuration from /config/te.yml.

    Uses mtime-based caching with content hash invalidation.
    Hot-reloadable — changes take effect without restart.
    """
    global _te_config_cache, _te_config_mtime

    try:
        current_mtime = os.stat(TE_CONFIG_PATH).st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"TE configuration file not found at {TE_CONFIG_PATH}. "
            "Mount /config/te.yml into the container."
        ) from exc

    if _te_config_cache is not None and current_mtime == _te_config_mtime:
        return _te_config_cache

    config, raw = _read_and_parse_te_config()
    return _apply_te_config(config, raw, current_mtime)


async def async_load_te_config() -> dict[str, Any]:
    """Async wrapper for load_te_config().

    Route handlers and flow nodes should prefer this over load_te_config().
    """
    global _te_config_cache, _te_config_mtime

    try:
        current_mtime = os.stat(TE_CONFIG_PATH).st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"TE configuration file not found at {TE_CONFIG_PATH}. "
            "Mount /config/te.yml into the container."
        ) from exc

    if _te_config_cache is not None and current_mtime == _te_config_mtime:
        return _te_config_cache

    loop = asyncio.get_running_loop()
    config, raw = await loop.run_in_executor(None, _read_and_parse_te_config)
    return _apply_te_config(config, raw, current_mtime)


CREDENTIALS_PATH = os.environ.get("CREDENTIALS_PATH", "/config/credentials/.env")

# Cached credentials with mtime check (same pattern as config cache).
_credentials_cache: dict[str, str] = {}
_credentials_mtime: float = 0.0


def _load_credentials() -> dict[str, str]:
    """Read the credentials .env file and return a dict of key=value pairs.

    Re-reads from disk when the file's mtime changes, enabling hot-reload
    of API keys without container restart (PG-39, REQ-001 §8.3).
    Falls back to os.environ for keys not found in the file.
    """
    global _credentials_cache, _credentials_mtime

    try:
        current_mtime = os.stat(CREDENTIALS_PATH).st_mtime
    except (OSError, FileNotFoundError):
        # No credentials file — fall back to environment only
        return {}

    if current_mtime == _credentials_mtime and _credentials_cache:
        return _credentials_cache

    creds: dict[str, str] = {}
    try:
        with open(CREDENTIALS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    creds[key.strip()] = value.strip()
    except (OSError, UnicodeDecodeError) as exc:
        _log.warning("Failed to read credentials file %s: %s", CREDENTIALS_PATH, exc)
        return _credentials_cache  # Return stale cache rather than nothing

    _credentials_cache = creds
    _credentials_mtime = current_mtime
    return creds


def get_api_key(provider_config: dict[str, Any]) -> str:
    """Resolve an API key for a provider.

    Every inference slot should specify api_key_env naming the environment
    variable that holds its API key. No defaults, no magic — explicit is
    better than implicit.

    Reads from the mounted credentials file first (hot-reloadable),
    then falls back to environment variables. This allows API keys to
    be changed without container restart (PG-39, REQ-001 §8.3).

    If api_key_env is absent or empty, returns empty string (keyless
    providers like Ollama).
    """
    env_var_name = provider_config.get("api_key_env", "")
    if not env_var_name:
        return ""
    # Try credentials file first (hot-reloadable)
    creds = _load_credentials()
    if env_var_name in creds:
        return creds[env_var_name]
    # Fall back to environment variable
    return os.environ.get(env_var_name, "")


def get_build_type_config(
    config: dict[str, Any], build_type_name: str
) -> dict[str, Any]:
    """Return the configuration for a named build type.

    Reads from TE config (imperator.yml). The config parameter may be
    either the full TE config or a legacy combined config — checks both.
    Raises ValueError if the build type is not defined.
    """
    build_types = config.get("build_types", {})
    if build_type_name not in build_types:
        raise ValueError(
            f"Build type '{build_type_name}' not found in config.yml. "
            f"Available: {list(build_types.keys())}"
        )
    return build_types[build_type_name]


def get_tuning(config: dict, key: str, default: Any) -> Any:
    """Return a tuning parameter from config, with fallback to default.

    Checks TE config tuning first, then AE config (for worker/lock settings
    that moved to AE). Accepts either the TE config or a legacy combined config.
    """
    # Check tuning section in the provided config
    val = config.get("tuning", {}).get(key, None)
    if val is not None:
        return val
    # Check workers and locks sections (AE config keys)
    val = config.get("workers", {}).get(key, None)
    if val is not None:
        return val
    val = config.get("locks", {}).get(key, None)
    if val is not None:
        return val
    return default


def get_log_level(config: dict[str, Any]) -> str:
    """Return the configured log level, defaulting to INFO."""
    return config.get("log_level", "INFO").upper()


def verbose_log(config: dict, logger: Any, message: str, *args: Any) -> None:
    """Log a message only if verbose_logging is enabled in tuning config.

    REQ-001 section 4.8: Verbose pipeline logging for node entry/exit with timing.
    Accepts printf-style args for lazy formatting.
    """
    if get_tuning(config, "verbose_logging", False):
        logger.info(message, *args)


def verbose_log_auto(logger: Any, message: str, *args: Any) -> None:
    """Log a verbose message by reading config on each call.

    For flows that don't carry config in their state.
    Falls back to no-op if config cannot be loaded.
    """
    try:
        cfg = load_config()
        if get_tuning(cfg, "verbose_logging", False):
            logger.info(message, *args)
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        # R6-m6: Broadened to catch bad YAML structure (ValueError/TypeError)
        # in addition to file-level errors. Verbose logging must never crash
        # the operation, but log the failure at DEBUG for diagnosability.
        logging.getLogger("emad_host.config").debug(
            "verbose_log_auto: config load failed: %s", exc
        )


# ============================================================
# Cached LLM / Embedding client factories (M-09)
# ============================================================

# G5-04: Individual dict operations (__getitem__, __setitem__, __contains__)
# are atomic under CPython's GIL, so these caches are safe for concurrent
# async reads/writes without an explicit asyncio.Lock. Compound operations
# (clear-and-set in load_config) are protected by _cache_lock above.
_llm_cache: dict[str, Any] = {}
_embeddings_cache: dict[str, Any] = {}

# R2-F07: Simple bounded-cache eviction threshold.
_MAX_CACHE_ENTRIES = 10


def get_chat_model(config: dict, role: str = "imperator", *, streaming: bool = False) -> Any:
    """Return a cached ChatOpenAI instance for the given role.

    Role determines where to look for the LLM config in the merged config:
    - "imperator": from TE config (config["imperator"])
    - "summarization": from AE config (config["summarization"])
    - "extraction": from AE config (config["extraction"])
    - Fallback: config["llm"] for legacy single-config deployments

    All providers use ChatOpenAI — the OpenAI-compatible wire protocol is
    universal (OpenAI, Anthropic, Google, xAI, Together, Ollama all support it).

    Args:
        streaming: If True, returns a streaming-enabled LLM instance.
            Required for astream_events to emit on_chat_model_stream events.
            Non-streaming (default) avoids "No generations found in stream"
            errors from providers when tool-call-only turns produce no content.

    R5-M12: Uses _cache_lock around the full check-and-set.
    """
    from langchain_openai import ChatOpenAI

    # Resolve config for this role
    llm_config = config.get(role, {})
    if not llm_config:
        llm_config = config.get("llm", {})

    api_key = get_api_key(llm_config)
    stream_suffix = ":stream" if streaming else ""
    cache_key = f"{role}:{llm_config.get('base_url')}:{llm_config.get('model')}:{hashlib.sha256((api_key or 'none').encode()).hexdigest()[:16]}{stream_suffix}"

    with _cache_lock:
        if cache_key not in _llm_cache:
            if len(_llm_cache) >= _MAX_CACHE_ENTRIES:
                oldest_key = next(iter(_llm_cache))
                del _llm_cache[oldest_key]

            kwargs = {
                "base_url": llm_config.get("base_url"),
                "model": llm_config.get("model", "gpt-4o-mini"),
                "api_key": api_key or "not-needed",
                "timeout": get_tuning(config, "llm_timeout_seconds", 1800),
                "streaming": streaming,
            }
            _llm_cache[cache_key] = ChatOpenAI(**kwargs)
        return _llm_cache[cache_key]


def get_embeddings_model(config: dict, config_key: str = "embeddings") -> Any:
    """Return a cached OpenAIEmbeddings instance keyed by (base_url, model).

    Avoids re-creating the client on every request.
    R5-M12: Uses _cache_lock around the full check-and-set to prevent
    two concurrent calls from both missing the cache and creating
    duplicate clients.

    Args:
        config: Full configuration dict.
        config_key: Config section to read embedding settings from.
                    Default "embeddings". Use "log_embeddings" for log vectorization.
    """
    from langchain_openai import OpenAIEmbeddings

    embeddings_config = config.get(config_key, {})
    api_key = get_api_key(embeddings_config)
    cache_key = f"{embeddings_config.get('base_url')}:{embeddings_config.get('model')}:{hashlib.sha256((api_key or 'default').encode()).hexdigest()[:16]}"
    with _cache_lock:
        if cache_key not in _embeddings_cache:
            if len(_embeddings_cache) >= _MAX_CACHE_ENTRIES:
                oldest_key = next(iter(_embeddings_cache))
                del _embeddings_cache[oldest_key]
            kwargs = {
                "model": embeddings_config.get("model", "text-embedding-3-small"),
                "base_url": embeddings_config.get("base_url"),
                "api_key": api_key or "not-needed",
                "tiktoken_enabled": embeddings_config.get("tiktoken_enabled", False),
                "check_embedding_ctx_length": embeddings_config.get(
                    "check_embedding_ctx_length", False
                ),
            }
            # MRL: pass dimensions parameter if configured, enabling
            # Matryoshka truncation for models that support it (OpenAI v3, Gemini v2).
            dims = embeddings_config.get("embedding_dims")
            if dims:
                kwargs["dimensions"] = int(dims)
            _embeddings_cache[cache_key] = OpenAIEmbeddings(**kwargs)
        return _embeddings_cache[cache_key]
