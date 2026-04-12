"""
TE/AE decoupling — TEContext protocol and singleton.

Defines the interface the TE needs from the AE kernel, without importing
any ``app.*`` modules.  The AE bootstrap creates a concrete implementation
and calls ``initialize(ctx)`` before the Imperator flow is compiled.

This is the ONLY module TE files import for kernel services.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class TEContext(Protocol):
    """Surface the TE needs from the AE kernel."""

    # ── Properties / fields ──────────────────────────────────────────────
    config_path: str
    te_config_path: str
    effective_utilization_default: float

    # ── Sync helpers ─────────────────────────────────────────────────────
    def get_pool(self) -> Any:
        """Return the asyncpg connection pool."""
        ...

    def load_config(self) -> dict:
        """Return the current AE config (sync, cached/memoized read)."""
        ...

    def load_merged_config(self) -> dict:
        """Return config.yml merged with te.yml overrides."""
        ...

    def get_chat_model(self, config: dict, role: str = "imperator") -> Any:
        """Return a LangChain chat model for *role*."""
        ...

    def get_embeddings_model(self, config: dict, config_key: str = "embeddings") -> Any:
        """Return a LangChain embeddings model."""
        ...

    def get_tuning(self, config: dict, key: str, default: Any = None) -> Any:
        """Read a tuning parameter from *config*."""
        ...

    def get_api_key(self, provider_config: dict) -> str:
        """Resolve an API key from provider config + env vars."""
        ...

    # ── Async helpers ────────────────────────────────────────────────────
    def async_load_config(self) -> Awaitable[dict]:
        """Return the current AE config (async variant)."""
        ...

    def async_load_prompt(self, name: str) -> Awaitable[str]:
        """Load a prompt template by *name*."""
        ...

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict,
        config: dict,
        state: Any,
    ) -> Awaitable[dict]:
        """Dispatch an AE core tool by name."""
        ...

    # ── Flow builders ────────────────────────────────────────────────────
    def get_flow_builder(self, name: str) -> Optional[Callable]:
        """Return a registered AE flow builder by *name*, or ``None``."""
        ...


# ── Module-level singleton ───────────────────────────────────────────────

_ctx: Optional[TEContext] = None


def initialize(ctx: TEContext) -> None:
    """Set the module-level TEContext singleton.

    Called once by the AE bootstrap before the Imperator flow is compiled.
    """
    global _ctx
    _ctx = ctx


def get_ctx() -> TEContext:
    """Return the TEContext singleton.

    Raises ``RuntimeError`` if ``initialize()`` has not been called yet.
    """
    if _ctx is None:
        raise RuntimeError(
            "TEContext not initialized — call emad_host_te._ctx.initialize() "
            "before using TE components."
        )
    return _ctx
