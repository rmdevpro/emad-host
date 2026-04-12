"""
KernelTEContext — concrete TEContext backed by ``app.*`` modules.

This is the ONLY file in the TE package that imports from ``app.*``.
It is the adapter boundary between the TE and the AE kernel.

NOTE: This file lives in the TE package for now but conceptually belongs
to the AE/kernel.  It will be moved when the template is created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class KernelTEContext:
    """Delegate every TEContext method to the corresponding ``app.*`` call."""

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def config_path(self) -> str:
        from app.config import CONFIG_PATH

        return CONFIG_PATH

    @property
    def te_config_path(self) -> str:
        from app.config import TE_CONFIG_PATH

        return TE_CONFIG_PATH

    @property
    def effective_utilization_default(self) -> float:
        from app.budget import EFFECTIVE_UTILIZATION_DEFAULT

        return EFFECTIVE_UTILIZATION_DEFAULT

    # ── Sync helpers ─────────────────────────────────────────────────────

    def get_pool(self) -> Any:
        from app.database import get_pg_pool

        return get_pg_pool()

    def load_config(self) -> dict:
        from app.config import load_config

        return load_config()

    def load_merged_config(self) -> dict:
        from app.config import load_merged_config

        return load_merged_config()

    def get_chat_model(self, config: dict, role: str = "imperator") -> Any:
        from app.config import get_chat_model

        return get_chat_model(config, role=role)

    def get_embeddings_model(self, config: dict, config_key: str = "embeddings") -> Any:
        from app.config import get_embeddings_model

        return get_embeddings_model(config, config_key=config_key)

    def get_tuning(self, config: dict, key: str, default: Any = None) -> Any:
        from app.config import get_tuning

        return get_tuning(config, key, default)

    def get_api_key(self, provider_config: dict) -> str:
        from app.config import get_api_key

        return get_api_key(provider_config)

    # ── Async helpers ────────────────────────────────────────────────────

    async def async_load_config(self) -> dict:
        from app.config import async_load_config

        return await async_load_config()

    async def async_load_prompt(self, name: str) -> str:
        from app.prompt_loader import async_load_prompt

        return await async_load_prompt(name)

    async def dispatch_tool(
        self,
        tool_name: str,
        args: dict,
        config: dict,
        state: Any,
    ) -> dict:
        from app.flows.tool_dispatch import dispatch_tool

        return await dispatch_tool(tool_name, args, config, state)

    # ── Flow builders ────────────────────────────────────────────────────

    def get_flow_builder(self, name: str) -> Optional[Callable]:
        from app.stategraph_registry import get_flow_builder

        return get_flow_builder(name)
