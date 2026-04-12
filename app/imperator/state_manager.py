"""
Imperator persistent state manager.

Manages the Imperator's conversation_id across restarts.
Reads/writes /data/imperator_state.json.

In CB mode (emad_host.url configured): creates conversations on the
remote Context Broker and tracks the conversation_id locally.
In no-CB mode: no conversation tracking needed — returns None.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import httpx

_log = logging.getLogger("emad_host.imperator.state_manager")

IMPERATOR_STATE_FILE = Path("/data/imperator_state.json")


class ImperatorStateManager:
    """Manages the Imperator's persistent conversation state."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._conversation_id: Optional[uuid.UUID] = None

    def _get_cb_url(self) -> Optional[str]:
        """Return the Context Broker URL from config, or None."""
        cb_config = self._config.get("context_broker", {})
        if not cb_config:
            return None
        url = cb_config.get("url")
        if not url or url == "null":
            return None
        return url.rstrip("/")

    async def initialize(self) -> None:
        """Initialize the Imperator's conversation state.

        CB mode: reads state file, verifies/creates conversation on CB.
        No-CB mode: no-op.
        """
        cb_url = self._get_cb_url()
        if not cb_url:
            _log.info("Imperator: no Context Broker configured — no conversation tracking")
            return

        loop = asyncio.get_running_loop()
        saved_conv_id = await loop.run_in_executor(None, self._read_state_file)

        if saved_conv_id is not None:
            self._conversation_id = saved_conv_id
            _log.info("Imperator: resuming conversation %s", self._conversation_id)
            return

        # Create new conversation on the CB
        self._conversation_id = await self._create_imperator_conversation(cb_url)
        await loop.run_in_executor(None, self._write_state_file, self._conversation_id)
        _log.info("Imperator: created new conversation %s on CB", self._conversation_id)

    async def get_conversation_id(self) -> Optional[uuid.UUID]:
        """Return the Imperator's current conversation ID, or None in no-CB mode."""
        return self._conversation_id

    async def get_context_window_id(self) -> Optional[uuid.UUID]:
        """Backward compatibility — returns conversation_id."""
        return await self.get_conversation_id()

    def _read_state_file(self) -> Optional[uuid.UUID]:
        """Read the conversation ID from the state file."""
        if not IMPERATOR_STATE_FILE.exists():
            return None

        try:
            with open(IMPERATOR_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)

            conv_str = data.get("conversation_id")
            if conv_str:
                return uuid.UUID(conv_str)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            _log.warning("Failed to read imperator state file: %s", exc)

        return None

    def _write_state_file(self, conversation_id: uuid.UUID) -> None:
        """Write the conversation ID to the state file."""
        try:
            IMPERATOR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(IMPERATOR_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"conversation_id": str(conversation_id)}, f)
        except OSError as exc:
            _log.error("Failed to write imperator state file: %s", exc)

    async def _create_imperator_conversation(self, cb_url: str) -> uuid.UUID:
        """Create a new conversation on the Context Broker via MCP."""
        new_id = uuid.uuid4()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tools/call",
                    "params": {
                        "name": "conv_create_conversation",
                        "arguments": {
                            "conversation_id": str(new_id),
                            "title": "Imperator — System Conversation",
                        },
                    },
                }
                response = await client.post(f"{cb_url}/mcp", json=payload)
                response.raise_for_status()
                result = response.json()
                if "error" in result:
                    raise RuntimeError(f"CB conv_create_conversation failed: {result['error']}")
                conv_id_str = result.get("result", {}).get("conversation_id", str(new_id))
                return uuid.UUID(conv_id_str)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _log.warning("Failed to create conversation on CB: %s — using local ID", exc)
            return new_id
