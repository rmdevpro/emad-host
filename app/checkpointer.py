"""
Shared PostgresSaver checkpointer for all TEs.

Initialized during app startup (lifespan). Stored as module singleton.
All stategraphs use this for conversation persistence.
"""

import logging

_log = logging.getLogger("emad_host.checkpointer")

_checkpointer = None


def set_checkpointer(cp) -> None:
    """Set the checkpointer singleton. Called from main.py lifespan."""
    global _checkpointer
    _checkpointer = cp
    _log.info("Checkpointer set: %s", type(cp).__name__)


def get_checkpointer():
    """Return the checkpointer. Raises if not initialized."""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized — startup not complete")
    return _checkpointer
