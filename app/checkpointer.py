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
    _log.info("Checkpointer set: %s (id=%s)", type(cp).__name__, id(cp))


def get_checkpointer():
    """Return the checkpointer. Raises if not initialized."""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized — startup not complete")
    _log.info("Checkpointer get: %s (id=%s)", type(_checkpointer).__name__, id(_checkpointer))
    return _checkpointer
