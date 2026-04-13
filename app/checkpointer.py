"""
Shared PostgresSaver checkpointer for all TEs.

Provides a singleton PostgresSaver connected to the pMAD's Postgres.
All stategraphs use this for conversation persistence.
"""

import logging
import os

_log = logging.getLogger("emad_host.checkpointer")

_checkpointer = None
_connection = None


def get_checkpointer():
    """Return a PostgresSaver connected to the pMAD's Postgres.

    Singleton — created once, reused by all stategraphs.
    Creates the checkpoint tables on first call.
    """
    global _checkpointer, _connection
    if _checkpointer is not None:
        return _checkpointer

    from langgraph.checkpoint.postgres import PostgresSaver
    import psycopg

    host = os.environ.get("POSTGRES_HOST", "emad-host-postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "emad_host")
    user = os.environ.get("POSTGRES_USER", "emad_host")
    password = os.environ.get("POSTGRES_PASSWORD", "")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    _connection = psycopg.connect(dsn)
    _checkpointer = PostgresSaver(conn=_connection)
    _checkpointer.setup()
    _log.info("PostgresSaver checkpointer initialized")

    return _checkpointer
