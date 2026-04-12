"""
Structured JSON logging for the Context Broker.

All logs go to stdout in JSON format (one object per line).
Log level is configurable via config.yml.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Include context fields if present
        for field in ("request_id", "tool_name", "conversation_id", "window_id"):
            value = getattr(record, field, None)
            if value is not None:
                entry[field] = value

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry)


class HealthCheckFilter(logging.Filter):
    """Suppress noisy health check request logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message and "GET /health" not in message


def setup_logging() -> None:
    """Configure application logging with JSON formatter.

    Sets up the root logger and suppresses noisy third-party loggers.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(HealthCheckFilter())

    # Configure root logger (R5-m3: guard against duplicate handlers on reload)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger("emad_host").setLevel(logging.INFO)


def update_log_level(level: str) -> None:
    """Update the log level after config is loaded.

    Called from the application lifespan after config.yml is read.
    Accepts standard level names: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.getLogger("emad_host").warning(
            "Invalid log level '%s' in config — keeping INFO", level
        )
        return

    logging.getLogger().setLevel(numeric_level)
    logging.getLogger("emad_host").setLevel(numeric_level)
