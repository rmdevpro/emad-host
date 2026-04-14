"""Notification tools — send alerts via webhook.

The webhook URL is configured in te.yml under imperator.notification_webhook.

Default: http://pmad-template-alerter:8000/webhook (the alerter sidecar).
The alerter routes, formats via LLM, and fans out to Slack/Discord/ntfy/SMTP.

Alternative: point directly at an external service (ntfy.sh, Slack) to
skip the alerter entirely.
"""

import asyncio
import logging
import socket
from datetime import datetime, timezone

from app.config import load_merged_config
from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.notify")

@tool
async def send_notification(
    message: str,
    event_type: str = "imperator.notification",
    severity: str = "info",
    title: str = "",
) -> str:
    """Send a notification via the configured webhook.

    Use this to alert the user about important events: health issues,
    completed tasks, errors that need attention, or scheduled reports.

    When the alerter sidecar is configured (default), sends CloudEvents
    format — the alerter handles routing, LLM formatting, and fan-out.
    When pointing at an external service, sends a format appropriate
    for that service.

    Args:
        message: Notification content.
        event_type: CloudEvents type for routing (default "imperator.notification").
            Use dot-notation categories like "health.degraded", "extraction.complete",
            "pipeline.error", "schedule.report".
        severity: Level — "info", "warning", "error", "critical" (default "info").
        title: Optional short title for the notification.
    """
    try:

        loop = asyncio.get_running_loop()
        config = await loop.run_in_executor(None, load_merged_config)
        webhook_url = config.get("imperator", {}).get(
            "notification_webhook", "http://emad-host-alerter:8000/webhook"
        )

        import httpx
        import ssl

        # Detect if pointing at the alerter (CloudEvents) or external service
        is_alerter = "alerter" in webhook_url and "/webhook" in webhook_url
        is_ntfy = "ntfy.sh" in webhook_url or "ntfy" in webhook_url

        if is_alerter:
            # CloudEvents format for the alerter sidecar
            payload = {
                "type": event_type,
                "source": socket.gethostname(),
                "subject": title or severity,
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "message": message,
                    "severity": severity,
                    "title": title,
                },
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    webhook_url, json=payload, timeout=10
                )
                resp.raise_for_status()
                result = resp.json()
                succeeded = result.get("channels_succeeded", [])
                failed = result.get("channels_failed", [])
                return (
                    f"Alert sent ({event_type}): {len(succeeded)} channels succeeded"
                    + (f", {len(failed)} failed" if failed else "")
                )

        elif is_ntfy:
            # ntfy.sh native format
            headers = {
                "Title": title or f"Imperator [{severity}]",
                "Priority": {
                    "info": "default",
                    "warning": "high",
                    "error": "urgent",
                    "critical": "max",
                }.get(severity, "default"),
                "Tags": severity,
            }
            async with httpx.AsyncClient(
                verify=ssl.create_default_context()
            ) as client:
                resp = await client.post(
                    webhook_url, content=message, headers=headers, timeout=10
                )
                resp.raise_for_status()

        else:
            # Generic webhook — JSON payload
            payload = {
                "text": message,
                "title": title or f"Imperator [{severity}]",
                "severity": severity,
                "type": event_type,
            }
            async with httpx.AsyncClient(
                verify=ssl.create_default_context()
            ) as client:
                resp = await client.post(
                    webhook_url, json=payload, timeout=10
                )
                resp.raise_for_status()

        return f"Notification sent ({severity}): {message[:100]}"
    except ImportError:
        return "Notifications unavailable — httpx not installed."
    except (httpx.HTTPError, RuntimeError, OSError, ValueError) as exc:
        return f"Notification failed: {exc}"


def get_tools(te_config: dict | None = None) -> list:
    """Return notification tools if webhook is configured."""
    if te_config and te_config.get("notification_webhook"):
        return [send_notification]
    # Always return the tool — it will explain configuration is needed when called
    return [send_notification]
