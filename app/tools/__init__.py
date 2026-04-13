"""AE Tool Registry — maps tool names to callables.

All tools are AE-owned. TEs call them via MCP. Each TE's config
specifies which tools it can use.
"""

from app.tools.admin import (
    config_read,
    config_write,
    change_inference,
    db_query,
    verbose_toggle,
)
from app.tools.alerting import (
    add_alert_instruction,
    delete_alert_instruction,
    list_alert_instructions,
    update_alert_instruction,
)
from app.tools.diagnostic import log_query
from app.tools.filesystem import (
    file_list,
    file_read,
    file_search,
    file_write,
    read_system_prompt,
    update_system_prompt,
)
from app.tools.notify import send_notification
from app.tools.operational import search_domain_info, store_domain_info
from app.tools.system import run_command
from app.tools.web import web_read, web_search

# Master registry: tool_name -> tool callable
TOOL_REGISTRY: dict = {
    # Admin (host Imperator only)
    "config_read": config_read,
    "config_write": config_write,
    "change_inference": change_inference,
    "db_query": db_query,
    "verbose_toggle": verbose_toggle,
    "read_system_prompt": read_system_prompt,
    "update_system_prompt": update_system_prompt,
    # Alerting
    "add_alert_instruction": add_alert_instruction,
    "list_alert_instructions": list_alert_instructions,
    "update_alert_instruction": update_alert_instruction,
    "delete_alert_instruction": delete_alert_instruction,
    # Diagnostic
    "log_query": log_query,
    # Filesystem
    "file_read": file_read,
    "file_list": file_list,
    "file_search": file_search,
    "file_write": file_write,
    # Notify
    "send_notification": send_notification,
    # Domain info
    "store_domain_info": store_domain_info,
    "search_domain_info": search_domain_info,
    # System
    "run_command": run_command,
    # Web
    "web_search": web_search,
    "web_read": web_read,
}

# Tools restricted to the host Imperator (model: "host")
ADMIN_TOOLS = {
    "config_read",
    "config_write",
    "change_inference",
    "db_query",
    "verbose_toggle",
    "read_system_prompt",
    "update_system_prompt",
}


def get_tools_for_model(model_name: str, tool_names: list[str]) -> list:
    """Return tool callables for the given model, filtered by allowed names.

    Admin tools are only returned if model_name is "host".
    """
    tools = []
    for name in tool_names:
        if name in ADMIN_TOOLS and model_name != "host":
            continue
        tool_fn = TOOL_REGISTRY.get(name)
        if tool_fn is not None:
            tools.append(tool_fn)
    return tools
