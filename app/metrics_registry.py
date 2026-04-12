"""
Prometheus metrics registry.

All metrics are defined here and imported by flows and routes.
Metrics are incremented inside StateGraph nodes, not in route handlers.
"""

from prometheus_client import Counter, Histogram

# MCP tool request metrics
MCP_REQUESTS = Counter(
    "emad_host_mcp_requests_total",
    "Total MCP tool requests",
    ["tool", "status"],
)

MCP_REQUEST_DURATION = Histogram(
    "emad_host_mcp_request_duration_seconds",
    "Duration of MCP tool requests",
    ["tool"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

# Chat endpoint metrics
CHAT_REQUESTS = Counter(
    "emad_host_chat_requests_total",
    "Total chat completion requests",
    ["status"],
)

CHAT_REQUEST_DURATION = Histogram(
    "emad_host_chat_request_duration_seconds",
    "Duration of chat completion requests",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
