"""
Pydantic models for request/response validation.

All external inputs are validated through these models before
reaching StateGraph flows.
"""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================
# MCP Tool Input Models
# ============================================================


class ImperatorChatInput(BaseModel):
    """Input for imperator_chat."""

    message: str = Field(..., min_length=1, max_length=32000)
    context_window_id: Optional[UUID] = None


class MetricsGetInput(BaseModel):
    """Input for metrics_get (no required fields)."""

    pass


# ============================================================
# OpenAI-compatible chat models
# ============================================================


class ChatMessage(BaseModel):
    """A single message in an OpenAI-compatible chat request."""

    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request body."""

    model: str = Field(default="emad-host")
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    user: Optional[str] = Field(None, max_length=255)


# ============================================================
# MCP Protocol Models
# ============================================================


class MCPToolCall(BaseModel):
    """MCP JSON-RPC tools/call request."""

    jsonrpc: str = Field(default="2.0")
    id: Optional[Any] = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """MCP JSON-RPC tools/call response."""

    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
