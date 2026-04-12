"""
AE/TE Base Contract (REQ-001 §13).

Defines the standard interface between any AE and any TE,
enabling TE portability. TE packages depend on these types
but not on any specific AE implementation.

See also contracts.py for build type graph contracts (ARCH-18).
"""

from typing import Annotated, Callable, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ── What the AE passes to the TE on invocation (§13.2) ─────────────


class TEInputState(TypedDict, total=False):
    """Standard input state for TE invocation.

    AE provides: messages (conversation history), context identifiers,
    and the merged config dict.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    context_window_id: Optional[str]
    conversation_id: Optional[str]
    config: dict


# ── What the TE returns to the AE (§13.3) ──────────────────────────


class TEOutputState(TypedDict, total=False):
    """Standard output state from TE invocation.

    TE provides: final response text, messages to persist,
    and optional error.
    """

    response_text: Optional[str]
    messages: list[AnyMessage]
    error: Optional[str]


# ── What a TE package's register() function returns ─────────────────


class TERegistration(TypedDict, total=False):
    """Registration dict returned by a TE package's entry point.

    The kernel's package_registry.scan() processes this to
    register the Imperator builder and any tools the TE requires.
    """

    identity: str  # What the Imperator is
    purpose: str  # What the Imperator is for
    imperator_builder: Callable  # () -> compiled StateGraph
    tools_required: list[str]  # Tool names the TE needs from AE


# ── What an AE package's register() function returns ────────────────


class AERegistration(TypedDict, total=False):
    """Registration dict returned by an AE package's entry point.

    The kernel's package_registry.scan() processes this to
    register build types and flow builders in the appropriate registries.
    """

    build_types: dict[str, tuple[Callable, Callable]]  # name -> (assembly, retrieval)
    flows: dict[str, Callable]  # name -> builder callable
