"""
emad-generic — Generic eMAD package.

A configurable conversational agent that uses the host pMAD's Imperator
inference settings. No admin tools. System prompt configurable via
instance parameters.

Identity: Generic eMAD
Purpose: General-purpose conversational agent
"""

EMAD_PACKAGE_NAME = "emad-generic"
DESCRIPTION = (
    "Generic conversational eMAD — configurable system prompt, "
    "no admin tools. Uses the host Imperator's inference settings."
)
SUPPORTED_PARAMS = {
    "system_prompt": {
        "type": "string",
        "description": "System prompt text for the eMAD's Imperator. "
        "Defines the agent's identity and purpose.",
        "default": "You are a helpful assistant.",
    },
}

from emad_generic.flow import build_graph  # noqa: E402, F401
