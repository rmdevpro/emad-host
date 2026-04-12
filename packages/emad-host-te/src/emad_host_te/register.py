"""
TE — Package registration entry point.

Called by the bootstrap kernel's stategraph_registry.scan() when this
package is discovered via entry_points(group="emad_host.te").

Returns a TERegistration dict with the Imperator flow builder and
identity/purpose declarations.
"""


def register() -> dict:
    """Register the TE's cognitive StateGraphs.

    Returns a dict with:
    - identity: What the Imperator is
    - purpose: What the Imperator is for
    - imperator_builder: callable that builds the compiled Imperator StateGraph
    - initialize: callable(ctx) — AE bootstrap calls this before flow compilation
    """
    from emad_host_te._ctx import initialize
    from emad_host_te.imperator_flow import build_imperator_flow

    return {
        "identity": "Imperator",
        "purpose": "pMAD management and conversational interface",
        "imperator_builder": build_imperator_flow,
        "initialize": initialize,
        "tools_required": [],
        "flows": {},
    }
