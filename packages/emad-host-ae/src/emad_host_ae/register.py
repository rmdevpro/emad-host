"""
AE — Package registration entry point.

Called by the bootstrap kernel's stategraph_registry.scan() when this
package is discovered via entry_points(group="emad_host.ae").

Returns an AERegistration dict with build type registrations and
flow builders that the kernel processes to populate its registries.
"""


def register() -> dict:
    """Register the AE's infrastructure StateGraphs.

    Returns a dict with:
    - build_types: dict of (assembly_builder, retrieval_builder) pairs
    - flows: dict of flow_name -> builder callable
    """
    from emad_host_ae.health_flow import build_health_check_flow
    from emad_host_ae.metrics_flow import build_metrics_flow
    from emad_host_ae.autoprompt_dispatcher import build_autoprompt_dispatcher_flow

    return {
        "build_types": {},
        "flows": {
            "health_check": build_health_check_flow,
            "metrics": build_metrics_flow,
            "autoprompt_dispatcher": build_autoprompt_dispatcher_flow,
        },
    }
