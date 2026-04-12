"""
eMAD package registry — discovers and manages eMAD packages.

Scans emad_host.emads entry_points to build an in-memory map of
  package_name -> build_graph callable

Refreshed at startup and after each emad_install_package call.
Call importlib.invalidate_caches() before scan() to pick up newly
installed packages.

Follows the Kaiser pattern (REQ-002 §10.2): importlib.metadata for
discovery, sys.modules eviction for hot-reload without container restart.
"""

import importlib
import importlib.metadata
import logging
import sys
from typing import Callable

from langgraph.graph import StateGraph

_log = logging.getLogger("context_broker.emad_registry")

# package_name -> build_graph callable
_registry: dict[str, Callable[[dict], StateGraph]] = {}


def scan() -> list[str]:
    """Rescan emad_host.emads entry_points and rebuild the registry.

    Evicts previously loaded eMAD modules from sys.modules before
    rescanning so that ep.load() reimports from freshly installed
    package code (upgrade without restart).  importlib.invalidate_caches()
    ensures newly installed .dist-info directories are visible to the finder.

    Returns list of discovered package names.
    """
    # Evict old eMAD modules so ep.load() picks up newly installed code.
    for build_func in _registry.values():
        module_root = build_func.__module__.split(".")[0]
        evict = [
            k
            for k in list(sys.modules)
            if k == module_root or k.startswith(module_root + ".")
        ]
        for key in evict:
            del sys.modules[key]

    importlib.invalidate_caches()
    found: dict[str, Callable[[dict], StateGraph]] = {}

    eps = importlib.metadata.entry_points(group="emad_host.emads")
    for ep in eps:
        try:
            build_func = ep.load()
            found[ep.name] = build_func
            _log.info("Registered eMAD package: %s", ep.name)
        except (ImportError, AttributeError, TypeError) as exc:
            _log.error("Failed to load eMAD entry_point %s: %s", ep.name, exc)

    _registry.clear()
    _registry.update(found)
    _log.info("eMAD registry scan complete: %d package(s)", len(_registry))
    return list(_registry.keys())


def get_build_func(package_name: str) -> Callable[[dict], StateGraph] | None:
    """Return the build_graph function for the given package, or None.

    Triggers a scan on first call if the registry is empty (lazy init).
    """
    if not _registry:
        scan()
    return _registry.get(package_name)


def list_packages() -> list[str]:
    """Return all registered package names."""
    return list(_registry.keys())


def get_package_metadata(package_name: str) -> dict:
    """Read EMAD_PACKAGE_NAME, DESCRIPTION, SUPPORTED_PARAMS from the
    installed package.

    Returns {} if the package is not found or does not expose these attributes.
    """
    build_func = _registry.get(package_name)
    if build_func is None:
        return {}
    module = importlib.import_module(build_func.__module__.split(".")[0])
    return {
        "package_name": getattr(module, "EMAD_PACKAGE_NAME", package_name),
        "description": getattr(module, "DESCRIPTION", ""),
        "supported_params": getattr(module, "SUPPORTED_PARAMS", {}),
    }
