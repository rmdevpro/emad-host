"""
StateGraph package registry — bootstrap kernel component.

Discovers AE and TE packages via Python entry_points (REQ-001 §10.2).
Follows the Kaiser pattern: importlib.metadata for discovery,
sys.modules eviction for hot-reload without container restart.

Entry point groups:
  - emad_host.ae: AE packages (infrastructure StateGraphs)
  - emad_host.te: TE packages (cognitive StateGraphs — Imperator)
"""

import importlib
import importlib.metadata
import logging
import sys
import threading
from typing import Callable

_log = logging.getLogger("emad_host.stategraph_registry")
_lock = threading.Lock()

# Registry state
_te_packages: dict[str, dict] = {}
_ae_packages: dict[str, dict] = {}
_imperator_builder: Callable | None = None
_flow_builders: dict[str, Callable] = {}  # flow_name -> builder callable
_package_metadata: dict[str, dict] = {}


def scan() -> dict[str, list[str]]:
    """Scan entry_points for AE and TE packages.

    Evicts old modules from sys.modules before re-scanning (Kaiser pattern)
    to support hot-reload without container restart.

    Returns dict with 'ae' and 'te' lists of discovered package names.
    """
    global _imperator_builder

    # Step 1: Evict previously-loaded package modules
    with _lock:
        for pkg_name in list(_te_packages.keys()) + list(_ae_packages.keys()):
            _evict_package_modules(pkg_name)
        _te_packages.clear()
        _ae_packages.clear()
        _flow_builders.clear()
        _package_metadata.clear()
        _imperator_builder = None

    importlib.invalidate_caches()

    discovered: dict[str, list[str]] = {"ae": [], "te": []}

    # Step 2: Discover AE packages
    for ep in importlib.metadata.entry_points(group="emad_host.ae"):
        try:
            register_fn = ep.load()
            registration = register_fn()
            with _lock:
                _ae_packages[ep.name] = registration

                # Register build types
                if "build_types" in registration:
                    from app.flows.build_type_registry import register_build_type

                    for bt_name, (asm_builder, ret_builder) in registration[
                        "build_types"
                    ].items():
                        register_build_type(bt_name, asm_builder, ret_builder)

                # Register flow builders
                if "flows" in registration:
                    _flow_builders.update(registration["flows"])

                _package_metadata[ep.name] = {
                    "version": _get_package_version(ep.name),
                    "type": "ae",
                }

            discovered["ae"].append(ep.name)
            _log.info("Registered AE package: %s", ep.name)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _log.error("Failed to load AE entry point '%s': %s", ep.name, exc)

    # Step 3: Discover TE packages
    for ep in importlib.metadata.entry_points(group="emad_host.te"):
        try:
            register_fn = ep.load()
            registration = register_fn()
            # TE/AE decoupling: inject KernelTEContext before flow compilation
            init_fn = registration.get("initialize")
            if init_fn is not None:
                from emad_host_te._kernel_ctx import KernelTEContext

                init_fn(KernelTEContext())
                _log.info("Injected KernelTEContext into TE package: %s", ep.name)

            with _lock:
                _te_packages[ep.name] = registration

                if "imperator_builder" in registration:
                    _imperator_builder = registration["imperator_builder"]

                # CEA: Register TE flows (e.g., ceac_enrichment) alongside AE flows.
                # TE packages can now export a "flows" dict just like AE packages.
                if "flows" in registration:
                    _flow_builders.update(registration["flows"])

                _package_metadata[ep.name] = {
                    "version": _get_package_version(ep.name),
                    "type": "te",
                    "identity": registration.get("identity", ""),
                    "purpose": registration.get("purpose", ""),
                    "tools_required": registration.get("tools_required", []),
                }

            discovered["te"].append(ep.name)
            _log.info("Registered TE package: %s", ep.name)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _log.error("Failed to load TE entry point '%s': %s", ep.name, exc)

    return discovered


def get_imperator_builder() -> Callable | None:
    """Return the registered Imperator flow builder, or None."""
    return _imperator_builder


def get_flow_builder(name: str) -> Callable | None:
    """Return a registered AE flow builder by name, or None."""
    return _flow_builders.get(name)


def get_package_metadata() -> dict[str, dict]:
    """Return metadata for all registered packages."""
    with _lock:
        return dict(_package_metadata)


def is_loaded() -> bool:
    """Return True if at least one package has been loaded."""
    with _lock:
        return bool(_te_packages) or bool(_ae_packages)


def _evict_package_modules(package_name: str) -> None:
    """Remove a package's modules from sys.modules to allow reimport.

    Converts package name (hyphens) to module prefix (underscores)
    for sys.modules lookup.
    """
    module_prefix = package_name.replace("-", "_")
    to_remove = [
        key
        for key in sys.modules
        if key == module_prefix or key.startswith(module_prefix + ".")
    ]
    for key in to_remove:
        del sys.modules[key]
    if to_remove:
        _log.info("Evicted %d modules for package '%s'", len(to_remove), package_name)


def _get_package_version(package_name: str) -> str:
    """Get the installed version of a package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
