"""
Unified package registry — bootstrap kernel component.

Loads AE, TE, and eMAD packages by naming convention. No entry_points.
Package names map to Python modules: 'base-pmad-ae' -> 'base_pmad_ae'.

The config file specifies which AE and TE packages to load (with versions).
eMAD packages are installed at runtime via MCP tools and tracked in the DB.

Supports hot-reload via sys.modules eviction + importlib.invalidate_caches().
"""

import importlib
import importlib.metadata
import logging
import os
import sys
import threading
from typing import Callable

_log = logging.getLogger("emad_host.package_registry")
_lock = threading.Lock()

# Registry state
_ae_registration: dict | None = None
_te_registration: dict | None = None
_imperator_builder: Callable | None = None
_flow_builders: dict[str, Callable] = {}
_emad_build_funcs: dict[str, Callable] = {}
_package_metadata: dict[str, dict] = {}


def _to_module_name(package_name: str) -> str:
    """Convert package name to importable module name.

    'base-pmad-ae' -> 'base_pmad_ae'
    """
    return package_name.replace("-", "_")


def _evict_package_modules(package_name: str) -> None:
    """Remove a package's modules from sys.modules for hot-reload."""
    module_prefix = _to_module_name(package_name)
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


def load_ae(package_name: str) -> None:
    """Load an AE package by naming convention.

    Imports the module and calls register() to get flows and build_types.
    """
    global _ae_registration

    module_name = _to_module_name(package_name)
    _evict_package_modules(package_name)
    importlib.invalidate_caches()

    try:
        module = importlib.import_module(module_name)
        register_fn = getattr(module, "register", None)
        if register_fn is None:
            # Try .register submodule
            reg_module = importlib.import_module(f"{module_name}.register")
            register_fn = reg_module.register

        registration = register_fn()

        with _lock:
            _ae_registration = registration

            if "build_types" in registration:
                from app.flows.build_type_registry import register_build_type

                for bt_name, (asm_builder, ret_builder) in registration[
                    "build_types"
                ].items():
                    register_build_type(bt_name, asm_builder, ret_builder)

            if "flows" in registration:
                _flow_builders.update(registration["flows"])

            _package_metadata[package_name] = {
                "version": _get_package_version(package_name),
                "type": "ae",
            }

        _log.info("Loaded AE package: %s", package_name)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        _log.error("Failed to load AE package '%s': %s", package_name, exc)
        raise


def load_te(package_name: str) -> None:
    """Load a TE package by naming convention.

    Imports the module, calls register(), and injects KernelTEContext.
    """
    global _imperator_builder

    module_name = _to_module_name(package_name)
    _evict_package_modules(package_name)
    importlib.invalidate_caches()

    try:
        module = importlib.import_module(module_name)
        register_fn = getattr(module, "register", None)
        if register_fn is None:
            reg_module = importlib.import_module(f"{module_name}.register")
            register_fn = reg_module.register

        registration = register_fn()

        # Inject KernelTEContext before flow compilation
        init_fn = registration.get("initialize")
        if init_fn is not None:
            # Import KernelTEContext from the TE package itself
            kernel_ctx_module = importlib.import_module(f"{module_name}._kernel_ctx")
            ctx_class = getattr(kernel_ctx_module, "KernelTEContext")
            init_fn(ctx_class())
            _log.info("Injected KernelTEContext into TE package: %s", package_name)

        with _lock:
            if "imperator_builder" in registration:
                _imperator_builder = registration["imperator_builder"]

            if "flows" in registration:
                _flow_builders.update(registration["flows"])

            _package_metadata[package_name] = {
                "version": _get_package_version(package_name),
                "type": "te",
                "identity": registration.get("identity", ""),
                "purpose": registration.get("purpose", ""),
                "tools_required": registration.get("tools_required", []),
            }

        _log.info("Loaded TE package: %s", package_name)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        _log.error("Failed to load TE package '%s': %s", package_name, exc)
        raise


def load_emad(package_name: str) -> None:
    """Load an eMAD package by naming convention.

    Imports the module and stores its build_graph callable.
    """
    module_name = _to_module_name(package_name)
    _evict_package_modules(package_name)
    importlib.invalidate_caches()

    try:
        module = importlib.import_module(module_name)
        build_func = getattr(module, "build_graph", None)
        if build_func is None:
            raise AttributeError(
                f"eMAD package '{package_name}' has no build_graph function"
            )

        with _lock:
            _emad_build_funcs[package_name] = build_func
            _package_metadata[package_name] = {
                "version": _get_package_version(package_name),
                "type": "emad",
                "package_name": getattr(module, "EMAD_PACKAGE_NAME", package_name),
                "description": getattr(module, "DESCRIPTION", ""),
                "supported_params": getattr(module, "SUPPORTED_PARAMS", {}),
            }

        _log.info("Loaded eMAD package: %s", package_name)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        _log.error("Failed to load eMAD package '%s': %s", package_name, exc)
        raise


def scan_from_config(config: dict) -> dict:
    """Load AE and TE packages specified in config.

    Config format:
        packages:
          ae: base-pmad-ae==0.1.0
          te: base-pmad-te==0.1.0

    Returns dict with 'ae' and 'te' package names.
    """
    packages_config = config.get("packages", {})
    result = {"ae": None, "te": None}

    # Load AE
    ae_spec = packages_config.get("ae", "")
    if ae_spec:
        ae_name = ae_spec.split("==")[0].strip()
        try:
            load_ae(ae_name)
            result["ae"] = ae_name
        except (ImportError, AttributeError, TypeError, ValueError):
            _log.warning(
                "AE package '%s' not available — infrastructure flows will be missing",
                ae_name,
            )

    # Load TE
    te_spec = packages_config.get("te", "")
    if te_spec:
        te_name = te_spec.split("==")[0].strip()
        try:
            load_te(te_name)
            result["te"] = te_name
        except (ImportError, AttributeError, TypeError, ValueError):
            _log.warning(
                "TE package '%s' not available — Imperator will not be available",
                te_name,
            )

    # Scan emads/ directory and pre-load eMAD packages
    emads_dir = "/emads"
    if os.path.isdir(emads_dir):
        for name in os.listdir(emads_dir):
            config_path = os.path.join(emads_dir, name, "config.json")
            if os.path.isfile(config_path):
                # Each eMAD directory is a model name; we need to know
                # which TE package it uses. For now, all use runbook-emad-te.
                try:
                    load_emad("runbook-emad-te")
                    _log.info("eMAD discovered: %s", name)
                except (ImportError, AttributeError, TypeError, ValueError):
                    _log.warning("eMAD '%s' found but runbook-emad-te not installed", name)
                break  # Only need to load the package once; configs are per-model

    return result


# ── Public accessors ────────────────────────────────────────────────


def get_imperator_builder() -> Callable | None:
    """Return the registered Imperator flow builder, or None."""
    return _imperator_builder


def get_flow_builder(name: str) -> Callable | None:
    """Return a registered AE flow builder by name, or None."""
    return _flow_builders.get(name)


def get_build_func(package_name: str) -> Callable | None:
    """Return an eMAD's build_graph callable, or None."""
    return _emad_build_funcs.get(package_name)


def get_package_metadata() -> dict[str, dict]:
    """Return metadata for all registered packages."""
    with _lock:
        return dict(_package_metadata)


def is_loaded() -> bool:
    """Return True if at least one package has been loaded."""
    with _lock:
        return _ae_registration is not None or _imperator_builder is not None
