"""eMAD management tool — add, update, remove, rename, list eMADs.

AE tool for the host Imperator to manage eMAD lifecycle.
Handles pip install, routing table entries, and config directories.
"""

import asyncio
import importlib
import importlib.metadata
import json
import logging
import os
import shutil
import sys
from typing import Optional

import asyncpg
from app.database import get_pg_pool
from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.emad_management")


async def _pip_install(package_spec: str, extra_flags: list[str] | None = None) -> dict:
    """Run pip install --user --no-cache-dir for a package."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--user", "--no-cache-dir",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(package_spec)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        if proc.returncode != 0:
            return {"ok": False, "error": stderr.decode(errors="replace").strip()[:500]}
        return {"ok": True}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "pip install timed out"}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


async def _pip_uninstall(package_name: str) -> dict:
    """Run pip uninstall -y for a package."""
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package_name]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        if proc.returncode != 0:
            return {"ok": False, "error": stderr.decode(errors="replace").strip()[:500]}
        return {"ok": True}
    except (asyncio.TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def _evict_modules(package_name: str) -> None:
    """Evict a package's modules from sys.modules for hot-reload."""
    module_prefix = package_name.replace("-", "_")
    to_remove = [
        k for k in sys.modules
        if k == module_prefix or k.startswith(module_prefix + ".")
    ]
    for key in to_remove:
        del sys.modules[key]
    importlib.invalidate_caches()


def _load_package_metadata(package_name: str) -> dict:
    """Try to import the package and read its metadata."""
    module_name = package_name.replace("-", "_")
    try:
        module = importlib.import_module(module_name)
        return {
            "description": getattr(module, "DESCRIPTION", ""),
            "emad_name": getattr(module, "EMAD_NAME", package_name),
            "has_build_graph": hasattr(module, "build_graph"),
        }
    except ImportError:
        return {}


def _setup_emad_directory(name: str, package_name: str) -> None:
    """Create /emads/{name}/ directory with config from package data if available."""
    emad_dir = f"/emads/{name}"
    os.makedirs(emad_dir, exist_ok=True)

    # Try to find package data (config.json, runbook.md, etc.)
    module_name = package_name.replace("-", "_")
    try:
        module = importlib.import_module(module_name)
        pkg_dir = os.path.dirname(module.__file__)

        # Copy config files if they exist in the package
        for fname in ["config.json", "runbook.md"]:
            src = os.path.join(pkg_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(emad_dir, fname))

        # Copy platform-knowledge directory if it exists
        pk_src = os.path.join(pkg_dir, "platform-knowledge")
        pk_dst = os.path.join(emad_dir, "platform-knowledge")
        if os.path.isdir(pk_src):
            if os.path.isdir(pk_dst):
                shutil.rmtree(pk_dst)
            shutil.copytree(pk_src, pk_dst)

    except (ImportError, OSError) as exc:
        _log.warning("Could not copy package data for %s: %s", package_name, exc)


@tool
async def manage_emad(
    operation: str,
    name: str = "",
    package: str = "",
    version: str = "",
    new_name: str = "",
) -> str:
    """Manage eMAD lifecycle: add, update, remove, rename, or list.

    Operations:
    - add: Install an eMAD package and register it. Requires name and package.
    - update: Upgrade an installed eMAD to a new version. Requires name.
    - remove: Uninstall an eMAD and remove its routing. Requires name.
    - rename: Change the model name for an eMAD. Requires name and new_name.
    - list: Show all installed eMADs with their packages and status.

    Args:
        operation: One of: add, update, remove, rename, list.
        name: The model name for routing (e.g., "brin", "weather-reporter").
        package: PyPI package name (e.g., "google-workspace-emad"). Required for add.
        version: Specific version to install (e.g., "0.1.2"). Empty for latest.
        new_name: New model name, used with rename operation.
    """
    if operation == "list":
        return await _op_list()
    elif operation == "add":
        if not name or not package:
            return "ERROR: 'add' requires both name and package."
        return await _op_add(name, package, version)
    elif operation == "update":
        if not name:
            return "ERROR: 'update' requires name."
        return await _op_update(name, version)
    elif operation == "remove":
        if not name:
            return "ERROR: 'remove' requires name."
        return await _op_remove(name)
    elif operation == "rename":
        if not name or not new_name:
            return "ERROR: 'rename' requires name and new_name."
        return await _op_rename(name, new_name)
    else:
        return f"ERROR: Unknown operation '{operation}'. Use: add, update, remove, rename, list."


async def _op_list() -> str:
    """List all installed eMADs."""
    try:
        pool = get_pg_pool()
        rows = await pool.fetch("""
            SELECT i.emad_name, i.package_name, i.description, i.status,
                   p.installed_version
            FROM emad_instances i
            LEFT JOIN emad_packages p ON i.package_name = p.package_name
            ORDER BY i.emad_name
        """)
    except (asyncpg.PostgresError, RuntimeError) as exc:
        return f"ERROR: {exc}"

    if not rows:
        return "No eMADs installed."

    lines = ["Installed eMADs:"]
    for row in rows:
        ver = row["installed_version"] or "?"
        lines.append(
            f"  {row['emad_name']} → {row['package_name']}=={ver} "
            f"({row['status']}) — {row['description']}"
        )
    return "\n".join(lines)


async def _op_add(name: str, package: str, version: str) -> str:
    """Install a package and register the eMAD."""
    pkg_spec = f"{package}=={version}" if version else package

    # Check if name already exists
    try:
        pool = get_pg_pool()
        existing = await pool.fetchrow(
            "SELECT emad_name FROM emad_instances WHERE emad_name = $1", name
        )
        if existing:
            return f"ERROR: eMAD '{name}' already exists. Use 'update' or 'remove' first."
    except (asyncpg.PostgresError, RuntimeError) as exc:
        return f"ERROR checking existing: {exc}"

    # pip install
    result = await _pip_install(pkg_spec)
    if not result["ok"]:
        return f"ERROR installing {pkg_spec}: {result['error']}"

    _evict_modules(package)

    # Get installed version
    try:
        installed_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed_version = version or "unknown"

    # Load package and register in registry
    from app.package_registry import load_emad
    try:
        load_emad(package)
    except (ImportError, AttributeError) as exc:
        return f"ERROR: Package installed but failed to load: {exc}"

    # Record in DB
    try:
        pool = get_pg_pool()
        await pool.execute(
            """
            INSERT INTO emad_packages (package_name, installed_version, status)
            VALUES ($1, $2, 'active')
            ON CONFLICT (package_name)
            DO UPDATE SET installed_version = $2, installed_at = NOW(), status = 'active'
            """,
            package, installed_version,
        )

        meta = _load_package_metadata(package)
        await pool.execute(
            """
            INSERT INTO emad_instances (emad_name, package_name, description, parameters)
            VALUES ($1, $2, $3, $4::jsonb)
            """,
            name, package,
            meta.get("description", ""),
            json.dumps({}),
        )
    except (asyncpg.PostgresError, RuntimeError) as exc:
        return f"ERROR recording in DB: {exc}"

    # Set up config directory
    _setup_emad_directory(name, package)

    # Invalidate router cache
    from app.routes.chat import invalidate_graph_cache
    invalidate_graph_cache()

    return f"Installed {package}=={installed_version} as '{name}'."


async def _op_update(name: str, version: str) -> str:
    """Upgrade an installed eMAD."""
    try:
        pool = get_pg_pool()
        row = await pool.fetchrow(
            "SELECT package_name FROM emad_instances WHERE emad_name = $1", name
        )
        if not row:
            return f"ERROR: eMAD '{name}' not found."
        package = row["package_name"]
    except (asyncpg.PostgresError, RuntimeError) as exc:
        return f"ERROR: {exc}"

    if version:
        result = await _pip_install(f"{package}=={version}")
    else:
        result = await _pip_install(package, extra_flags=["--upgrade"])
    if not result["ok"]:
        return f"ERROR upgrading {package}: {result['error']}"

    _evict_modules(package)

    try:
        installed_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed_version = version or "unknown"

    # Reload in registry
    from app.package_registry import load_emad
    try:
        load_emad(package)
    except (ImportError, AttributeError) as exc:
        return f"WARNING: Upgraded but reload failed: {exc}"

    # Update DB
    try:
        pool = get_pg_pool()
        await pool.execute(
            """
            UPDATE emad_packages SET installed_version = $2, installed_at = NOW()
            WHERE package_name = $1
            """,
            package, installed_version,
        )
    except (asyncpg.PostgresError, RuntimeError) as exc:
        _log.warning("DB update failed: %s", exc)

    # Refresh config directory
    _setup_emad_directory(name, package)

    from app.routes.chat import invalidate_graph_cache
    invalidate_graph_cache()

    return f"Updated '{name}' to {package}=={installed_version}."


async def _op_remove(name: str) -> str:
    """Remove an eMAD: delete routing, remove config dir."""
    try:
        pool = get_pg_pool()
        row = await pool.fetchrow(
            "SELECT package_name FROM emad_instances WHERE emad_name = $1", name
        )
        if not row:
            return f"ERROR: eMAD '{name}' not found."
        package = row["package_name"]

        await pool.execute(
            "DELETE FROM emad_instances WHERE emad_name = $1", name
        )

        # Check if other instances use the same package
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM emad_instances WHERE package_name = $1", package
        )
        if count == 0:
            # No other instances — uninstall the package
            await _pip_uninstall(package)
            _evict_modules(package)
            await pool.execute(
                "DELETE FROM emad_packages WHERE package_name = $1", package
            )
    except (asyncpg.PostgresError, RuntimeError) as exc:
        return f"ERROR: {exc}"

    # Remove config directory
    emad_dir = f"/emads/{name}"
    if os.path.isdir(emad_dir):
        shutil.rmtree(emad_dir, ignore_errors=True)

    from app.routes.chat import invalidate_graph_cache
    invalidate_graph_cache()

    return f"Removed eMAD '{name}'."


async def _op_rename(name: str, new_name: str) -> str:
    """Rename an eMAD's model name in the routing table."""
    try:
        pool = get_pg_pool()
        result = await pool.execute(
            "UPDATE emad_instances SET emad_name = $2, updated_at = NOW() WHERE emad_name = $1",
            name, new_name,
        )
        if result == "UPDATE 0":
            return f"ERROR: eMAD '{name}' not found."
    except (asyncpg.PostgresError, RuntimeError) as exc:
        return f"ERROR: {exc}"

    # Rename config directory
    old_dir = f"/emads/{name}"
    new_dir = f"/emads/{new_name}"
    if os.path.isdir(old_dir):
        os.rename(old_dir, new_dir)

    from app.routes.chat import invalidate_graph_cache
    invalidate_graph_cache()

    return f"Renamed '{name}' to '{new_name}'."


def get_tools() -> list:
    """Return eMAD management tools."""
    return [manage_emad]
