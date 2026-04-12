"""
eMAD management flows.

Single-node programmatic StateGraphs for:
  - emad_install_package
  - emad_create
  - emad_update
  - emad_delete
  - emad_list

Adapted from kaiser-langgraph/flows/management.py for the emad-host template.
Uses app.database for DB access, app.emad_registry for package lookup,
app.config for package source settings.
"""

import asyncio
import importlib.metadata
import logging
import sys
from typing import TypedDict

import asyncpg
from langgraph.graph import END, StateGraph

from app import emad_registry
from app.database import get_pg_pool

_log = logging.getLogger("emad_host.flows.emad_management")


# ---------------------------------------------------------------------------
# emad_install_package
# ---------------------------------------------------------------------------


class InstallPackageState(TypedDict):
    package_name: str
    version: str | None
    result: dict | None


async def _install_package_node(state: InstallPackageState) -> InstallPackageState:
    """Install an eMAD package via pip and refresh the registry."""
    from app.config import load_config

    pkg = state["package_name"]
    ver = state.get("version")
    target = f"{pkg}=={ver}" if ver else pkg

    # Check if already installed at requested version
    try:
        installed = importlib.metadata.version(pkg)
        if ver is None or installed == ver:
            emad_registry.scan()
            return {
                **state,
                "result": {
                    "status": "already_installed",
                    "package_name": pkg,
                    "version": installed,
                },
            }
    except importlib.metadata.PackageNotFoundError:
        pass

    # Resolve package source from config (same pattern as install_stategraph)
    config = load_config()
    packages_config = config.get("packages", {})
    source = packages_config.get("source", "pypi")

    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--user", "--no-deps"]

    if source == "devpi":
        devpi_url = packages_config.get("devpi_url")
        if devpi_url:
            cmd.extend(["--index-url", devpi_url])
        cmd.append(target)
    elif source == "local":
        import os
        import shutil

        local_path = packages_config.get("local_path", "/app/packages")
        source_dir = f"{local_path}/{pkg}/"
        tmp_dir = f"/tmp/emad-install-{pkg}"
        if os.path.isdir(source_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.copytree(source_dir, tmp_dir)
            cmd.extend(["--force-reinstall", tmp_dir])
        else:
            return {
                **state,
                "result": {
                    "status": "error",
                    "package_name": pkg,
                    "version": "",
                    "detail": f"Source directory not found: {source_dir}",
                },
            }
    else:
        cmd.append(target)

    _log.info("install_package: running pip install %s (source=%s)", target, source)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
    except asyncio.TimeoutError:
        return {
            **state,
            "result": {
                "status": "error",
                "package_name": pkg,
                "version": "",
                "detail": "pip install timed out",
            },
        }
    except OSError as exc:
        return {
            **state,
            "result": {
                "status": "error",
                "package_name": pkg,
                "version": "",
                "detail": str(exc),
            },
        }

    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        _log.error("install_package: pip failed for %s: %s", pkg, detail)
        return {
            **state,
            "result": {
                "status": "error",
                "package_name": pkg,
                "version": "",
                "detail": detail[:500],
            },
        }

    # Clean up /tmp copy if local source was used
    if source == "local":
        import shutil

        shutil.rmtree(f"/tmp/emad-install-{pkg}", ignore_errors=True)

    # Rescan registry to pick up new package
    emad_registry.scan()

    try:
        installed_version = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        installed_version = ver or "unknown"

    # Record in postgres (best-effort)
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO emad_packages (package_name, installed_version, status)
                VALUES ($1, $2, 'active')
                ON CONFLICT (package_name)
                DO UPDATE SET installed_version = $2, installed_at = NOW(), status = 'active'
                """,
                pkg,
                installed_version,
            )
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        _log.warning("install_package: DB record failed (non-fatal): %s", exc)

    _log.info("install_package: installed %s==%s", pkg, installed_version)
    return {
        **state,
        "result": {
            "status": "installed",
            "package_name": pkg,
            "version": installed_version,
        },
    }


def build_install_emad_package_flow() -> StateGraph:
    g = StateGraph(InstallPackageState)
    g.add_node("install", _install_package_node)
    g.set_entry_point("install")
    g.add_edge("install", END)
    return g.compile()


# ---------------------------------------------------------------------------
# emad_create
# ---------------------------------------------------------------------------


class CreateEmadState(TypedDict):
    emad_name: str
    package_name: str
    description: str
    parameters: dict
    result: dict | None


async def _create_emad_node(state: CreateEmadState) -> CreateEmadState:
    """Register a new named eMAD instance backed by an installed package."""
    emad_name = state["emad_name"]
    package_name = state["package_name"]

    if emad_registry.get_build_func(package_name) is None:
        return {
            **state,
            "result": {
                "status": "error",
                "detail": f"Package '{package_name}' not installed — "
                "run emad_install_package first",
            },
        }

    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO emad_instances (emad_name, package_name, description, parameters)
                VALUES ($1, $2, $3, $4)
                """,
                emad_name,
                package_name,
                state.get("description", ""),
                state.get("parameters", {}),
            )
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        detail = str(exc)
        if "unique" in detail.lower() or "duplicate" in detail.lower():
            detail = f"emad_name '{emad_name}' already exists"
        _log.error("create_emad: %s", detail)
        return {**state, "result": {"status": "error", "detail": detail}}

    _log.info("create_emad: created %s (package=%s)", emad_name, package_name)
    return {**state, "result": {"status": "created", "emad_name": emad_name}}


def build_create_emad_flow() -> StateGraph:
    g = StateGraph(CreateEmadState)
    g.add_node("create", _create_emad_node)
    g.set_entry_point("create")
    g.add_edge("create", END)
    return g.compile()


# ---------------------------------------------------------------------------
# emad_update
# ---------------------------------------------------------------------------


class UpdateEmadState(TypedDict):
    emad_name: str
    description: str | None
    parameters: dict | None
    result: dict | None


async def _update_emad_node(state: UpdateEmadState) -> UpdateEmadState:
    """Update the description and/or parameters of an existing eMAD instance."""
    emad_name = state["emad_name"]
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT emad_name FROM emad_instances WHERE emad_name = $1",
                emad_name,
            )
            if row is None:
                return {
                    **state,
                    "result": {
                        "status": "error",
                        "detail": f"emad_name '{emad_name}' not found",
                    },
                }
            # Build partial update
            sets, args = [], [emad_name]
            if state.get("description") is not None:
                args.append(state["description"])
                sets.append(f"description = ${len(args)}")
            if state.get("parameters") is not None:
                args.append(state["parameters"])
                sets.append(f"parameters = ${len(args)}")
            sets.append("updated_at = NOW()")
            await conn.execute(
                f"UPDATE emad_instances SET {', '.join(sets)} WHERE emad_name = $1",
                *args,
            )
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        _log.error("update_emad: %s", exc)
        return {**state, "result": {"status": "error", "detail": str(exc)}}

    _log.info("update_emad: updated %s", emad_name)
    return {**state, "result": {"status": "updated", "emad_name": emad_name}}


def build_update_emad_flow() -> StateGraph:
    g = StateGraph(UpdateEmadState)
    g.add_node("update", _update_emad_node)
    g.set_entry_point("update")
    g.add_edge("update", END)
    return g.compile()


# ---------------------------------------------------------------------------
# emad_delete
# ---------------------------------------------------------------------------


class DeleteEmadState(TypedDict):
    emad_name: str
    result: dict | None


async def _delete_emad_node(state: DeleteEmadState) -> DeleteEmadState:
    """Remove a named eMAD instance. Does not uninstall the backing package."""
    emad_name = state["emad_name"]
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM emad_instances WHERE emad_name = $1", emad_name
            )
        if result == "DELETE 0":
            return {
                **state,
                "result": {
                    "status": "error",
                    "detail": f"emad_name '{emad_name}' not found",
                },
            }
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        _log.error("delete_emad: %s", exc)
        return {**state, "result": {"status": "error", "detail": str(exc)}}

    _log.info("delete_emad: deleted %s", emad_name)
    return {**state, "result": {"status": "deleted", "emad_name": emad_name}}


def build_delete_emad_flow() -> StateGraph:
    g = StateGraph(DeleteEmadState)
    g.add_node("delete", _delete_emad_node)
    g.set_entry_point("delete")
    g.add_edge("delete", END)
    return g.compile()


# ---------------------------------------------------------------------------
# emad_list
# ---------------------------------------------------------------------------


class ListEmadsState(TypedDict):
    result: dict | None


async def _list_emads_node(state: ListEmadsState) -> ListEmadsState:
    """List all registered eMAD instances with package and status info."""
    try:
        pool = get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT i.emad_name, i.package_name, i.description,
                       i.parameters, i.status, p.installed_version
                FROM emad_instances i
                LEFT JOIN emad_packages p ON i.package_name = p.package_name
                ORDER BY i.emad_name
            """)
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        _log.error("list_emads: %s", exc)
        return {**state, "result": {"emads": [], "error": str(exc)}}

    emads = []
    for row in rows:
        pkg_loaded = emad_registry.get_build_func(row["package_name"]) is not None
        emads.append(
            {
                "emad_name": row["emad_name"],
                "package_name": row["package_name"],
                "version": row["installed_version"] or "unknown",
                "description": row["description"],
                "parameters": dict(row["parameters"]) if row["parameters"] else {},
                "status": row["status"] if pkg_loaded else "error",
            }
        )

    return {**state, "result": {"emads": emads}}


def build_list_emads_flow() -> StateGraph:
    g = StateGraph(ListEmadsState)
    g.add_node("list", _list_emads_node)
    g.set_entry_point("list")
    g.add_edge("list", END)
    return g.compile()
