"""
install_stategraph — runtime StateGraph package installation (REQ-001 §10.1).

MCP tool that installs AE or TE packages at runtime without container restart.
Uses pip install --user with the configured package source (local/pypi/devpi).
After installation, rescans entry_points to discover and register new StateGraphs.
"""

import asyncio
import logging
import os
import subprocess

_log = logging.getLogger("emad_host.flows.install_stategraph")


async def install_stategraph(
    package_name: str,
    version: str | None = None,
) -> dict:
    """Install a StateGraph package and rescan entry_points.

    Runs pip install --user from the configured source, then calls
    scan() to refresh the registry. Clears compiled graph caches
    so next invocation uses the updated package.
    """
    from app.config import async_load_config

    config = await async_load_config()
    packages_config = config.get("packages", {})
    source = packages_config.get("source", "pypi")

    # Build pip install command
    pkg_spec = f"{package_name}=={version}" if version else package_name
    # --no-deps: only install the package itself, never touch transitive dependencies.
    # All dependencies are pinned in the system site-packages at image build time.
    # Installing deps here would pull latest versions from PyPI and overwrite pins.
    cmd = ["pip", "install", "--user", "--no-cache-dir", "--no-deps"]

    if source == "devpi":
        devpi_url = packages_config.get("devpi_url")
        if devpi_url:
            cmd.extend(["--index-url", devpi_url])
        cmd.append(pkg_spec)
    elif source == "local":
        # Install from the source directory on the bind mount.
        # RB-34: bind mount is :ro — copy source to /tmp before pip install
        # to avoid polluting host repo with build artifacts.
        local_path = packages_config.get("local_path", "/app/packages")
        source_dir = f"{local_path}/{package_name}/"
        tmp_dir = f"/tmp/sg-install-{package_name}"
        import shutil

        if os.path.isdir(source_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.copytree(source_dir, tmp_dir)
            cmd.extend(["--force-reinstall", tmp_dir])
        else:
            return {
                "status": "error",
                "package": pkg_spec,
                "error": f"Source directory not found: {source_dir}",
            }
    else:
        # pypi: no extra flags needed
        cmd.append(pkg_spec)

    _log.info("Installing StateGraph package: %s (source=%s)", pkg_spec, source)

    # Run pip in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_pip, cmd)

    if result["returncode"] != 0:
        _log.error("pip install failed: %s", result["stderr"])
        return {
            "status": "error",
            "package": pkg_spec,
            "error": f"Installation failed: {result['stderr'][:500]}",
        }

    _log.info("pip install succeeded for %s", pkg_spec)

    # Clean up /tmp copy if local source was used
    if source == "local":
        import shutil

        shutil.rmtree(f"/tmp/sg-install-{package_name}", ignore_errors=True)

    # Reload the package via convention-based registry
    from app.package_registry import load_ae, load_te, load_emad
    from app.config import async_load_config as _async_load_config

    _cfg = await _async_load_config()
    pkg_config = _cfg.get("packages", {})
    ae_spec = pkg_config.get("ae", "")
    te_spec = pkg_config.get("te", "")
    if ae_spec and package_name == ae_spec.split("==")[0].strip():
        load_ae(package_name)
    elif te_spec and package_name == te_spec.split("==")[0].strip():
        load_te(package_name)
    else:
        try:
            load_emad(package_name)
        except (ImportError, AttributeError):
            _log.warning("Package '%s' loaded but not recognized as AE, TE, or eMAD", package_name)

    # Clear compiled graph caches so next use picks up new code
    from app.flows.build_type_registry import clear_compiled_cache

    clear_compiled_cache()

    # Record in database (best-effort)
    await _record_package_install(package_name, version or "latest")

    return {
        "status": "installed",
        "package": pkg_spec,
        "discovered": discovered,
    }


def _run_pip(cmd: list[str]) -> dict:
    """Run pip as a subprocess with timeout."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": "pip install timed out after 120 seconds",
        }


async def _record_package_install(package_name: str, version: str) -> None:
    """Record the package installation in PostgreSQL (best-effort)."""
    try:
        from app.database import get_pg_pool

        pool = get_pg_pool()
        await pool.execute(
            """
            INSERT INTO stategraph_packages (package_name, version, installed_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (package_name) DO UPDATE
            SET version = EXCLUDED.version, installed_at = NOW()
            """,
            package_name,
            version,
        )
    except (OSError, RuntimeError) as exc:
        _log.warning("Failed to record package install in DB: %s", exc)
