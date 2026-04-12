"""Filesystem tools — read-only access + downloads + system prompt.

Read-only: sandboxed to allowed roots (/app, /config, /data).
Write: only to /data/downloads/.
System prompt: read and update the Imperator's own prompt file.
"""

import asyncio
import logging
import os
import re
from pathlib import Path

from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.filesystem")

# Allowed read roots — the Imperator can inspect its own code and config
_READ_ROOTS = ["/app", "/config", "/data"]
# Write is restricted to downloads directory only
_DOWNLOADS_DIR = "/data/downloads"
# System prompt directory
_PROMPTS_DIR = "/config/prompts"


def _is_safe_read_path(path: str) -> bool:
    """Check if a path is within allowed read roots after symlink resolution."""
    try:
        resolved = str(Path(path).resolve())
        return any(resolved.startswith(root) for root in _READ_ROOTS)
    except (OSError, ValueError):
        return False


def _is_safe_write_path(path: str) -> bool:
    """Check if a path is within the downloads directory."""
    try:
        resolved = str(Path(path).resolve())
        return resolved.startswith(_DOWNLOADS_DIR)
    except (OSError, ValueError):
        return False


def _sync_file_read(path: str, max_chars: int) -> str:
    """Synchronous file read helper."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
        return content
    except FileNotFoundError:
        return f"File not found: {path}"
    except (OSError, PermissionError) as exc:
        return f"Error reading {path}: {exc}"


@tool
async def file_read(path: str, max_chars: int = 50000) -> str:
    """Read a file from the filesystem.

    Sandboxed to /app, /config, and /data directories.
    Use this to inspect source code, configuration, prompts, or data files.

    Args:
        path: Absolute path to the file.
        max_chars: Maximum characters to return (default 50000).
    """
    if not _is_safe_read_path(path):
        return f"Access denied: {path} is outside allowed directories."
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_file_read, path, max_chars)


def _sync_file_list(path: str) -> str:
    """Synchronous directory listing helper."""
    try:
        entries = sorted(os.listdir(path))
        lines = [f"Contents of {path} ({len(entries)} entries):"]
        for entry in entries:
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                lines.append(f"  {entry}/")
            else:
                size = os.path.getsize(full)
                lines.append(f"  {entry} ({size} bytes)")
        return "\n".join(lines)
    except FileNotFoundError:
        return f"Directory not found: {path}"
    except (OSError, PermissionError) as exc:
        return f"Error listing {path}: {exc}"


@tool
async def file_list(path: str = "/app") -> str:
    """List directory contents.

    Sandboxed to /app, /config, and /data directories.

    Args:
        path: Directory path to list (default /app).
    """
    if not _is_safe_read_path(path):
        return f"Access denied: {path} is outside allowed directories."
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_file_list, path)


def _sync_file_search(path: str, compiled: re.Pattern, max_results: int) -> str:
    """Synchronous file search helper."""
    results = []
    try:
        for root, _, files in os.walk(path):
            if not _is_safe_read_path(root):
                continue
            for fn in sorted(files):
                if len(results) >= max_results:
                    break
                fpath = os.path.join(root, fn)
                try:
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if compiled.search(line):
                                rel = os.path.relpath(fpath, path)
                                results.append(f"{rel}:{i}: {line.rstrip()[:120]}")
                                if len(results) >= max_results:
                                    break
                except (OSError, PermissionError):
                    continue

        if not results:
            return f"No matches for '{compiled.pattern}' in {path}"
        return "\n".join(results)
    except (OSError, PermissionError) as exc:
        return f"Search error: {exc}"


@tool
async def file_search(path: str, pattern: str, max_results: int = 20) -> str:
    """Search file contents for a pattern (like grep).

    Searches recursively through text files in the given directory.
    Sandboxed to /app, /config, and /data directories.

    Args:
        path: Directory to search in.
        pattern: Regex pattern to search for.
        max_results: Maximum matching lines to return (default 20).
    """
    if not _is_safe_read_path(path):
        return f"Access denied: {path} is outside allowed directories."
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regex: {exc}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_file_search, path, compiled, max_results)


def _sync_file_write(path: str, content: str) -> str:
    """Synchronous file write helper."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except (OSError, PermissionError) as exc:
        return f"Error writing {path}: {exc}"


@tool
async def file_write(path: str, content: str) -> str:
    """Write content to a file in the downloads directory.

    Only writes to /data/downloads/ — cannot write anywhere else.
    Creates the file and parent directories if they don't exist.

    Args:
        path: File path within /data/downloads/.
        content: Content to write.
    """
    # Ensure path is within downloads
    if not path.startswith(_DOWNLOADS_DIR):
        path = os.path.join(_DOWNLOADS_DIR, path.lstrip("/"))

    if not _is_safe_write_path(path):
        return f"Access denied: can only write to {_DOWNLOADS_DIR}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_file_write, path, content)


def _sync_read_system_prompt() -> str:
    """Synchronous system prompt read helper."""
    from emad_host_te._ctx import get_ctx

    config = get_ctx().load_merged_config()
    prompt_name = config.get("imperator", {}).get("system_prompt", "imperator_identity")
    prompt_path = os.path.join(_PROMPTS_DIR, f"{prompt_name}.md")
    try:
        with open(prompt_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"System prompt not found: {prompt_path}"
    except (OSError, PermissionError) as exc:
        return f"Error reading system prompt: {exc}"


@tool
async def read_system_prompt() -> str:
    """Read the Imperator's current system prompt.

    Returns the full content of the system prompt file.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_read_system_prompt)


def _sync_update_system_prompt(content: str) -> str:
    """Synchronous system prompt update helper."""
    from emad_host_te._ctx import get_ctx

    config = get_ctx().load_merged_config()
    prompt_name = config.get("imperator", {}).get("system_prompt", "imperator_identity")
    prompt_path = os.path.join(_PROMPTS_DIR, f"{prompt_name}.md")
    try:
        # Back up current prompt
        backup_path = f"{prompt_path}.backup"
        if os.path.exists(prompt_path):
            with open(prompt_path, encoding="utf-8") as f:
                old = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(old)

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"System prompt updated ({len(content)} chars). Backup saved to {backup_path}."
    except (OSError, PermissionError) as exc:
        return f"Error updating system prompt: {exc}"


@tool
async def update_system_prompt(content: str) -> str:
    """Update the Imperator's system prompt.

    Writes the new content to the system prompt file. The change takes
    effect on the next chat invocation (no restart needed).

    This is the only TE configuration the Imperator can modify.
    All other TE config (model, build_type, admin_tools) is architect-only.

    Args:
        content: The new system prompt content.
    """
    if not content or len(content) < 20:
        return "System prompt too short — must be at least 20 characters."

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_update_system_prompt, content)


def get_tools() -> list:
    """Return all filesystem tools."""
    return [
        file_read,
        file_list,
        file_search,
        file_write,
        read_system_prompt,
        update_system_prompt,
    ]
