"""
Prompt template loader.

Loads externalized prompt templates from /config/prompts/.
Templates are cached with mtime check — only re-read when the file changes (M-11).
"""

import asyncio
import logging
import os
from pathlib import Path

_log = logging.getLogger("emad_host.prompt_loader")

PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/config/prompts"))

# Cache: name -> (mtime, content)
_prompt_cache: dict[str, tuple[float, str]] = {}


def _read_prompt_file(path: Path) -> str:
    """Read and strip a prompt template file from disk.

    Separated from load_prompt() so that async_load_prompt() can
    offload only this blocking portion to run_in_executor.
    """
    return path.read_text(encoding="utf-8").strip()


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without extension).

    Reads from /config/prompts/{name}.md. Caches the result and only
    re-reads the file when its mtime changes. os.stat() is near-instant
    so this avoids repeated synchronous file I/O in async paths (M-11).

    G5-06: This function performs blocking file I/O (os.stat + read_text).
    The mtime cache means the file is only re-read when it actually changes
    on disk, which is rare in production. The os.stat() fast-path check is
    near-instant for local files. Async callers (route handlers, flow nodes)
    should use async_load_prompt() instead, which offloads the file read to
    run_in_executor when a re-read is triggered.

    Raises RuntimeError if the template file cannot be found.
    """
    path = PROMPTS_DIR / f"{name}.md"
    try:
        current_mtime = os.stat(path).st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Prompt template not found: {path}. "
            "Ensure prompt files are mounted at /config/prompts/."
        ) from exc

    cached = _prompt_cache.get(name)
    if cached is not None and cached[0] == current_mtime:
        return cached[1]

    content = _read_prompt_file(path)
    _prompt_cache[name] = (current_mtime, content)
    return content


async def async_load_prompt(name: str) -> str:
    """Async wrapper for load_prompt().

    Uses the same mtime-based cache as load_prompt(). The os.stat()
    fast-path check is synchronous (near-instant for local files).
    Only when a re-read is actually needed does it offload the file
    read to run_in_executor to avoid blocking the event loop.

    Route handlers and flow nodes should prefer this over load_prompt().
    """
    path = PROMPTS_DIR / f"{name}.md"
    try:
        current_mtime = os.stat(path).st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Prompt template not found: {path}. "
            "Ensure prompt files are mounted at /config/prompts/."
        ) from exc

    cached = _prompt_cache.get(name)
    if cached is not None and cached[0] == current_mtime:
        return cached[1]

    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, _read_prompt_file, path)
    _prompt_cache[name] = (current_mtime, content)
    return content
