"""Scripting tools — sandboxed Python execution for data processing.

Used by eMADs that need to parse HTML, transform data, etc.
Runs in a subprocess with a timeout. No network access, no file writes
outside /data/downloads/.
"""

import asyncio
import logging
import tempfile

from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.scripting")


@tool
async def run_python(script: str) -> str:
    """Execute a Python script and return its stdout output.

    The script runs in a sandboxed subprocess with a 60-second timeout.
    It can read files but should only write to /data/downloads/.
    Available modules: re, json, os, sys, pathlib, datetime, collections.

    Args:
        script: Python code to execute.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        output = stdout.decode("utf-8", errors="replace")
        errors = stderr.decode("utf-8", errors="replace")

        result = output[:10000] if output.strip() else ""
        if errors:
            result += f"\n--- stderr ---\n{errors[:2000]}"
        if not result.strip():
            result = "(no output)"
        return result
    except asyncio.TimeoutError:
        return "ERROR: Script timed out after 60 seconds"
    except (OSError, RuntimeError) as exc:
        return f"ERROR: {exc}"
    finally:
        import os
        try:
            os.unlink(script_path)
        except OSError:
            pass
