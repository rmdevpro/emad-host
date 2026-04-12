"""System tools — infrastructure inspection and utilities.

Always available to the Imperator.
"""

import logging
import math
import shlex

from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.system")

# Allowlisted binaries — read-only system inspection only
# Allowlisted binaries — read-only system inspection only.
# SECURITY: Do not add python, pip, cat, env, sh, bash, or any binary
# that can execute arbitrary code or read arbitrary files. The Imperator
# is LLM-driven and vulnerable to prompt injection.
_ALLOWED_BINARIES = {
    "df",
    "uptime",
    "free",
    "hostname",
    "whoami",
    "id",
    "ping",
    "curl",
    "dig",
    "nslookup",
}


def _is_command_allowed(args: list[str]) -> bool:
    """Check if a parsed command's binary is in the allowlist."""
    if not args:
        return False
    return args[0] in _ALLOWED_BINARIES


@tool
async def run_command(command: str) -> str:
    """Execute an allowlisted shell command for infrastructure inspection.

    Only read-only commands are permitted: docker ps, df, uptime, ping, etc.
    No write operations, no arbitrary execution.

    Args:
        command: Shell command to execute (must be in the allowlist).
    """
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return f"Invalid command syntax: {exc}"

    if not _is_command_allowed(args):
        allowed = "\n".join(f"  {b}" for b in sorted(_ALLOWED_BINARIES))
        return f"Command not allowed. Permitted binaries:\n{allowed}"

    try:
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode("utf-8", errors="replace")
        errors = stderr.decode("utf-8", errors="replace")
        result = output[:5000]
        if errors:
            result += f"\n--- stderr ---\n{errors[:1000]}"
        if not result.strip():
            result = "(no output)"
        return result
    except asyncio.TimeoutError:
        return "Command timed out after 30 seconds."
    except (OSError, RuntimeError) as exc:
        return f"Command error: {exc}"


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic, powers, sqrt, log, abs, round, min, max.
    Does NOT execute arbitrary Python — only math operations.

    Args:
        expression: Math expression (e.g., "1024 * 85 / 100", "sqrt(144)").
    """
    # Safe math namespace — no builtins, no imports
    safe_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
        "int": int,
        "float": float,
    }
    try:
        # Reject anything that looks like code injection
        if any(kw in expression for kw in ["import", "__", "exec", "eval", "open"]):
            return "Expression rejected — contains unsafe keywords."
        result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
        return str(result)
    except (SyntaxError, NameError, TypeError, ValueError, ZeroDivisionError) as exc:
        return f"Calculation error: {exc}"


def get_tools() -> list:
    """Return all system tools."""
    return [run_command, calculate]
