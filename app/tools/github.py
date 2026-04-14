"""GitHub CLI tools — scoped issue operations for job search pipeline.

These tools wrap specific `gh` commands needed by the linkedin-jobsearch
and linkedin-jobapply eMADs. They do NOT provide general shell access.
"""

import asyncio
import logging

from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.github")


async def _run_gh(args: list[str], timeout: float = 30.0) -> str:
    """Run a gh CLI command and return output."""
    cmd = ["gh"] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")
        errors = stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            return f"ERROR (exit {proc.returncode}): {errors[:500]}"
        return output[:5000] if output.strip() else "(no output)"
    except asyncio.TimeoutError:
        return "ERROR: gh command timed out"
    except (OSError, FileNotFoundError) as exc:
        return f"ERROR: {exc}"


@tool
async def gh_issue_list(
    repo: str,
    labels: str = "",
    limit: int = 20,
    state: str = "open",
) -> str:
    """List GitHub issues in a repository, optionally filtered by labels.

    Args:
        repo: Repository in owner/repo format (e.g., rmdevpro/JobSearch).
        labels: Comma-separated label filter (e.g., "status:new,easy-apply").
        limit: Maximum issues to return.
        state: Issue state filter: open, closed, or all.
    """
    args = ["issue", "list", "--repo", repo, "--limit", str(limit), "--state", state]
    if labels:
        args.extend(["--label", labels])
    args.extend(["--json", "number,title,body,labels,createdAt"])
    return await _run_gh(args)


@tool
async def gh_issue_create(
    repo: str,
    title: str,
    body: str,
    labels: str = "",
) -> str:
    """Create a GitHub issue.

    Args:
        repo: Repository in owner/repo format.
        title: Issue title.
        body: Issue body (markdown).
        labels: Comma-separated labels to apply.
    """
    args = ["issue", "create", "--repo", repo, "--title", title, "--body", body]
    if labels:
        args.extend(["--label", labels])
    return await _run_gh(args)


@tool
async def gh_issue_edit(
    repo: str,
    issue_number: int,
    add_labels: str = "",
    remove_labels: str = "",
) -> str:
    """Edit a GitHub issue's labels.

    Args:
        repo: Repository in owner/repo format.
        issue_number: Issue number to edit.
        add_labels: Comma-separated labels to add.
        remove_labels: Comma-separated labels to remove.
    """
    args = ["issue", "edit", str(issue_number), "--repo", repo]
    if add_labels:
        args.extend(["--add-label", add_labels])
    if remove_labels:
        args.extend(["--remove-label", remove_labels])
    return await _run_gh(args)


@tool
async def gh_issue_comment(
    repo: str,
    issue_number: int,
    body: str,
) -> str:
    """Add a comment to a GitHub issue.

    Args:
        repo: Repository in owner/repo format.
        issue_number: Issue number.
        body: Comment body (markdown).
    """
    args = ["issue", "comment", str(issue_number), "--repo", repo, "--body", body]
    return await _run_gh(args)
