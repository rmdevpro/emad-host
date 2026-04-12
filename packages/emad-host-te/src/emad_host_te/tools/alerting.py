"""Alerting tools — manage alert instructions for the alerter sidecar.

The alerter uses an instruction table to decide how to format and route
alerts. These tools let the Imperator add, list, update, and delete
instructions at runtime.
"""

import json
import logging

import asyncpg

from langchain_core.tools import tool

from emad_host_te._ctx import get_ctx

_log = logging.getLogger("emad_host.tools.alerting")


@tool
async def add_alert_instruction(
    description: str, instruction: str, channels: str
) -> str:
    """Add an alert instruction that tells the alerter how to handle a type of event.

    When an alert arrives, the alerter searches instructions by semantic similarity
    to find the best match. The matched instruction becomes the LLM system prompt
    for formatting, and its channels define where the message is sent.

    Args:
        description: Short description of what events this handles
            (e.g., "health degradation alerts", "extraction pipeline errors").
            This is what the alerter searches against.
        instruction: Full instruction text used as the LLM system prompt.
            Tell the LLM how to format this type of alert.
            Example: "You receive health alerts. Write a brief push notification
            including the affected service and error details."
        channels: JSON array of channel configurations. Each channel has a "type"
            and type-specific fields. Types: slack, discord, ntfy, smtp, webhook, log.
            Example: [{"type": "ntfy", "url": "https://ntfy.sh/my-alerts", "priority": "high"}]
    """
    try:
        parsed_channels = json.loads(channels)
        if not isinstance(parsed_channels, list):
            return "Error: channels must be a JSON array."
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON in channels: {exc}"

    pool = get_ctx().get_pool()

    # Embed the description for semantic search
    embedding = await _embed_description(description)
    vec_str = None
    if embedding:
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    try:
        row = await pool.fetchrow(
            """
            INSERT INTO alert_instructions (description, instruction, channels, embedding)
            VALUES ($1, $2, $3::jsonb, $4::vector)
            RETURNING id
            """,
            description,
            instruction,
            json.dumps(parsed_channels),
            vec_str,
        )
        return (
            f"Alert instruction added (id={row['id']}). "
            f"Description: {description}. "
            f"Channels: {[c.get('type') for c in parsed_channels]}."
        )
    except (asyncpg.PostgresError, json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
        return f"Failed to add instruction: {exc}"


@tool
async def list_alert_instructions() -> str:
    """List all alert instructions configured for the alerter.

    Shows each instruction's ID, description, channels, and creation date.
    """
    pool = get_ctx().get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT id, description, channels, created_at, updated_at
            FROM alert_instructions
            ORDER BY created_at DESC
            """
        )
        if not rows:
            return "No alert instructions configured. Use add_alert_instruction to create one."

        lines = [f"{len(rows)} alert instruction(s):\n"]
        for row in rows:
            channels = row["channels"]
            if isinstance(channels, str):
                channels = json.loads(channels)
            ch_types = [c.get("type", "?") for c in channels]
            lines.append(
                f"  [{row['id']}] {row['description']}\n"
                f"       Channels: {ch_types}\n"
                f"       Created: {row['created_at'].isoformat()}"
            )
        return "\n".join(lines)
    except (asyncpg.PostgresError, OSError) as exc:
        return f"Failed to list instructions: {exc}"


@tool
async def update_alert_instruction(
    instruction_id: int,
    description: str = "",
    instruction: str = "",
    channels: str = "",
) -> str:
    """Update an existing alert instruction.

    Only the provided fields are updated — omit fields to keep current values.

    Args:
        instruction_id: ID of the instruction to update.
        description: New description (empty to keep current).
        instruction: New instruction text (empty to keep current).
        channels: New channels JSON array (empty to keep current).
    """
    pool = get_ctx().get_pool()

    # Verify it exists
    existing = await pool.fetchrow(
        "SELECT id FROM alert_instructions WHERE id = $1", instruction_id
    )
    if not existing:
        return f"No instruction with id={instruction_id}."

    updates = []
    params = []
    idx = 1

    if description:
        updates.append(f"description = ${idx}")
        params.append(description)
        idx += 1
        # Re-embed
        embedding = await _embed_description(description)
        if embedding:
            vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
            updates.append(f"embedding = ${idx}::vector")
            params.append(vec_str)
            idx += 1

    if instruction:
        updates.append(f"instruction = ${idx}")
        params.append(instruction)
        idx += 1

    if channels:
        try:
            parsed = json.loads(channels)
            if not isinstance(parsed, list):
                return "Error: channels must be a JSON array."
            updates.append(f"channels = ${idx}::jsonb")
            params.append(json.dumps(parsed))
            idx += 1
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON in channels: {exc}"

    if not updates:
        return "Nothing to update — provide at least one field."

    updates.append(f"updated_at = NOW()")
    params.append(instruction_id)

    try:
        await pool.execute(
            f"UPDATE alert_instructions SET {', '.join(updates)} WHERE id = ${idx}",
            *params,
        )
        return f"Instruction {instruction_id} updated."
    except (asyncpg.PostgresError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return f"Failed to update instruction: {exc}"


@tool
async def delete_alert_instruction(instruction_id: int) -> str:
    """Delete an alert instruction by ID.

    Args:
        instruction_id: ID of the instruction to delete.
    """
    pool = get_ctx().get_pool()
    try:
        result = await pool.execute(
            "DELETE FROM alert_instructions WHERE id = $1", instruction_id
        )
        if result == "DELETE 1":
            return f"Instruction {instruction_id} deleted."
        return f"No instruction with id={instruction_id}."
    except (asyncpg.PostgresError, OSError) as exc:
        return f"Failed to delete instruction: {exc}"


async def _embed_description(text: str) -> list[float] | None:
    """Embed the description using the MAD's configured embedding model."""
    try:
        ctx = get_ctx()
        config = await ctx.async_load_config()
        model = ctx.get_embeddings_model(config)
        vectors = await model.aembed_documents([text])
        return vectors[0]
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        _log.warning("Failed to embed description: %s", exc)
        return None


def get_tools(te_config: dict | None = None) -> list:
    """Return alerting tools."""
    return [
        add_alert_instruction,
        list_alert_instructions,
        update_alert_instruction,
        delete_alert_instruction,
    ]
