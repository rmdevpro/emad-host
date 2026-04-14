"""eMAD Host Chat UI — Gradio-based multi-model client.

Talk to the Imperator (host) or any installed eMAD by selecting
a model from the dropdown. Conversations are tracked via
conversation_id from the chat completions endpoint.
"""

import json
import logging
import os

import gradio as gr
import httpx
import yaml

from mad_client import MADClient

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("ui")


# ── Config ───────────────────────────────────────────────────────────


def load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", "/app/config.yml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
MAD_CFG = CONFIG.get("mads", [{}])[0]
CLIENT = MADClient(
    MAD_CFG.get("name", "eMAD Host"),
    MAD_CFG.get("url", "http://emad-host-langgraph:8000"),
    MAD_CFG.get("hostname", ""),
)

# Models available — host is always present, eMADs are listed in config
DEFAULT_MODELS = CONFIG.get("models", ["host"])


# ── Helpers ──────────────────────────────────────────────────────────


async def _get_health_text() -> str:
    health = await CLIENT.health()
    lines = [f"Status: {health.get('status', 'unknown')}"]
    for key, val in health.items():
        if key != "status":
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


# ── Event handlers ───────────────────────────────────────────────────


async def on_page_load():
    health = await _get_health_text()
    indicator = await check_health()
    return health, indicator


async def check_health() -> str:
    health = await CLIENT.health()
    status = health.get("status", "unknown")
    indicator = {"healthy": "\u2705", "degraded": "\u26a0\ufe0f"}.get(status, "\u274c")
    return f"{indicator} eMAD Host: {status}"


async def on_new_conversation():
    return [], None, ""


async def on_chat_submit(message, history, model, conv_id):
    """Send message. Uses conversation_id for persistence."""
    if not message.strip():
        yield history, conv_id, gr.update()
        return

    history = history + [{"role": "user", "content": message}]
    yield history, conv_id, gr.update()

    # Build messages for the API — just the latest user message
    # (the server's checkpointer has the full history)
    api_messages = [{"role": "user", "content": message}]

    try:
        result = await CLIENT.chat(
            model=model,
            messages=api_messages,
            conversation_id=conv_id,
        )

        # Extract response
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        new_conv_id = result.get("conversation_id", conv_id)

        if not content:
            error = result.get("error", {})
            if isinstance(error, dict):
                content = f"Error: {error.get('message', 'Unknown error')}"
            elif error:
                content = f"Error: {error}"
            else:
                content = "[No response]"

        history = history + [{"role": "assistant", "content": content}]
        yield history, new_conv_id, gr.update()

    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        history = history + [{"role": "assistant", "content": f"Error: {exc}"}]
        yield history, conv_id, gr.update()


async def on_model_changed():
    """Clear chat when switching models."""
    return [], None, ""


async def on_refresh_logs():
    try:
        entries = await CLIENT.query_logs(limit=40)
        if not entries:
            return "No log entries"
        lines = []
        for e in entries:
            ts = (e.get("timestamp") or "?")[-8:]
            lvl = e.get("level", "?")
            msg = e.get("message", "")[:120]
            lines.append(f"[{ts}] [{lvl}] {msg}")
        return "\n".join(lines)
    except (RuntimeError, OSError):
        return "Failed to load logs"


# ── Build the UI ─────────────────────────────────────────────────────


with gr.Blocks(title="eMAD Host", theme=gr.themes.Soft()) as demo:
    # State: current conversation_id (returned by server)
    conv_state = gr.State(None)

    gr.Markdown("# eMAD Host")
    health_bar = gr.Markdown("")

    with gr.Row():
        # ── Left sidebar ─────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            model_selector = gr.Dropdown(
                choices=DEFAULT_MODELS,
                value=DEFAULT_MODELS[0] if DEFAULT_MODELS else "host",
                label="Model",
            )

            new_conv_btn = gr.Button(
                "New Conversation", size="sm", variant="primary"
            )

            conv_id_display = gr.Textbox(
                label="Conversation ID",
                interactive=False,
                lines=1,
                max_lines=1,
            )

            resume_input = gr.Textbox(
                label="Resume conversation",
                placeholder="Paste conversation ID...",
                lines=1,
                max_lines=1,
            )
            resume_btn = gr.Button("Resume", size="sm")

        # ── Chat panel ───────────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", height=550)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Message...",
                    show_label=False,
                    scale=6,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

    # ── System Info (bottom) ─────────────────────────────────
    with gr.Accordion("System Info", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Health")
                health_detail = gr.Textbox(
                    lines=4, interactive=False, show_label=False
                )
            with gr.Column():
                gr.Markdown("#### Logs")
                log_panel = gr.Textbox(
                    lines=6, interactive=False, show_label=False
                )
                refresh_logs_btn = gr.Button("Refresh Logs", size="sm")

    # ── Events ───────────────────────────────────────────────

    demo.load(fn=on_page_load, outputs=[health_detail, health_bar])

    # Model switch — clear chat
    model_selector.change(
        fn=on_model_changed,
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # New conversation
    new_conv_btn.click(
        fn=on_new_conversation,
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # Resume conversation by ID
    def on_resume(resume_id):
        if resume_id and resume_id.strip():
            return [], resume_id.strip(), resume_id.strip()
        return gr.update(), gr.update(), gr.update()

    resume_btn.click(
        fn=on_resume,
        inputs=[resume_input],
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # Chat submit
    send_btn.click(
        fn=on_chat_submit,
        inputs=[msg_input, chatbot, model_selector, conv_state],
        outputs=[chatbot, conv_state, conv_id_display],
    ).then(
        fn=lambda: "", outputs=[msg_input]
    ).then(
        fn=lambda cid: cid or "", inputs=[conv_state], outputs=[conv_id_display]
    )

    msg_input.submit(
        fn=on_chat_submit,
        inputs=[msg_input, chatbot, model_selector, conv_state],
        outputs=[chatbot, conv_state, conv_id_display],
    ).then(
        fn=lambda: "", outputs=[msg_input]
    ).then(
        fn=lambda cid: cid or "", inputs=[conv_state], outputs=[conv_id_display]
    )

    # Logs
    refresh_logs_btn.click(fn=on_refresh_logs, outputs=[log_panel])


if __name__ == "__main__":
    port = CONFIG.get("port", 7860)
    demo.launch(server_name="0.0.0.0", server_port=port)
