"""eMAD Host Chat UI — Gradio-based multi-model client.

Streaming chatbot with model selector, conversation management,
and stop button. Talks to the backend via OpenAI-compatible
/v1/chat/completions endpoint.
"""

import json
import logging
import os
import time

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
DEFAULT_MODELS = CONFIG.get("models", ["host"])


# ── Session store (in-memory, per UI instance) ───────────────────────
# Maps conversation_id -> {model, title, updated}


_sessions: dict[str, dict] = {}


def _save_session(conv_id: str, model: str, last_msg: str):
    """Track a conversation in the session store."""
    if not conv_id:
        return
    title = last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
    _sessions[conv_id] = {
        "model": model,
        "title": title,
        "updated": time.time(),
    }


def _get_sessions_for_model(model: str) -> list[tuple[str, str]]:
    """Return (conv_id, title) pairs for a model, newest first."""
    items = [
        (cid, s["title"])
        for cid, s in _sessions.items()
        if s["model"] == model
    ]
    items.sort(key=lambda x: _sessions[x[0]]["updated"], reverse=True)
    return items


# ── Event handlers ───────────────────────────────────────────────────


def on_new_conversation():
    return [], None, ""


def on_model_changed(model):
    """Switch model — clear chat, update session list."""
    sessions = _get_sessions_for_model(model)
    choices = [f"{title} | {cid}" for cid, title in sessions]
    return [], None, "", gr.update(choices=choices, value=None)


async def on_chat_submit(message, history, model, conv_id):
    """Stream response tokens into the chat. Yields intermediate states."""
    if not message or not message.strip():
        yield history, conv_id, conv_id or ""
        return

    # Yield immediately — user sees their message + thinking indicator
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, conv_id, conv_id or ""

    # Send only the latest user message — checkpointer has full history
    api_messages = [{"role": "user", "content": message}]

    try:
        accumulated = ""
        new_conv_id = conv_id

        # Try streaming first
        async for token in CLIENT.chat_stream(
            model=model,
            messages=api_messages,
            conversation_id=conv_id,
        ):
            accumulated += token
            history[-1] = {"role": "assistant", "content": accumulated}
            yield history, new_conv_id, new_conv_id or ""

        # If streaming returned nothing, fall back to non-streaming
        if not accumulated:
            result = await CLIENT.chat(
                model=model,
                messages=api_messages,
                conversation_id=conv_id,
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            new_conv_id = result.get("conversation_id", conv_id)
            history[-1] = {"role": "assistant", "content": content or "[No response]"}

        # If we still don't have a conversation_id (streaming doesn't return it),
        # make a lightweight non-streaming call to get it
        if not new_conv_id and accumulated:
            try:
                result = await CLIENT.chat(
                    model=model,
                    messages=[{"role": "user", "content": ""}],
                    conversation_id=conv_id,
                )
                new_conv_id = result.get("conversation_id", conv_id)
            except Exception:
                pass

        # Save to session store
        _save_session(new_conv_id, model, message)
        yield history, new_conv_id, new_conv_id or ""

    except httpx.ReadTimeout:
        history[-1] = {"role": "assistant", "content": "Response timed out. The agent may still be processing. Try asking 'what happened?' to resume."}
        yield history, conv_id, conv_id or ""
    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        history[-1] = {"role": "assistant", "content": f"Error: {exc}"}
        yield history, conv_id, conv_id or ""


def on_session_selected(selection, model):
    """Load a previous conversation by its ID."""
    if not selection or "|" not in selection:
        return gr.update(), gr.update(), gr.update()
    conv_id = selection.split("|")[-1].strip()
    return [], conv_id, conv_id


def on_refresh_sessions(model):
    """Refresh the session list for the current model."""
    sessions = _get_sessions_for_model(model)
    choices = [f"{title} | {cid}" for cid, title in sessions]
    return gr.update(choices=choices)


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


with gr.Blocks(
    title="eMAD Host",
    theme=gr.themes.Soft(),
    css="""
        .chatbot-container { min-height: 500px; }
        .stop-btn { background-color: #ef4444 !important; }
    """,
) as demo:
    conv_state = gr.State(None)
    msg_state = gr.State("")  # Holds message while input clears

    with gr.Row():
        # ── Left sidebar ─────────────────────────────────────
        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### eMAD Host")

            model_selector = gr.Dropdown(
                choices=DEFAULT_MODELS,
                value=DEFAULT_MODELS[0] if DEFAULT_MODELS else "host",
                label="Agent",
            )

            new_conv_btn = gr.Button(
                "New Conversation", size="sm", variant="primary"
            )

            gr.Markdown("#### Conversations")
            session_list = gr.Radio(
                choices=[],
                value=None,
                label="",
                show_label=False,
                interactive=True,
            )
            refresh_sessions_btn = gr.Button("Refresh", size="sm")

            gr.Markdown("---")
            conv_id_display = gr.Textbox(
                label="Conversation ID",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            resume_input = gr.Textbox(
                label="Resume by ID",
                placeholder="Paste conversation ID...",
                lines=1,
                max_lines=1,
            )
            resume_btn = gr.Button("Resume", size="sm")

        # ── Chat panel ───────────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages",
                height=600,
                show_label=False,
                elem_classes=["chatbot-container"],
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Message...",
                    show_label=False,
                    scale=7,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
                stop_btn = gr.Button("Stop", scale=1, variant="stop")

    # ── System Info (bottom) ─────────────────────────────────
    with gr.Accordion("System Info", open=False):
        log_panel = gr.Textbox(lines=6, interactive=False, show_label=False)
        refresh_logs_btn = gr.Button("Refresh Logs", size="sm")

    # ── Events ───────────────────────────────────────────────

    # Model switch — clear chat, update session list
    model_selector.change(
        fn=on_model_changed,
        inputs=[model_selector],
        outputs=[chatbot, conv_state, conv_id_display, session_list],
    )

    # New conversation
    new_conv_btn.click(
        fn=on_new_conversation,
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # Session list — select previous conversation
    session_list.change(
        fn=on_session_selected,
        inputs=[session_list, model_selector],
        outputs=[chatbot, conv_state, conv_id_display],
    )

    refresh_sessions_btn.click(
        fn=on_refresh_sessions,
        inputs=[model_selector],
        outputs=[session_list],
    )

    # Resume by pasting ID
    def on_resume(resume_id):
        if resume_id and resume_id.strip():
            return [], resume_id.strip(), resume_id.strip()
        return gr.update(), gr.update(), gr.update()

    resume_btn.click(
        fn=on_resume,
        inputs=[resume_input],
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # Chat submit — Send button
    # 1. Stash message and clear input immediately
    # 2. Run streaming generator
    submit_click = send_btn.click(
        fn=lambda m: ("", m),
        inputs=[msg_input],
        outputs=[msg_input, msg_state],
    ).then(
        fn=on_chat_submit,
        inputs=[msg_state, chatbot, model_selector, conv_state],
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # Chat submit — Enter key
    submit_enter = msg_input.submit(
        fn=lambda m: ("", m),
        inputs=[msg_input],
        outputs=[msg_input, msg_state],
    ).then(
        fn=on_chat_submit,
        inputs=[msg_state, chatbot, model_selector, conv_state],
        outputs=[chatbot, conv_state, conv_id_display],
    )

    # Stop button cancels the running generator
    stop_btn.click(fn=None, cancels=[submit_click, submit_enter])

    # Logs
    refresh_logs_btn.click(fn=on_refresh_logs, outputs=[log_panel])


if __name__ == "__main__":
    port = CONFIG.get("port", 7860)
    demo.launch(server_name="0.0.0.0", server_port=port)
