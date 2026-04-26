"""eMAD Host Chat UI — Gradio-based multi-model client.

Split into user_step (synchronous, instant: clears input + shows user
message + thinking indicator) and bot_step (async generator: streams the
assistant response token-by-token). The stop button cancels the chain.
Sidebar lists prior conversations filtered by the selected model.
"""

import logging
import os

import gradio as gr
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
INITIAL_MODEL = DEFAULT_MODELS[0] if DEFAULT_MODELS else "host"

THINKING = "_⏳ Thinking…_"


# ── Helpers ──────────────────────────────────────────────────────────


def _format_conv_choices(convs: list[dict]) -> list[tuple[str, str]]:
    """Format conversation list as (label, value) tuples for a Radio component."""
    choices = []
    for c in convs:
        cid = c.get("conversation_id", "")
        title = (c.get("title") or "").strip() or f"({cid[:8]}…)"
        label = title if len(title) <= 60 else title[:57] + "…"
        choices.append((label, cid))
    return choices


# ── Event handlers ───────────────────────────────────────────────────


async def on_startup(model):
    """Load conversation list for the initial model."""
    convs = await CLIENT.list_conversations(model=model, limit=50)
    return gr.update(choices=_format_conv_choices(convs), value=None)


async def on_refresh_conversations(model):
    convs = await CLIENT.list_conversations(model=model, limit=50)
    return gr.update(choices=_format_conv_choices(convs), value=None)


async def on_model_changed(model):
    """Model changed: clear chat, clear conv_id, refresh conversation list."""
    convs = await CLIENT.list_conversations(model=model, limit=50)
    return (
        [],
        None,
        "",
        gr.update(choices=_format_conv_choices(convs), value=None),
    )


def on_new_conversation():
    return [], None, "", gr.update(value=None)


async def on_pick_conversation(conv_id):
    """User clicked a conversation in the sidebar — load its messages."""
    if not conv_id:
        return gr.update(), gr.update(), gr.update()
    conv = await CLIENT.get_conversation(conv_id)
    if conv is None:
        return [], None, ""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in conv.get("messages", [])
    ]
    return messages, conv_id, conv_id


async def on_delete_conversation(conv_id, current_conv_id, model):
    """Delete the selected conversation. If it was active, clear chat."""
    if not conv_id:
        return gr.update(), gr.update(), gr.update(), gr.update()
    await CLIENT.delete_conversation(conv_id)
    convs = await CLIENT.list_conversations(model=model, limit=50)
    choices_update = gr.update(choices=_format_conv_choices(convs), value=None)
    if conv_id == current_conv_id:
        return [], None, "", choices_update
    return gr.update(), gr.update(), gr.update(), choices_update


def user_step(message, history, conv_id):
    """Synchronous: clear input, append user message + thinking indicator.

    Returns instantly so the user sees their message appear immediately.
    """
    msg = (message or "").strip()
    if not msg:
        return "", history, conv_id or "", ""
    history = history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": THINKING},
    ]
    return "", history, conv_id or "", msg


async def bot_step(history, conv_id, pending_message, current_model):
    """Async generator: stream assistant response token-by-token."""
    if not pending_message:
        yield history, conv_id, conv_id or ""
        return

    api_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[:-1]
    ]

    accumulated = ""
    new_conv_id = conv_id
    saw_token = False

    try:
        async for event in CLIENT.chat_stream(
            model=current_model,
            messages=api_messages,
            conversation_id=conv_id,
        ):
            if event.kind == "token":
                accumulated += event.text
                saw_token = True
                history[-1] = {"role": "assistant", "content": accumulated}
                yield history, new_conv_id, new_conv_id or ""
            elif event.kind == "meta" and event.conversation_id:
                new_conv_id = event.conversation_id
                yield history, new_conv_id, new_conv_id or ""
            elif event.kind == "error":
                history[-1] = {
                    "role": "assistant",
                    "content": f"⚠️ Stream error: {event.text}",
                }
                yield history, new_conv_id, new_conv_id or ""
                return
    except Exception as exc:  # noqa: BLE001
        history[-1] = {
            "role": "assistant",
            "content": f"⚠️ {type(exc).__name__}: {exc}",
        }
        yield history, new_conv_id, new_conv_id or ""
        return

    if not saw_token:
        history[-1] = {"role": "assistant", "content": "[No response]"}
        yield history, new_conv_id, new_conv_id or ""


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
    conv_state = gr.State(None)
    pending_msg_state = gr.State("")

    gr.Markdown("# eMAD Host")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            model_selector = gr.Dropdown(
                choices=DEFAULT_MODELS,
                value=INITIAL_MODEL,
                label="Model",
                interactive=True,
            )
            new_conv_btn = gr.Button(
                "➕ New Conversation", size="sm", variant="primary"
            )

            gr.Markdown("#### Conversations")
            conv_list = gr.Radio(
                choices=[],
                label="",
                interactive=True,
                container=False,
            )
            with gr.Row():
                refresh_convs_btn = gr.Button("🔄 Refresh", size="sm")
                delete_conv_btn = gr.Button("🗑 Delete", size="sm", variant="stop")

            conv_id_display = gr.Textbox(
                label="Active conversation ID",
                interactive=False,
                lines=1,
                max_lines=1,
            )

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages",
                height=560,
                show_copy_button=True,
                avatar_images=(None, None),
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type a message and press Enter…",
                    show_label=False,
                    scale=6,
                    autofocus=True,
                    submit_btn=False,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
                stop_btn = gr.Button("Stop", scale=1, variant="stop")

    with gr.Accordion("System Info", open=False):
        gr.Markdown("#### Logs")
        log_panel = gr.Textbox(lines=6, interactive=False, show_label=False)
        refresh_logs_btn = gr.Button("Refresh Logs", size="sm")

    # ── Events ───────────────────────────────────────────────

    demo.load(
        fn=on_startup,
        inputs=[model_selector],
        outputs=[conv_list],
    )

    model_selector.change(
        fn=on_model_changed,
        inputs=[model_selector],
        outputs=[chatbot, conv_state, conv_id_display, conv_list],
    )

    new_conv_btn.click(
        fn=on_new_conversation,
        outputs=[chatbot, conv_state, conv_id_display, conv_list],
    )

    refresh_convs_btn.click(
        fn=on_refresh_conversations,
        inputs=[model_selector],
        outputs=[conv_list],
    )

    conv_list.change(
        fn=on_pick_conversation,
        inputs=[conv_list],
        outputs=[chatbot, conv_state, conv_id_display],
    )

    delete_conv_btn.click(
        fn=on_delete_conversation,
        inputs=[conv_list, conv_state, model_selector],
        outputs=[chatbot, conv_state, conv_id_display, conv_list],
    )

    def _make_chat_chain(trigger):
        return trigger(
            fn=user_step,
            inputs=[msg_input, chatbot, conv_state],
            outputs=[msg_input, chatbot, conv_id_display, pending_msg_state],
        ).then(
            fn=bot_step,
            inputs=[chatbot, conv_state, pending_msg_state, model_selector],
            outputs=[chatbot, conv_state, conv_id_display],
        ).then(
            fn=on_refresh_conversations,
            inputs=[model_selector],
            outputs=[conv_list],
        )

    send_event = _make_chat_chain(send_btn.click)
    enter_event = _make_chat_chain(msg_input.submit)

    stop_btn.click(fn=None, cancels=[send_event, enter_event])

    refresh_logs_btn.click(fn=on_refresh_logs, outputs=[log_panel])


if __name__ == "__main__":
    port = CONFIG.get("port", 7860)
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=port
    )
