# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Gradio terminal UI for the Terminus environment."""

from __future__ import annotations

from typing import Any, Optional

import gradio as gr


COMMAND_EXAMPLES = {
    "Inspect workspace": "pwd && ls -la",
    "Create answer file": "mkdir -p /home/user/work && echo done > /home/user/work/answer.txt",
    "Run Python": "python - <<'PY'\nprint('hello from the sandbox')\nPY",
    "Install package": "pip install -q pytest",
    "Run tests": "pytest -q",
}

DEFAULT_SETUP = "mkdir -p /home/user/work"
DEFAULT_VERIFY = "test -f /home/user/work/answer.txt"

_STYLE = """
<style>
.term-shell {
  background: #020403;
  border: 1px solid #16a34a;
  box-shadow: inset 0 0 0 1px #052e16, 0 0 24px rgba(22, 163, 74, 0.18);
  color: #d1fae5;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  min-height: 520px;
  max-height: 720px;
  overflow-y: auto;
  padding: 14px;
}
.term-empty {
  color: #6ee7b7;
  padding: 30px 10px;
  text-align: center;
}
.term-entry {
  border-bottom: 1px solid #064e3b;
  padding: 10px 0 12px;
}
.term-entry:last-child { border-bottom: 0; }
.term-meta {
  color: #34d399;
  font-size: 12px;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0;
}
.term-prompt {
  color: #22c55e;
  white-space: pre-wrap;
  word-break: break-word;
}
.term-prompt::before {
  color: #86efac;
}
.term-output {
  color: #d1fae5;
  margin-top: 8px;
  white-space: pre-wrap;
  word-break: break-word;
}
.term-error {
  color: #fca5a5;
  margin-top: 8px;
  white-space: pre-wrap;
  word-break: break-word;
}
.term-success { color: #86efac; }
.term-fail { color: #f87171; }
.term-final {
  border: 1px solid #16a34a;
  background: #04130a;
  margin-top: 12px;
  padding: 12px;
}
</style>
"""

_EMPTY_TERMINAL = (
    "<div class='term-shell'><div class='term-empty'>"
    "Reset the environment to create an E2B sandbox, then run commands."
    "</div></div>"
)


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _extract_tool_text(result: dict[str, Any]) -> str:
    obs = result.get("observation", {})
    if not isinstance(obs, dict):
        return ""
    tool_result = obs.get("result", {})
    if not isinstance(tool_result, dict):
        return ""
    content = tool_result.get("content", [])
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
    return str(tool_result.get("data", ""))


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": state.get("episode_id"),
        "sandbox_id": state.get("sandbox_id"),
        "step_count": state.get("step_count", 0),
        "commands": len(state.get("commands", [])),
        "setup_commands": len(state.get("setup_results", [])),
        "verify_commands": len(state.get("verify_commands", [])),
        "last_reward": state.get("last_reward"),
        "last_error": state.get("last_error"),
    }


def render_terminal_html(state: dict[str, Any]) -> str:
    entries: list[str] = []

    for index, item in enumerate(state.get("setup_results", []), start=1):
        status = "ok" if item.get("success", False) else "failed"
        status_cls = "term-success" if item.get("success", False) else "term-fail"
        entries.append(
            "<div class='term-entry'>"
            f"<div class='term-meta'>setup {index} - <span class='{status_cls}'>{status}</span></div>"
            f"<div class='term-prompt'>$ {_escape(item.get('command', ''))}</div>"
            f"<div class='term-output'>{_escape(item.get('output', ''))}</div>"
            f"{_error_html(item)}"
            "</div>"
        )

    for index, item in enumerate(state.get("commands", []), start=1):
        status = "ok" if item.get("success", False) else "failed"
        status_cls = "term-success" if item.get("success", False) else "term-fail"
        entries.append(
            "<div class='term-entry'>"
            f"<div class='term-meta'>command {index} - <span class='{status_cls}'>{status}</span></div>"
            f"<div class='term-prompt'>$ {_escape(item.get('command', ''))}</div>"
            f"<div class='term-output'>{_escape(item.get('output', ''))}</div>"
            f"{_error_html(item)}"
            "</div>"
        )

    if state.get("submitted_answer") is not None:
        answer = _escape(state.get("submitted_answer", ""))
        reward = _escape(state.get("last_reward", ""))
        entries.append(
            "<div class='term-final'>"
            "<div class='term-meta'>final answer</div>"
            f"<div>{answer}</div>"
            f"<div class='term-meta'>reward: {reward}</div>"
            "</div>"
        )

    if not entries:
        return _STYLE + _EMPTY_TERMINAL
    return _STYLE + "<div class='term-shell'>" + "\n".join(entries) + "</div>"


def _closed_terminal_html() -> str:
    return (
        _STYLE + "<div class='term-shell'><div class='term-empty'>"
        "Session closed. Reset the sandbox to start a new terminal."
        "</div></div>"
    )


def _error_html(item: dict[str, Any]) -> str:
    if not item.get("error"):
        return ""
    return f"<div class='term-error'>{_escape(item['error'])}</div>"


def terminus_ui_builder(
    web_manager,
    action_fields: list[dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    def current_state() -> dict[str, Any]:
        try:
            return web_manager.get_state()
        except RuntimeError:
            return {}

    with gr.Blocks(title=f"{title} - Terminal") as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "Single-tool terminal environment backed by an E2B sandbox. "
            "Reset creates a fresh session; commands run through `terminal(command=...)`."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Episode")
                setup_input = gr.Textbox(
                    label="Setup commands",
                    value=DEFAULT_SETUP,
                    lines=5,
                    placeholder="One shell command per line",
                )
                verify_input = gr.Textbox(
                    label="Verify commands",
                    value=DEFAULT_VERIFY,
                    lines=4,
                    placeholder="One shell command per line",
                )
                reset_btn = gr.Button("Reset sandbox", variant="primary", size="lg")
                close_btn = gr.Button("Stop / Close session", variant="secondary")
                state_box = gr.JSON(label="Session state", value={})

                gr.Markdown("### Command")
                example_dd = gr.Dropdown(
                    choices=list(COMMAND_EXAMPLES.keys()),
                    label="Load example",
                    value=None,
                    interactive=True,
                )
                command_input = gr.Textbox(
                    label="Terminal command",
                    value="pwd && ls -la",
                    lines=6,
                    placeholder="Run shell commands in the E2B sandbox",
                )
                run_btn = gr.Button("Run command", variant="primary")

                gr.Markdown("### Submit")
                answer_input = gr.Textbox(
                    label="Final answer",
                    value="done",
                    lines=2,
                )
                submit_btn = gr.Button("Submit final answer", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### Terminal")
                terminal_display = gr.HTML(value=_STYLE + _EMPTY_TERMINAL)
                status_bar = gr.Textbox(
                    label="Status",
                    value="Reset the sandbox before running commands.",
                    interactive=False,
                    lines=2,
                )

        async def on_reset(setup_text: str, verify_text: str):
            payload = {
                "setup": _lines(setup_text),
                "verify": _lines(verify_text),
            }
            result = await web_manager.reset_environment(payload)
            state = current_state()
            done = result.get("done", False)
            metadata = result.get("metadata", {})
            if done and isinstance(metadata, dict):
                status = (
                    metadata.get("error") or metadata.get("message") or "Reset failed."
                )
            else:
                status = "Sandbox reset. Setup and verify configuration applied."
            return render_terminal_html(state), _state_summary(state), status

        async def on_close():
            await web_manager._run_sync_in_thread_pool(web_manager.env.close)
            return (
                _closed_terminal_html(),
                {},
                "Session closed. The E2B sandbox was released.",
            )

        async def on_run(command: str):
            if not command.strip():
                return gr.update(), gr.update(), "No command provided."
            result = await web_manager.step_environment(
                {
                    "tool_name": "terminal",
                    "arguments": {"command": command},
                }
            )
            state = current_state()
            text = _extract_tool_text(result)
            status = text if text.startswith("Error:") else "Command completed."
            return render_terminal_html(state), _state_summary(state), status

        async def on_submit(answer: str):
            if not answer.strip():
                return gr.update(), gr.update(), "No final answer provided."
            result = await web_manager.step_environment(
                {
                    "tool_name": "terminal",
                    "arguments": {"final_answer": answer},
                }
            )
            state = current_state()
            status = _extract_tool_text(result) or "Final answer submitted."
            return render_terminal_html(state), _state_summary(state), status

        def on_example(choice: str):
            if choice and choice in COMMAND_EXAMPLES:
                return COMMAND_EXAMPLES[choice]
            return gr.update()

        reset_btn.click(
            fn=on_reset,
            inputs=[setup_input, verify_input],
            outputs=[terminal_display, state_box, status_bar],
        )
        close_btn.click(
            fn=on_close,
            outputs=[terminal_display, state_box, status_bar],
        )
        run_btn.click(
            fn=on_run,
            inputs=[command_input],
            outputs=[terminal_display, state_box, status_bar],
        )
        submit_btn.click(
            fn=on_submit,
            inputs=[answer_input],
            outputs=[terminal_display, state_box, status_bar],
        )
        example_dd.change(
            fn=on_example,
            inputs=[example_dd],
            outputs=[command_input],
        )

    return demo
