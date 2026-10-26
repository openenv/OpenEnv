"""
Custom Gradio UI for the Jupyter Agent environment.

A single-page interface with:
  - Code editor with example snippets (prefill dropdown)
  - Shell command input with examples
  - Reset / Get State controls
  - Live notebook panel that updates after every action
"""

from typing import Any, Dict, List, Optional

import gradio as gr

# ── Example snippets ──────────────────────────────────────────────────────────

CODE_EXAMPLES = {
    "Hello World": 'print("Hello, World!")',
    "Variable Persistence": "x = 42\nprint(f'x = {x}')",
    "Data Analysis": (
        "import numpy as np\n"
        "data = np.array([1, 4, 9, 16, 25])\n"
        "print(f'mean: {data.mean():.2f}')\n"
        "print(f'std:  {data.std():.2f}')"
    ),
    "Pandas DataFrame": (
        "import pandas as pd\n"
        "df = pd.DataFrame({'name': ['Alice','Bob','Carol'], 'score': [95, 82, 91]})\n"
        "print(df)\n"
        "print(f'\\nAvg score: {df.score.mean():.1f}')"
    ),
    "Matplotlib Plot": (
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "x = np.linspace(0, 2*np.pi, 100)\n"
        "plt.figure(figsize=(6,3))\n"
        "plt.plot(x, np.sin(x), label='sin')\n"
        "plt.plot(x, np.cos(x), label='cos')\n"
        "plt.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "Error Example": "print(1 / 0)  # ZeroDivisionError",
}

SHELL_EXAMPLES = {
    "List Files": "ls -la",
    "Python Version": "python --version",
    "Installed Packages": "pip list | head -20",
    "Install Package": "pip install httpx",
    "Working Directory": "pwd && ls",
}

DEFAULT_SETUP = "mkdir -p /home/user/work"
DEFAULT_VERIFY = "test -f /home/user/work/answer.txt"

# ── Notebook HTML renderer ────────────────────────────────────────────────────
# Colors match the OpenEnv GitHub-style dark theme (gradio_theme.py):
#   background: #0d1117  |  surface: #161b22  |  border: #30363d
#   text: #c9d1d9        |  muted: #8b949e    |  success: #3fb950
#   error: #f85149       |  shell: #58a6ff

_STYLE = """
<style>
.nb-wrap { font-family: 'JetBrains Mono','Fira Code','Cascadia Code',monospace; }
.nb-cell {
    border: 1px solid #30363d;
    margin: 8px 0;
    overflow: hidden;
    background: #161b22;
}
.nb-success { border-left: 4px solid #3fb950; }
.nb-error   { border-left: 4px solid #f85149; }
.nb-shell   { border-left: 4px solid #58a6ff; }
.nb-prompt  { font-size: 11px; color: #8b949e; padding: 4px 10px 2px; background: #161b22; }
.nb-input   { background: #161b22; color: #e6edf3; padding: 6px 10px 8px;
              white-space: pre-wrap; word-break: break-all;
              font-size: 13px; line-height: 1.5; }
.nb-output  { border-top: 1px solid #30363d; padding: 8px 10px; background: #0d1117;
              color: #c9d1d9; white-space: pre-wrap; word-break: break-word;
              font-size: 13px; line-height: 1.5; }
.nb-err-txt { color: #f85149; }
.nb-img     { max-width: 100%; margin: 6px 0; display: block; }
</style>
"""

_EMPTY_NB = (
    "<div style='color:#8b949e;font-style:italic;padding:20px;text-align:center;"
    "background:#161b22;border:1px solid #30363d'>"
    "No cells yet — click <b>Reset Episode</b> to start, then run some code."
    "</div>"
)


def _e(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lines(value: str) -> List[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def render_notebook_html(cells: List[Dict[str, Any]]) -> str:
    if not cells:
        return _STYLE + _EMPTY_NB

    parts = [_STYLE, "<div class='nb-wrap'>"]
    for cell in cells:
        t = cell.get("cell_type", "code")
        ok = cell.get("success", True)
        ec = cell.get("execution_count", "")

        border = "nb-shell" if t == "shell" else ("nb-success" if ok else "nb-error")
        prompt = (
            f"Shell [{ec}]:"
            if t == "shell"
            else (f"In [{ec}]:" if ok else f"In [{ec}]: ✗")
        )

        code_html = _e(cell.get("code", ""))
        out_html = ""

        if cell.get("output"):
            out_html += f"<span>{_e(cell['output'])}</span>"
        if cell.get("error"):
            out_html += f"<span class='nb-err-txt'>{_e(cell['error'])}</span>"
        for b64 in cell.get("images", []):
            out_html += f"<img class='nb-img' src='data:image/png;base64,{b64}'/>"

        output_section = f"<div class='nb-output'>{out_html}</div>" if out_html else ""

        parts.append(f"""
<div class='nb-cell {border}'>
  <div class='nb-prompt'>{prompt}</div>
  <div class='nb-input'>{code_html}</div>
  {output_section}
</div>""")

    parts.append("</div>")
    return "\n".join(parts)


def _closed_notebook_html() -> str:
    return (
        _STYLE
        + "<div style='color:#8b949e;font-style:italic;padding:20px;text-align:center;"
        "background:#161b22;border:1px solid #30363d'>"
        "Session closed. Reset the episode to create a new E2B sandbox."
        "</div>"
    )


# ── gradio_builder ────────────────────────────────────────────────────────────


def jupyter_ui_builder(
    web_manager,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:

    def _cells_from_state() -> List[Dict]:
        return web_manager.get_state().get("cells", [])

    def _state_summary() -> Dict:
        s = web_manager.get_state()
        return {
            "episode_id": s.get("episode_id"),
            "step_count": s.get("step_count", 0),
            "sandbox_id": s.get("sandbox_id"),
            "last_cell_success": s.get("last_cell_success", True),
            "cell_count": len(s.get("cells", [])),
            "setup_commands": len(s.get("setup_results", [])),
            "verify_commands": len(s.get("verify_commands", [])),
            "last_reward": s.get("last_reward"),
        }

    with gr.Blocks(title=f"{title} — Notebook") as demo:
        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "Interactive Jupyter notebook backed by an **E2B sandbox**. "
            "Each episode is an isolated Python kernel — variables persist between cells."
        )

        with gr.Row():
            # ── Left column: controls ──────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                # Reset
                gr.Markdown("### Episode")
                setup_input = gr.Textbox(
                    label="Setup commands",
                    value=DEFAULT_SETUP,
                    lines=4,
                    placeholder="One shell command per line",
                )
                verify_input = gr.Textbox(
                    label="Verify commands",
                    value=DEFAULT_VERIFY,
                    lines=4,
                    placeholder="One shell command per line",
                )
                reset_btn = gr.Button("🔄 Reset Episode", variant="primary", size="lg")
                close_btn = gr.Button("Stop / Close Session", variant="secondary")
                episode_state = gr.JSON(label="Episode State", value={})

                gr.Markdown("---")

                # ── Execute Code ──
                gr.Markdown("### Execute Code")
                code_example_dd = gr.Dropdown(
                    choices=list(CODE_EXAMPLES.keys()),
                    label="Load example",
                    value=None,
                    interactive=True,
                )
                code_input = gr.Code(
                    label="Python code",
                    language="python",
                    value='print("Hello from E2B!")',
                    lines=8,
                )
                with gr.Row():
                    run_btn = gr.Button("▶ Run Code", variant="primary")
                    edit_btn = gr.Button("✏️ Edit & Re-run Last", variant="secondary")

                gr.Markdown("---")

                # ── Shell Command ──
                gr.Markdown("### Shell Command")
                shell_example_dd = gr.Dropdown(
                    choices=list(SHELL_EXAMPLES.keys()),
                    label="Load example",
                    value=None,
                    interactive=True,
                )
                shell_input = gr.Textbox(
                    label="Shell command",
                    placeholder="ls -la",
                    lines=1,
                )
                shell_btn = gr.Button("$ Run Shell", variant="secondary")

                gr.Markdown("---")

                # ── Notebook State ──
                gr.Markdown("### Notebook State")
                gr.Markdown(
                    "Returns a compact text summary of recent cells — "
                    "useful to check what has already been computed."
                )
                state_btn = gr.Button("📋 Get Notebook State")
                state_output = gr.Textbox(
                    label="State summary",
                    lines=6,
                    interactive=False,
                )

            # ── Right column: notebook viewer ──────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Live Notebook")
                notebook_display = gr.HTML(
                    value=_STYLE + _EMPTY_NB,
                    label="Notebook",
                )

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = gr.Textbox(
            value="Ready — click Reset Episode to start.",
            label="Status",
            interactive=False,
            lines=1,
        )

        # ── Event handlers ────────────────────────────────────────────────────

        async def on_reset(setup_text: str, verify_text: str):
            result = await web_manager.reset_environment(
                {
                    "setup": _lines(setup_text),
                    "verify": _lines(verify_text),
                }
            )
            cells = _cells_from_state()
            done = result.get("done", False)
            metadata = result.get("metadata", {})
            if done and isinstance(metadata, dict):
                status = (
                    metadata.get("error") or metadata.get("message") or "Reset failed."
                )
            else:
                setup_count = len(_lines(setup_text))
                verify_count = len(_lines(verify_text))
                status = (
                    "Episode reset — fresh E2B sandbox created. "
                    f"Setup: {setup_count}; verify: {verify_count}."
                )
            return (
                render_notebook_html(cells),
                _state_summary(),
                status,
            )

        async def on_close():
            await web_manager._run_sync_in_thread_pool(web_manager.env.close)
            return (
                _closed_notebook_html(),
                {},
                "Session closed. The E2B sandbox was released.",
            )

        async def on_run_code(code: str):
            if not code.strip():
                return gr.update(), gr.update(), "⚠ No code to run."
            action = {
                "tool_name": "add_and_execute_code_cell",
                "arguments": {"code": code},
            }
            await web_manager.step_environment(action)
            cells = _cells_from_state()
            ok = cells[-1]["success"] if cells else True
            status = "✓ Cell executed." if ok else "✗ Cell raised an error."
            return render_notebook_html(cells), _state_summary(), status

        async def on_edit_run(code: str):
            if not code.strip():
                return gr.update(), gr.update(), "⚠ No code to run."
            action = {
                "tool_name": "edit_and_execute_current_cell",
                "arguments": {"code": code},
            }
            await web_manager.step_environment(action)
            cells = _cells_from_state()
            ok = cells[-1]["success"] if cells else True
            status = (
                "✓ Last cell replaced and re-executed."
                if ok
                else "✗ Edit raised an error."
            )
            return render_notebook_html(cells), _state_summary(), status

        async def on_shell(command: str):
            if not command.strip():
                return gr.update(), gr.update(), "⚠ No command to run."
            action = {
                "tool_name": "execute_shell_command",
                "arguments": {"command": command},
            }
            await web_manager.step_environment(action)
            cells = _cells_from_state()
            return render_notebook_html(cells), _state_summary(), f"✓ Shell: {command}"

        async def on_get_state():
            action = {"tool_name": "get_notebook_state", "arguments": {}}
            result = await web_manager.step_environment(action)
            # Extract text from MCP result
            obs = result.get("observation", {})
            summary = ""
            if isinstance(obs, dict):
                res = obs.get("result", {})
                if isinstance(res, dict):
                    content = res.get("content", [])
                    if content and isinstance(content, list):
                        summary = content[0].get("text", "")
                    if not summary:
                        summary = res.get("data", "")
            return summary or "(no state yet)", "📋 Notebook state retrieved."

        def on_code_example(choice: str):
            if choice and choice in CODE_EXAMPLES:
                return CODE_EXAMPLES[choice]
            return gr.update()

        def on_shell_example(choice: str):
            if choice and choice in SHELL_EXAMPLES:
                return SHELL_EXAMPLES[choice]
            return gr.update()

        # ── Wire up ───────────────────────────────────────────────────────────

        reset_btn.click(
            fn=on_reset,
            inputs=[setup_input, verify_input],
            outputs=[notebook_display, episode_state, status_bar],
        )
        close_btn.click(
            fn=on_close,
            outputs=[notebook_display, episode_state, status_bar],
        )
        run_btn.click(
            fn=on_run_code,
            inputs=[code_input],
            outputs=[notebook_display, episode_state, status_bar],
        )
        edit_btn.click(
            fn=on_edit_run,
            inputs=[code_input],
            outputs=[notebook_display, episode_state, status_bar],
        )
        shell_btn.click(
            fn=on_shell,
            inputs=[shell_input],
            outputs=[notebook_display, episode_state, status_bar],
        )
        state_btn.click(
            fn=on_get_state,
            outputs=[state_output, status_bar],
        )
        code_example_dd.change(
            fn=on_code_example,
            inputs=[code_example_dd],
            outputs=[code_input],
        )
        shell_example_dd.change(
            fn=on_shell_example,
            inputs=[shell_example_dd],
            outputs=[shell_input],
        )

    return demo
