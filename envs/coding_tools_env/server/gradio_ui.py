# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Gradio UI for coding_tools_env."""

from __future__ import annotations

import json
from typing import Any, Optional

import gradio as gr


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "bash": {
        "icon": "🖥️",
        "description": "Execute a bash command inside the E2B sandbox.",
        "signature": "bash(command: str, timeout: float = 30)",
    },
    "read": {
        "icon": "📖",
        "description": "Read a file from the sandbox. Optional `offset` and `limit` slice the lines.",
        "signature": "read(file_path: str, offset?: int, limit?: int)",
    },
    "write": {
        "icon": "✏️",
        "description": "Create or overwrite a file with `content`.",
        "signature": "write(file_path: str, content: str)",
    },
    "edit": {
        "icon": "🔧",
        "description": "Find-and-replace `old_string` with `new_string` in a file. Use `replace_all` for every occurrence.",
        "signature": "edit(file_path, old_string, new_string, replace_all=False)",
    },
    "multi_edit": {
        "icon": "🔁",
        "description": "Apply a list of {old_string, new_string, replace_all?} edits sequentially in one file.",
        "signature": "multi_edit(file_path, edits: list[dict])",
    },
    "glob": {
        "icon": "🔍",
        "description": "Glob for files matching a shell-style pattern, optionally rooted at `path`.",
        "signature": "glob(pattern: str, path?: str)",
    },
    "grep": {
        "icon": "🧵",
        "description": "Regex search across files. `include` is an optional file glob filter.",
        "signature": "grep(pattern: str, path?: str, include?: str)",
    },
    "ls": {
        "icon": "📂",
        "description": "List a directory. `ignore` is a comma-separated list of patterns to skip.",
        "signature": "ls(path: str = '.', ignore?: list[str])",
    },
    "todo_write": {
        "icon": "🗒️",
        "description": "Replace the agent's todo list. Each item: {id, content, status, priority}. Only one `in_progress` at a time.",
        "signature": "todo_write(todos: list[dict])",
    },
    "submit_solution": {
        "icon": "🚀",
        "description": "Run the configured verify commands and finalise the episode with a reward.",
        "signature": "submit_solution()",
    },
}

TOOL_CHOICES = [(f"{spec['icon']}  {name}", name) for name, spec in TOOL_SPECS.items()]


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _extract_tool_error(result: dict[str, Any]) -> bool:
    obs = result.get("observation", {})
    if not isinstance(obs, dict):
        return False
    tool_result = obs.get("result", {})
    if not isinstance(tool_result, dict):
        return False
    return bool(tool_result.get("is_error", False))


def _format_status(state: dict[str, Any]) -> str:
    if not state:
        return "**No active session.** Configure setup/verify and click *Reset sandbox*."
    sandbox_id = state.get("sandbox_id") or "—"
    step_count = state.get("step_count", 0)
    submitted = state.get("submitted", False)
    last_reward = state.get("last_reward")
    reward_str = "—" if last_reward is None else f"{last_reward}"
    submitted_str = "✅ submitted" if submitted else "⏳ in progress"
    return (
        f"**Session active** · sandbox `{sandbox_id}` · "
        f"steps `{step_count}` · {submitted_str} · reward `{reward_str}`"
    )


def _format_history(state: dict[str, Any]) -> list[list[str]]:
    history = state.get("tool_history", []) or []
    rows: list[list[str]] = []
    for entry in history[-25:][::-1]:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool", ""))
        ok = "✅" if entry.get("ok") else "❌"
        output = str(entry.get("output") or entry.get("error") or "")
        if len(output) > 140:
            output = output[:137] + "…"
        rows.append([ok, tool, output])
    return rows


def coding_tools_ui_builder(
    web_manager,
    action_fields: list[dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    def state_payload() -> dict[str, Any]:
        try:
            return web_manager.get_state()
        except RuntimeError:
            return {}

    with gr.Blocks(
        title=f"{title} - Coding Tools",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as demo:
        gr.Markdown(f"# 🧰 {title}")
        gr.Markdown(
            "E2B-backed coding environment exposing a SETA-style tool surface. "
            "Pick a tool, fill the inputs, and inspect the output, state, and call history."
        )

        status_md = gr.Markdown(_format_status({}))

        with gr.Row():
            # ───────── Left: Session controls ─────────
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion("Session controls", open=True):
                    setup_input = gr.Textbox(
                        label="Setup commands (one per line)",
                        value="mkdir -p /home/user/work",
                        lines=3,
                    )
                    verify_input = gr.Textbox(
                        label="Verify commands (one per line)",
                        value="pytest -q",
                        lines=2,
                    )
                    with gr.Row():
                        reset_btn = gr.Button("🔄 Reset sandbox", variant="primary")
                        close_btn = gr.Button("🛑 Close", variant="secondary")

            # ───────── Middle: Tool selection + dynamic inputs ─────────
            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### Tool")
                tool_dropdown = gr.Dropdown(
                    choices=TOOL_CHOICES,
                    value="bash",
                    label="Select a tool",
                    interactive=True,
                )
                tool_help = gr.Markdown(
                    f"**{TOOL_SPECS['bash']['signature']}**  \n{TOOL_SPECS['bash']['description']}"
                )

                # One Group per tool. Visibility toggled by tool_dropdown.
                with gr.Group(visible=True) as g_bash:
                    bash_command = gr.Textbox(
                        label="command", value="pwd && ls -la", lines=3
                    )
                    bash_timeout = gr.Number(label="timeout (seconds)", value=30)

                with gr.Group(visible=False) as g_read:
                    read_path = gr.Textbox(
                        label="file_path", value="/home/user/work/main.py"
                    )
                    with gr.Row():
                        read_offset = gr.Number(label="offset (optional)", value=None)
                        read_limit = gr.Number(label="limit (optional)", value=None)

                with gr.Group(visible=False) as g_write:
                    write_path = gr.Textbox(
                        label="file_path", value="/home/user/work/main.py"
                    )
                    write_content = gr.Textbox(label="content", lines=8)

                with gr.Group(visible=False) as g_edit:
                    edit_path = gr.Textbox(
                        label="file_path", value="/home/user/work/main.py"
                    )
                    edit_old = gr.Textbox(label="old_string", lines=3)
                    edit_new = gr.Textbox(label="new_string", lines=3)
                    edit_replace_all = gr.Checkbox(label="replace_all", value=False)

                with gr.Group(visible=False) as g_multi_edit:
                    multi_edit_path = gr.Textbox(
                        label="file_path", value="/home/user/work/main.py"
                    )
                    multi_edit_json = gr.Code(
                        label="edits (JSON array)",
                        language="json",
                        value=(
                            '[\n'
                            '  {"old_string": "TODO", "new_string": "DONE", "replace_all": false}\n'
                            ']'
                        ),
                        lines=8,
                    )

                with gr.Group(visible=False) as g_glob:
                    glob_pattern = gr.Textbox(label="pattern", value="**/*.py")
                    glob_path = gr.Textbox(
                        label="path (optional)", value="/home/user/work"
                    )

                with gr.Group(visible=False) as g_grep:
                    grep_pattern = gr.Textbox(label="pattern", value="TODO")
                    grep_path = gr.Textbox(
                        label="path (optional)", value="/home/user/work"
                    )
                    grep_include = gr.Textbox(
                        label="include (optional file glob)", value=""
                    )

                with gr.Group(visible=False) as g_ls:
                    ls_path = gr.Textbox(label="path", value="/home/user/work")
                    ls_ignore = gr.Textbox(
                        label="ignore (comma-separated globs)", value=""
                    )

                with gr.Group(visible=False) as g_todo:
                    todo_json = gr.Code(
                        label="todos (JSON array)",
                        language="json",
                        value=(
                            '[\n'
                            '  {"id":"1","content":"Inspect files",'
                            '"status":"in_progress","priority":"high"}\n'
                            ']'
                        ),
                        lines=8,
                    )

                with gr.Group(visible=False) as g_submit:
                    gr.Markdown(
                        "_Submit runs the configured verify commands and ends the "
                        "episode with a reward._"
                    )

                run_btn = gr.Button("▶️  Run tool", variant="primary", size="lg")

            # ───────── Right: Output + State ─────────
            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### Output")
                output_status = gr.Markdown("_Awaiting first call._")
                output_view = gr.Code(
                    label="Tool output (text)",
                    language=None,
                    value="",
                    lines=14,
                )
                with gr.Accordion("Raw step response", open=False):
                    raw_response = gr.Code(
                        label="raw JSON",
                        language="json",
                        value="",
                        lines=14,
                    )

                gr.Markdown("### State")
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        state_summary = gr.Markdown(_format_status({}))
                        history_table = gr.Dataframe(
                            headers=["✓", "tool", "output"],
                            datatype=["str", "str", "str"],
                            row_count=(0, "dynamic"),
                            wrap=True,
                            label="Tool history (latest 25)",
                        )
                    with gr.Tab("JSON"):
                        state_json = gr.Code(
                            label="full session state",
                            language="json",
                            value="{}",
                            lines=18,
                        )

        # ───────── Tool dropdown → input form visibility ─────────
        TOOL_GROUPS = {
            "bash": g_bash,
            "read": g_read,
            "write": g_write,
            "edit": g_edit,
            "multi_edit": g_multi_edit,
            "glob": g_glob,
            "grep": g_grep,
            "ls": g_ls,
            "todo_write": g_todo,
            "submit_solution": g_submit,
        }
        group_components = list(TOOL_GROUPS.values())

        def on_tool_change(tool: str):
            spec = TOOL_SPECS.get(tool, {})
            help_md = (
                f"**{spec.get('signature', tool)}**  \n{spec.get('description', '')}"
            )
            updates = [gr.update(visible=(name == tool)) for name in TOOL_GROUPS]
            return [help_md, *updates]

        tool_dropdown.change(
            on_tool_change, inputs=[tool_dropdown], outputs=[tool_help, *group_components]
        )

        # ───────── Result rendering helper ─────────
        def render_result(tool: str, raw: dict[str, Any]) -> tuple[str, str, str, str, str, list[list[str]]]:
            text = _extract_tool_text(raw)
            is_error = _extract_tool_error(raw) or text.startswith("ERROR:") or text.startswith("Error:")
            badge = "❌ error" if is_error else "✅ ok"
            status_line = f"**{tool}** — {badge}"
            state = state_payload()
            return (
                status_line,                       # output_status
                text,                              # output_view
                json.dumps(raw, indent=2),         # raw_response
                _format_status(state),             # state_summary (top + summary panel — same content)
                json.dumps(state, indent=2, default=str),  # state_json
                _format_history(state),            # history_table
            )

        # ───────── Session handlers ─────────
        async def on_reset(setup_text: str, verify_text: str):
            raw = await web_manager.reset_environment(
                {"setup": _lines(setup_text), "verify": _lines(verify_text)}
            )
            state = state_payload()
            obs = raw.get("observation", {}) if isinstance(raw, dict) else {}
            meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
            err = meta.get("error") if isinstance(meta, dict) else None
            sandbox_id = state.get("sandbox_id") or meta.get("sandbox_id") or "—"
            if err:
                status_line = f"**reset** — ❌ {err}"
                text = err
            else:
                status_line = f"**reset** — ✅ sandbox `{sandbox_id}` ready"
                text = f"Sandbox ready: {sandbox_id}"
            return (
                _format_status(state),
                status_line,
                text,
                json.dumps(raw, indent=2),
                _format_status(state),
                json.dumps(state, indent=2, default=str),
                _format_history(state),
            )

        async def on_close():
            await web_manager._run_sync_in_thread_pool(web_manager.env.close)
            return (
                _format_status({}),
                "**close** — 🛑 session closed",
                "Session closed.",
                "{}",
                _format_status({}),
                "{}",
                [],
            )

        # ───────── Universal run handler ─────────
        async def on_run(
            tool: str,
            # bash
            bash_command: str, bash_timeout: float,
            # read
            read_path: str, read_offset: float | None, read_limit: float | None,
            # write
            write_path: str, write_content: str,
            # edit
            edit_path: str, edit_old: str, edit_new: str, edit_replace_all: bool,
            # multi_edit
            multi_edit_path: str, multi_edit_json: str,
            # glob
            glob_pattern: str, glob_path: str,
            # grep
            grep_pattern: str, grep_path: str, grep_include: str,
            # ls
            ls_path: str, ls_ignore: str,
            # todo_write
            todo_json: str,
        ):
            try:
                if tool == "bash":
                    args = {
                        "command": bash_command,
                        "timeout": float(bash_timeout) if bash_timeout else 30,
                    }
                elif tool == "read":
                    args = {"file_path": read_path}
                    if read_offset not in (None, "", 0):
                        args["offset"] = int(read_offset)
                    if read_limit not in (None, ""):
                        args["limit"] = int(read_limit)
                elif tool == "write":
                    args = {"file_path": write_path, "content": write_content}
                elif tool == "edit":
                    args = {
                        "file_path": edit_path,
                        "old_string": edit_old,
                        "new_string": edit_new,
                        "replace_all": bool(edit_replace_all),
                    }
                elif tool == "multi_edit":
                    args = {
                        "file_path": multi_edit_path,
                        "edits": json.loads(multi_edit_json),
                    }
                elif tool == "glob":
                    args = {"pattern": glob_pattern}
                    if glob_path.strip():
                        args["path"] = glob_path
                elif tool == "grep":
                    args = {"pattern": grep_pattern}
                    if grep_path.strip():
                        args["path"] = grep_path
                    if grep_include.strip():
                        args["include"] = grep_include
                elif tool == "ls":
                    args = {"path": ls_path or "."}
                    ignore = _csv_list(ls_ignore)
                    if ignore:
                        args["ignore"] = ignore
                elif tool == "todo_write":
                    args = {"todos": json.loads(todo_json)}
                elif tool == "submit_solution":
                    args = {}
                else:
                    return (
                        _format_status(state_payload()),
                        f"**{tool}** — ❌ unknown tool",
                        f"unknown tool: {tool}",
                        "{}",
                        _format_status(state_payload()),
                        "{}",
                        [],
                    )
            except (json.JSONDecodeError, ValueError) as exc:
                state = state_payload()
                return (
                    _format_status(state),
                    f"**{tool}** — ❌ bad input: {exc}",
                    f"Input parse error: {exc}",
                    "{}",
                    _format_status(state),
                    json.dumps(state, indent=2, default=str),
                    _format_history(state),
                )

            raw = await web_manager.step_environment(
                {"tool_name": tool, "arguments": args}
            )
            state = state_payload()
            rendered = render_result(tool, raw)
            return (_format_status(state), *rendered)

        # ───────── Wire up events ─────────
        all_inputs = [
            tool_dropdown,
            bash_command, bash_timeout,
            read_path, read_offset, read_limit,
            write_path, write_content,
            edit_path, edit_old, edit_new, edit_replace_all,
            multi_edit_path, multi_edit_json,
            glob_pattern, glob_path,
            grep_pattern, grep_path, grep_include,
            ls_path, ls_ignore,
            todo_json,
        ]
        all_outputs = [
            status_md,
            output_status,
            output_view,
            raw_response,
            state_summary,
            state_json,
            history_table,
        ]

        run_btn.click(on_run, inputs=all_inputs, outputs=all_outputs)
        reset_btn.click(
            on_reset, inputs=[setup_input, verify_input], outputs=all_outputs
        )
        close_btn.click(on_close, outputs=all_outputs)

    return demo
