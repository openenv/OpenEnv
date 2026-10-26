"""
Custom Gradio UI for the AWM environment.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any

import gradio as gr

from openenv.core.env_server.serialization import serialize_observation

from .data_loader import AWMDataLoader
from .prompts import DEFAULT_SYSTEM_PROMPT
from .web_agent import AwmAgent
from .config import DEFAULT_REWARD_CONFIG


# Keep in sync with DEFAULT_REWARD_CONFIG in config.py.
_DEFAULT_REWARD_JSON = json.dumps(
    DEFAULT_REWARD_CONFIG, indent=2
)


def _format_obs_md(payload: dict | None) -> str:
    if not payload:
        return "*No observation yet.*"
    obs = payload.get("observation") if isinstance(payload, dict) else None
    if obs is None:
        obs = payload
    reward = payload.get("reward") if isinstance(payload, dict) else None
    done = payload.get("done") if isinstance(payload, dict) else None

    lines: list[str] = []
    if reward is not None:
        lines.append(f"**reward**: `{reward}`")
    if done is not None:
        lines.append(f"**done**: `{done}`")
    if isinstance(obs, dict):
        for key in (
            "reward_type",
            "scenario",
            "task",
            "task_idx",
            "num_tools",
            "tool_name",
            "error",
            "warning",
        ):
            v = obs.get(key)
            if v is not None and v != "":
                lines.append(f"**{key}**: `{v}`")
        if obs.get("tool_result") is not None:
            tr = obs["tool_result"]
            tr_text = (
                tr if isinstance(tr, str) else json.dumps(tr, indent=2, default=str)
            )
            if len(tr_text) > 2000:
                tr_text = tr_text[:2000] + "\n... (truncated)"
            lines.append("\n**tool_result:**")
            lines.append(f"```\n{tr_text}\n```")
        if obs.get("verify_result"):
            vr = obs["verify_result"]
            vr_text = json.dumps(vr, indent=2, default=str)
            if len(vr_text) > 2000:
                vr_text = vr_text[:2000] + "\n... (truncated)"
            lines.append("\n**verify_result:**")
            lines.append(f"```json\n{vr_text}\n```")
        if obs.get("trajectory_path"):
            lines.append(f"\n**trajectory_path**: `{obs['trajectory_path']}`")
    return "\n\n".join(lines) if lines else "*Empty observation.*"


def _make_args_template(input_schema: dict | None) -> str:
    if not input_schema or not isinstance(input_schema, dict):
        return "{}"
    props = input_schema.get("properties") or {}
    template: dict[str, Any] = {}
    for name, info in props.items():
        ty = (info or {}).get("type", "string")
        template[name] = {
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
        }.get(ty, None)
    return json.dumps(template, indent=2)


def build_awm_gradio_app(
    web_manager: Any,
    action_fields: list[dict] | None = None,
    metadata: Any = None,
    is_chat_env: bool = False,
    title: str = "AWM Environment",
    quick_start_md: str | None = None,
) -> gr.Blocks:
    data_loader = AWMDataLoader(cache_dir=os.environ.get("AWM_DATA_DIR"))

    readme_md = ""
    if metadata is not None and getattr(metadata, "readme_content", None):
        readme_md = metadata.readme_content

    # openenv 0.2.3 added a ``reset_kwargs`` parameter to
    # ``WebInterfaceManager.reset_environment``. PyPI's 0.2.1 takes no args,
    # which silently drops scenario/task_idx — fall back to calling env.reset
    # directly and replicate the episode-state updates the manager would do.
    _reset_env_supports_kwargs = (
        len(
            [
                p
                for p in inspect.signature(
                    web_manager.reset_environment
                ).parameters.values()
                if p.name != "self"
            ]
        )
        > 0
    )

    async def _safe_reset(reset_kwargs: dict[str, Any]) -> dict[str, Any]:
        if _reset_env_supports_kwargs:
            return await web_manager.reset_environment(reset_kwargs)

        env = web_manager.env
        params = inspect.signature(env.reset).parameters
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        valid = {k: v for k, v in reset_kwargs.items() if has_var_kw or k in params}
        loop = asyncio.get_event_loop()
        observation = await loop.run_in_executor(None, lambda: env.reset(**valid))
        serialized = serialize_observation(observation)

        es = web_manager.episode_state
        es.episode_id = env.state.episode_id
        es.step_count = 0
        es.current_observation = serialized["observation"]
        es.action_logs = []
        es.is_reset = True
        try:
            await web_manager._send_state_update()
        except Exception:
            pass
        return serialized

    async def _do_reset(
        scenario: str,
        task_idx: int,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        reward_config_json: str,
    ):
        if not scenario:
            return (
                "❌ Pick a scenario first.",
                "",
                "*Reset failed.*",
                "{}",
                gr.update(choices=[], value=None),
                "{}",
            )

        reset_kwargs: dict[str, Any] = {
            "scenario": scenario,
            "task_idx": int(task_idx),
        }
        if llm_base_url:
            reset_kwargs["llm_base_url"] = llm_base_url
        if llm_api_key:
            reset_kwargs["llm_api_key"] = llm_api_key
        if llm_model:
            reset_kwargs["llm_model"] = llm_model
        if reward_config_json and reward_config_json.strip():
            try:
                reset_kwargs["reward_config"] = json.loads(reward_config_json)
            except json.JSONDecodeError as e:
                return (
                    f"❌ Invalid reward_config JSON: {e}",
                    "",
                    "*Reset failed.*",
                    "{}",
                    gr.update(choices=[], value=None),
                    "{}",
                )

        try:
            result = await _safe_reset(reset_kwargs)
        except Exception as e:
            return (
                f"❌ Reset error: {e}",
                "",
                f"*Reset failed: {e}*",
                "{}",
                gr.update(choices=[], value=None),
                "{}",
            )

        obs = result.get("observation", {}) or {}
        rt = obs.get("reward_type")
        ok = rt in ("reset_ok", "reset_warning")
        status = "✅ Reset OK." if ok else f"❌ Reset failed: {obs.get('error') or rt}"
        task_md = f"**Task** (`{scenario}`, idx={task_idx}):\n\n{obs.get('task') or '*(no task description)*'}"

        tool_names: list[str] = []
        tool_lookup: dict[str, dict] = {}
        if ok:
            try:
                tools_result = await web_manager.step_environment(
                    {"type": "list_tools"}
                )
                tools = (tools_result.get("observation", {}) or {}).get("tools") or []
                for t in tools:
                    if isinstance(t, dict):
                        n = t.get("name", "")
                        tool_names.append(n)
                        tool_lookup[n] = t
                    else:
                        n = getattr(t, "name", "")
                        tool_names.append(n)
                        tool_lookup[n] = {
                            "name": n,
                            "description": getattr(t, "description", ""),
                            "input_schema": getattr(t, "input_schema", {}),
                        }
            except Exception as e:
                status += f" (list_tools warning: {e})"

        tool_choice = gr.update(
            choices=tool_names,
            value=(tool_names[0] if tool_names else None),
        )
        return (
            status,
            task_md,
            _format_obs_md(result),
            json.dumps(result, indent=2, default=str),
            tool_choice,
            json.dumps(tool_lookup),
        )

    async def _refresh_scenarios():
        try:
            scens = data_loader.list_scenarios()
            names = sorted(s["name"] for s in scens)
            return gr.update(
                choices=names,
                value=(names[0] if names else None),
            ), f"Loaded {len(names)} scenarios."
        except Exception as e:
            return gr.update(choices=[]), f"❌ Failed to load scenarios: {e}"

    async def _on_tool_change(tool_name: str, tool_lookup_json: str):
        try:
            lookup = json.loads(tool_lookup_json or "{}")
        except json.JSONDecodeError:
            lookup = {}
        if not tool_name or tool_name not in lookup:
            return "{}", ""
        meta = lookup[tool_name]
        schema = meta.get("input_schema") or meta.get("inputSchema") or {}
        return _make_args_template(schema), meta.get("description", "")

    async def _do_call_tool(tool_name: str, args_json: str):
        if not tool_name:
            return "*Pick a tool first.*", "{}"
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"❌ Invalid args JSON: {e}", "{}"
        try:
            result = await web_manager.step_environment(
                {"type": "call_tool", "tool_name": tool_name, "arguments": args}
            )
        except Exception as e:
            return f"❌ step error: {e}", "{}"
        return _format_obs_md(result), json.dumps(result, indent=2, default=str)

    async def _do_list_tools():
        try:
            result = await web_manager.step_environment({"type": "list_tools"})
        except Exception as e:
            return f"❌ {e}", "{}"
        return _format_obs_md(result), json.dumps(result, indent=2, default=str)

    async def _do_verify(verifier_mode: Any, final_answer: str):
        if isinstance(verifier_mode, dict):
            verifier_mode = next(iter(verifier_mode.keys()), "code")
        verifier_mode = str(verifier_mode).strip().lower() or "code"
        if verifier_mode not in ("code", "sql"):
            verifier_mode = "code"
        args: dict[str, Any] = {"verifier_mode": verifier_mode}
        if final_answer:
            args["final_answer"] = final_answer
        try:
            result = await web_manager.step_environment(
                {"type": "call_tool", "tool_name": "verify", "arguments": args}
            )
        except Exception as e:
            return f"❌ verify error: {e}", "{}"
        return _format_obs_md(result), json.dumps(result, indent=2, default=str)

    async def _do_done(keep_session: bool):
        try:
            result = await web_manager.step_environment(
                {
                    "type": "call_tool",
                    "tool_name": "done",
                    "arguments": {"keep_session": bool(keep_session)},
                }
            )
        except Exception as e:
            return f"❌ done error: {e}", "{}", None
        obs = result.get("observation", {}) or {}
        traj_path = obs.get("trajectory_path")
        return (
            _format_obs_md(result),
            json.dumps(result, indent=2, default=str),
            traj_path if traj_path and os.path.exists(traj_path) else None,
        )

    async def _do_list_scenarios_via_tool():
        try:
            result = await web_manager.step_environment(
                {
                    "type": "call_tool",
                    "tool_name": "__list_scenarios__",
                    "arguments": {},
                }
            )
        except Exception as e:
            return f"❌ {e}", "{}"
        return _format_obs_md(result), json.dumps(result, indent=2, default=str)

    agent_state: dict[str, AwmAgent | None] = {"agent": None, "stop": False}

    async def _do_run_agent(
        scenario: str,
        task_idx: int,
        verifier_mode: Any,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        system_prompt: str,
        max_iter: int,
        temperature: float,
        max_tokens: int,
        auto_verify: bool,
        auto_done: bool,
    ):
        if isinstance(verifier_mode, dict):
            verifier_mode = next(iter(verifier_mode.keys()), "code")
        verifier_mode = str(verifier_mode).strip().lower() or "code"
        if verifier_mode not in ("code", "sql"):
            verifier_mode = "code"
        if not scenario:
            yield "❌ Pick a scenario first.", None
            return
        if not (llm_base_url and llm_api_key and llm_model):
            yield (
                "❌ LLM config required for Agent mode (base_url + api_key + model).",
                None,
            )
            return

        try:
            reset_result = await _safe_reset(
                {
                    "scenario": scenario,
                    "task_idx": int(task_idx),
                    "llm_base_url": llm_base_url,
                    "llm_api_key": llm_api_key,
                    "llm_model": llm_model,
                }
            )
        except Exception as e:
            yield f"❌ Reset failed: {e}", None
            return
        obs = reset_result.get("observation", {}) or {}
        if obs.get("reward_type") not in ("reset_ok", "reset_warning"):
            yield (
                f"❌ Reset returned reward_type={obs.get('reward_type')}, error={obs.get('error')}",
                None,
            )
            return
        task_text = obs.get("task") or ""

        agent = AwmAgent(
            web_manager=web_manager,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            max_iterations=int(max_iter),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        agent_state["agent"] = agent
        agent_state["stop"] = False

        log_lines: list[str] = [
            "### Agent run started",
            f"**scenario**: `{scenario}` &nbsp;&nbsp; **task_idx**: `{task_idx}`",
            f"**task**: {task_text}",
            "",
        ]
        # Two outputs: (log_markdown, trajectory_file_path_or_None)
        yield "\n".join(log_lines), None

        traj_path: str | None = None
        async for ev in agent.run(
            task=task_text,
            verifier_mode=verifier_mode,
            auto_verify=auto_verify,
            auto_done=auto_done,
        ):
            if agent_state["stop"]:
                agent.request_stop()

            if ev.kind == "info":
                log_lines.append(f"_ℹ️ {ev.text}_")
            elif ev.kind == "llm_response":
                step = ev.payload.get("step", "?")
                log_lines.append(f"\n**[Step {step}] LLM response:**")
                log_lines.append(f"```\n{ev.text[:4000]}\n```")
            elif ev.kind == "tool_call":
                log_lines.append(f"→ **tool_call**: {ev.text}")
            elif ev.kind == "tool_result":
                log_lines.append("← **tool_result:**")
                log_lines.append(f"```\n{ev.text[:1500]}\n```")
            elif ev.kind == "verify":
                log_lines.append(f"\n🧪 **Verify**: {ev.text}")
                if ev.payload.get("verify_result"):
                    vr = json.dumps(ev.payload["verify_result"], indent=2, default=str)[
                        :1500
                    ]
                    log_lines.append(f"```json\n{vr}\n```")
            elif ev.kind == "done":
                log_lines.append(f"\n🏁 **Done**: {ev.text}")
                # Capture the trajectory path for download
                p = ev.payload.get("trajectory_path")
                if p and os.path.exists(p):
                    traj_path = p
            elif ev.kind == "error":
                log_lines.append(f"\n❌ **Error**: {ev.text}")

            yield "\n\n".join(log_lines), traj_path

        log_lines.append("\n_Agent run finished._")
        yield "\n\n".join(log_lines), traj_path

    def _do_stop_agent():
        agent_state["stop"] = True
        a = agent_state["agent"]
        if a is not None:
            a.request_stop()
        return "🛑 Stop requested. The agent will exit before its next iteration."

    async def _load_trajectory_from_state():
        try:
            logs = web_manager.episode_state.action_logs
        except Exception:
            return [], None
        rows: list[list[Any]] = []
        for i, log in enumerate(logs, 1):
            action = getattr(log, "action", {}) or {}
            obs = getattr(log, "observation", {}) or {}
            tool_name = action.get("tool_name") if isinstance(action, dict) else ""
            atype = action.get("type") if isinstance(action, dict) else ""
            rt = obs.get("reward_type") if isinstance(obs, dict) else ""
            preview = ""
            if isinstance(obs, dict):
                tr = obs.get("tool_result")
                if isinstance(tr, str):
                    preview = tr[:200]
                elif tr is not None:
                    preview = json.dumps(tr, default=str)[:200]
                elif obs.get("error"):
                    preview = f"ERROR: {obs['error']}"[:200]
            rows.append(
                [
                    i,
                    atype or "",
                    tool_name or "",
                    rt or "",
                    getattr(log, "reward", None),
                    preview,
                ]
            )
        return rows, None

    with gr.Blocks(title=f"AWM — {title}") as blocks:
        tools_state = gr.State("{}")

        gr.Markdown("# 🤖 Agent World Model — Web Console")
        gr.Markdown(
            "Pick a scenario, set LLM credentials (only needed for SQL verifier "
            "or Agent mode), then explore via Human or Agent mode."
        )

        with gr.Group():
            gr.Markdown("## ⚙️ Setup")
            with gr.Row():
                scenario_dd = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Scenario",
                    info="1,000 scenarios — click 'Load' first",
                    elem_id="awm_scenario_dd",
                    interactive=True,
                    allow_custom_value=False,
                )
                load_scen_btn = gr.Button(
                    "🔄 Load scenarios", scale=0, elem_id="awm_load_scen"
                )
                task_idx_slider = gr.Slider(
                    minimum=0,
                    maximum=9,
                    step=1,
                    value=0,
                    label="Task idx (0-9)",
                    elem_id="awm_task_idx",
                )
                verifier_mode_radio = gr.Textbox(
                    value="code",
                    label="Verifier mode (code or sql)",
                    elem_id="awm_verifier_mode",
                    info="Type 'code' or 'sql'. SQL mode requires LLM config above.",
                )
            with gr.Accordion(
                "LLM config (for SQL verifier and Agent mode)", open=False
            ):
                llm_base_url_in = gr.Textbox(
                    label="LLM base_url",
                    placeholder="https://...",
                    value="",
                    elem_id="awm_llm_base_url",
                )
                llm_api_key_in = gr.Textbox(
                    label="LLM api_key",
                    type="password",
                    value="",
                    elem_id="awm_llm_api_key",
                )
                llm_model_in = gr.Textbox(
                    label="LLM model",
                    placeholder="gpt-4.1, gpt-5, ...",
                    value="",
                    elem_id="awm_llm_model",
                )
            with gr.Accordion("Reward config (advanced)", open=False):
                reward_cfg_in = gr.Code(
                    language="json",
                    value=_DEFAULT_REWARD_JSON,
                    label="reward_config (JSON)",
                    elem_id="awm_reward_cfg",
                )
            with gr.Row():
                reset_btn = gr.Button(
                    "🔄 Reset", variant="primary", elem_id="awm_reset_btn"
                )
                status_box = gr.Markdown("Status: *idle*", elem_id="awm_status_box")
            task_md = gr.Markdown("*No task loaded yet.*", elem_id="awm_task_md")

        with gr.Tabs():
            with gr.Tab("👤 Human Mode"):
                with gr.Row():
                    list_tools_btn = gr.Button(
                        "📋 List Tools", elem_id="awm_human_list_tools"
                    )
                    list_scenarios_btn = gr.Button(
                        "🌐 List Scenarios", elem_id="awm_human_list_scenarios"
                    )
                with gr.Row():
                    tool_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Tool",
                        elem_id="awm_human_tool_dd",
                        interactive=True,
                        allow_custom_value=False,
                    )
                tool_desc_md = gr.Markdown("", elem_id="awm_human_tool_desc")
                tool_args_in = gr.Code(
                    language="json",
                    value="{}",
                    label="Tool arguments (JSON)",
                    elem_id="awm_human_tool_args",
                )
                with gr.Row():
                    call_tool_btn = gr.Button(
                        "▶️ Call Tool", variant="primary", elem_id="awm_human_call_tool"
                    )
                with gr.Group():
                    gr.Markdown("### Episode controls")
                    final_answer_in = gr.Textbox(
                        label="Final answer (optional, used in code-mode verify)",
                        value="",
                        elem_id="awm_human_final_answer",
                    )
                    with gr.Row():
                        verify_btn = gr.Button("🧪 Verify", elem_id="awm_human_verify")
                        keep_session_cb = gr.Checkbox(
                            value=True,
                            label="keep_session on done",
                            elem_id="awm_human_keep_session",
                        )
                        done_btn = gr.Button(
                            "🏁 Done", variant="stop", elem_id="awm_human_done"
                        )
                gr.Markdown("### Latest observation")
                obs_md = gr.Markdown("*No action yet.*", elem_id="awm_human_obs_md")
                with gr.Accordion("Raw JSON", open=False):
                    obs_json = gr.Code(
                        language="json",
                        value="{}",
                        elem_id="awm_human_obs_json",
                    )
                trajectory_file_dl = gr.File(
                    label="trajectory.json (after Done)",
                    interactive=False,
                    elem_id="awm_human_traj_dl",
                )

            with gr.Tab("🤖 Agent Mode"):
                gr.Markdown(
                    "Drives an LLM agent through the env. Reset is done"
                    " automatically using the scenario/task_idx selected above."
                )
                system_prompt_in = gr.Textbox(
                    label="System prompt",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=8,
                    max_lines=20,
                    elem_id="awm_agent_system_prompt",
                )
                with gr.Row():
                    max_iter_slider = gr.Slider(
                        minimum=1,
                        maximum=30,
                        step=1,
                        value=10,
                        label="Max iterations",
                        elem_id="awm_agent_max_iter",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Temperature",
                        elem_id="awm_agent_temperature",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=2048,
                        label="Max tokens / call",
                        elem_id="awm_agent_max_tokens",
                    )
                with gr.Row():
                    auto_verify_cb = gr.Checkbox(
                        value=True,
                        label="Auto verify at end",
                        elem_id="awm_agent_auto_verify",
                    )
                    auto_done_cb = gr.Checkbox(
                        value=True,
                        label="Auto done at end (keep_session=True)",
                        elem_id="awm_agent_auto_done",
                    )
                with gr.Row():
                    start_agent_btn = gr.Button(
                        "▶️ Start Agent", variant="primary", elem_id="awm_agent_start"
                    )
                    stop_agent_btn = gr.Button(
                        "⏹ Stop", variant="stop", elem_id="awm_agent_stop"
                    )
                stop_status = gr.Markdown("", elem_id="awm_agent_stop_status")
                agent_log_md = gr.Markdown(
                    "_Agent log will appear here._", elem_id="awm_agent_log_md"
                )
                agent_traj_dl = gr.File(
                    label="trajectory.json (auto-populated when agent finishes)",
                    interactive=False,
                    elem_id="awm_agent_traj_dl",
                )

            with gr.Tab("📜 Trajectory"):
                gr.Markdown(
                    "Step-by-step history of the current episode. "
                    "Click **Refresh** after actions to update."
                )
                refresh_traj_btn = gr.Button("🔄 Refresh", elem_id="awm_traj_refresh")
                traj_table = gr.Dataframe(
                    headers=["#", "type", "tool", "reward_type", "reward", "preview"],
                    datatype=["number", "str", "str", "str", "number", "str"],
                    interactive=False,
                    elem_id="awm_traj_table",
                )

            if readme_md:
                with gr.Tab("📖 README"):
                    gr.Markdown(readme_md)

        load_scen_btn.click(
            _refresh_scenarios, inputs=None, outputs=[scenario_dd, status_box]
        )
        reset_btn.click(
            _do_reset,
            inputs=[
                scenario_dd,
                task_idx_slider,
                llm_base_url_in,
                llm_api_key_in,
                llm_model_in,
                reward_cfg_in,
            ],
            outputs=[status_box, task_md, obs_md, obs_json, tool_dd, tools_state],
        )
        list_tools_btn.click(_do_list_tools, inputs=None, outputs=[obs_md, obs_json])
        list_scenarios_btn.click(
            _do_list_scenarios_via_tool, inputs=None, outputs=[obs_md, obs_json]
        )
        tool_dd.change(
            _on_tool_change,
            inputs=[tool_dd, tools_state],
            outputs=[tool_args_in, tool_desc_md],
        )
        call_tool_btn.click(
            _do_call_tool,
            inputs=[tool_dd, tool_args_in],
            outputs=[obs_md, obs_json],
        )
        verify_btn.click(
            _do_verify,
            inputs=[verifier_mode_radio, final_answer_in],
            outputs=[obs_md, obs_json],
        )
        done_btn.click(
            _do_done,
            inputs=[keep_session_cb],
            outputs=[obs_md, obs_json, trajectory_file_dl],
        )
        start_agent_btn.click(
            _do_run_agent,
            inputs=[
                scenario_dd,
                task_idx_slider,
                verifier_mode_radio,
                llm_base_url_in,
                llm_api_key_in,
                llm_model_in,
                system_prompt_in,
                max_iter_slider,
                temperature_slider,
                max_tokens_slider,
                auto_verify_cb,
                auto_done_cb,
            ],
            outputs=[agent_log_md, agent_traj_dl],
        )
        stop_agent_btn.click(_do_stop_agent, inputs=None, outputs=[stop_status])
        refresh_traj_btn.click(
            _load_trajectory_from_state,
            inputs=None,
            outputs=[traj_table, trajectory_file_dl],
        )

    return blocks
