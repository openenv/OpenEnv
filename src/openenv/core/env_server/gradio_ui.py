# SPDX-License-Identifier: BSD-3-Clause

"""
Gradio-based web UI for OpenEnv environments.

Mounted at /web via gr.mount_gradio_app() from create_web_interface_app() when
ENABLE_WEB_INTERFACE is set. The page follows the loop an agent runs: reset,
take an action, read the result, with the same call in Python next to it.
"""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from .mcp_environment import MCPEnvironment
from .mcp_types import ListToolsAction
from .types import EnvironmentMetadata

_SECRET_NAMES = ("api_key", "token", "secret", "password")
_LONG_TEXT_NAMES = (
    "code",
    "message",
    "response",
    "text",
    "content",
    "answer",
    "prompt",
)


def get_gradio_display_title(
    metadata: Optional[EnvironmentMetadata],
    fallback: str = "OpenEnv Environment",
) -> str:
    """Return the title used for the Gradio app (browser tab and Blocks)."""
    name = metadata.name if metadata else fallback
    return f"OpenEnv Agentic Environment: {name}"


def _readme_body(readme: str) -> str:
    """The README without the Hugging Face Space front matter."""
    return re.sub(r"\A---\n.*?\n---\n", "", readme, flags=re.S)


def _description(metadata: Optional[EnvironmentMetadata]) -> str:
    """The metadata description, or the README's first paragraph when it's the default."""
    if not metadata:
        return ""
    if (
        not re.fullmatch(r"\S+ environment", metadata.description or "")
        or not metadata.readme_content
    ):
        return metadata.description
    for paragraph in re.split(r"\n\s*\n", _readme_body(metadata.readme_content)):
        text = paragraph.strip()
        if text and not text.startswith(("#", "<", "[!", "!", "|", "```", ">")):
            return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", " ".join(text.split()))
    return metadata.description


def _header_html(metadata: Optional[EnvironmentMetadata], kind: str) -> str:
    name = metadata.name if metadata else "environment"
    title = name.removesuffix("_env").replace("_", " ").title()
    return (
        f'<div class="oe-header"><span class="oe-eyebrow">{html.escape(name)} · {kind}</span>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(_description(metadata))}</p></div>"
    )


def _step_title(number: int, title: str, text: str) -> str:
    return f'<div class="oe-step"><h2>{number}. {title}</h2><p>{text}</p></div>'


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def _folded(value: Any) -> Any:
    """
    The value with each array of more than 64 numbers (a grid, a screen, a screenshot)
    shown as its shape, so the page stays light. Smaller boards and lists keep their values.
    """
    if isinstance(value, dict):
        return {k: _folded(v) for k, v in value.items()}
    if not isinstance(value, list):
        return value
    shape, first = [], value
    while isinstance(first, list) and first:
        shape.append(len(first))
        first = first[0]
    if isinstance(first, (int, float)) and math.prod(shape) > 64:
        return f"list · {'×'.join(map(str, shape))}"
    return [_folded(v) for v in value]


def _short(value: Any, limit: int = 160) -> str:
    value = _rounded(value)
    if isinstance(value, list) and len(value) > 20:
        text = f"{len(value)} items: {json.dumps(value[:8], default=str)[:-1]}, …]"
    else:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _error(obs: Dict[str, Any]) -> Optional[str]:
    """The error an observation reports: a failed tool call, or an env error in metadata."""
    error = obs.get("error") or (obs.get("metadata") or {}).get("error")
    if not error:
        return None
    return str(error.get("message", error) if isinstance(error, dict) else error)


def _output(obs: Dict[str, Any]) -> Any:
    """What a tool call returned."""
    result = obs.get("result")
    return result.get("data", result) if isinstance(result, dict) else result


def _result_html(data: Dict[str, Any], step_count: int) -> str:
    """The observation (a tool's output or its fields), then reward, done and step."""
    obs = data.get("observation", {}) or {}
    error = _error(obs)
    if error:
        body = (
            f'<pre class="oe-output oe-error">Error: {html.escape(error[:2000])}</pre>'
        )
    elif obs.get("result") is not None:
        output = _output(obs)
        text = (
            output
            if isinstance(output, str)
            else json.dumps(_rounded(output), indent=2, default=str)
        )
        body = f'<pre class="oe-output">{html.escape(text[:4000])}</pre>'
    else:
        rows = "".join(
            f'<div><span title="{html.escape(k)}">{html.escape(k)}</span>'
            f"<code>{html.escape(_short(v, 600 if isinstance(v, str) else 160))}</code></div>"
            for k, v in obs.items()
            if k not in ("metadata", "done", "reward", "result", "error")
            and v not in (None, "", [], {})
        )
        body = f'<div class="oe-fields">{rows}</div>' if rows else ""
    reward = data.get("reward")
    stats = "".join(
        f"<div><span>{label}</span><code>{html.escape(str(value))}</code></div>"
        for label, value in (
            ("reward", "–" if reward is None else _rounded(reward)),
            ("done", str(bool(data.get("done"))).lower()),
            ("step", step_count),
        )
    )
    return f'<div class="oe-result">{body}<div class="oe-stats">{stats}</div></div>'


def _error_html(message: str, step_count: int) -> str:
    return _result_html({"observation": {"error": message}}, step_count)


def _episode_html(entries: List[List[str]]) -> str:
    if not entries:
        items = '<p class="oe-muted">Nothing yet. Reset, then run a step.</p>'
    else:
        items = (
            "<ol>"
            + "".join(
                f"<li><span>{html.escape(badge)}</span><div><code>{html.escape(call)}</code>"
                f"<small>{html.escape(result)}</small></div></li>"
                for badge, call, result in entries
            )
            + "</ol>"
        )
    return f'<div class="oe-episode"><h2>Episode</h2>{items}</div>'


def _list_tools(env: Any) -> List[Any]:
    return list(env.step(ListToolsAction()).tools)


def _tool_label(tool: Any) -> str:
    """The tool's name and the first line of its description, if it has one."""
    first_line = (tool.description or "").strip().split("\n")[0]
    return f"{tool.name}: {first_line}" if first_line else tool.name


def _params(schema: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], bool]]:
    """
    The (name, schema, required) of each property of a JSON schema, with `$ref`s
    resolved and `X | None` unwrapped to `X`.
    """
    defs = schema.get("$defs", {})

    def resolve(prop: Dict[str, Any]) -> Dict[str, Any]:
        if "$ref" in prop:
            prop = {
                **defs.get(prop["$ref"].split("/")[-1], {}),
                **{k: v for k, v in prop.items() if k != "$ref"},
            }
        for key in ("anyOf", "oneOf", "allOf"):
            options = [o for o in prop.get(key, []) if o.get("type") != "null"]
            if options:
                merged = {k: v for k, v in prop.items() if k != key}
                return resolve({**resolve(options[0]), **merged})
        return prop

    required = set(schema.get("required", []))
    return [
        (name, resolve(prop), name in required)
        for name, prop in schema.get("properties", {}).items()
        if name != "metadata"
    ]


def _call_args(arguments: Dict[str, Any]) -> str:
    """Arguments as `name=value` for the episode log, with secrets masked and long values cut."""
    return ", ".join(
        f"{k}={'***' if k.lower().endswith(_SECRET_NAMES) else _short(json.dumps(v), 60)}"
        for k, v in arguments.items()
    )


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _param_input(
    name: str, schema: Dict[str, Any], required: bool
) -> Tuple[Any, Callable[[Any], Any]]:
    """
    A form input for one parameter, and a function that turns its value into
    the one to send (`None` to leave the parameter out).
    """
    kind = schema.get("type", "string")
    default = schema.get("default")
    is_json = kind in ("array", "object")
    label = f"{name} ({kind}{', JSON' if is_json else ''}){' · required' if required else ''}"
    info = (schema.get("description") or "").strip()[:200] or None

    if "enum" in schema:
        # A required choice starts on its first option, so the form can run as is.
        value = schema["enum"][0] if required else default
        return (
            gr.Dropdown(choices=schema["enum"], value=value, label=label, info=info),
            lambda v: None if _blank(v) else v,
        )
    if kind == "boolean":
        return gr.Checkbox(label=label, value=bool(default), info=info), bool
    if kind in ("integer", "number"):
        cast = int if kind == "integer" else float

        def number(value: Any) -> Any:
            if _blank(value):
                return None
            try:
                return cast(float(value))
            except ValueError:
                raise ValueError(f"{name} must be a number")

        # Gradio shows an empty number box as 0, so optional numbers without a default are text.
        if default is None and not required:
            return gr.Textbox(label=label, info=info, placeholder="optional"), number
        return gr.Number(
            label=label,
            value=default,
            info=info,
            precision=0 if kind == "integer" else None,
        ), number
    if is_json:

        def parse(value: Any) -> Any:
            if _blank(value):
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"{name} must be JSON: {e}")

        return (
            gr.Textbox(
                label=label,
                info=info,
                lines=3,
                value=None if default is None else json.dumps(default),
                placeholder="[ ... ]" if kind == "array" else "{ ... }",
            ),
            parse,
        )
    long_text = (
        any(n in name.lower() for n in _LONG_TEXT_NAMES)
        or schema.get("maxLength", 0) > 100
    )
    return (
        gr.Textbox(
            label=label,
            info=info,
            value=default if isinstance(default, str) else None,
            lines=4 if long_text else 1,
            type="password" if name.lower().endswith(_SECRET_NAMES) else "text",
        ),
        lambda v: None if _blank(v) else v,
    )


class _Form:
    """Inputs for a set of parameters, and the dict to send built from their values."""

    def __init__(self, params: List[Tuple[str, Dict[str, Any], bool]]):
        required = [p for p in params if p[2]]
        optional = [p for p in params if not p[2]]
        self.names: List[Tuple[str, bool]] = []
        self.inputs: List[Any] = []
        self.converters: List[Callable[[Any], Any]] = []
        self._add(required)
        if required and optional:
            with gr.Accordion("More fields", open=False, elem_classes="oe-raw"):
                self._add(optional)
        else:
            self._add(optional)

    def _add(self, params: List[Tuple[str, Dict[str, Any], bool]]) -> None:
        for name, schema, required in params:
            component, convert = _param_input(name, schema, required)
            self.names.append((name, required))
            self.inputs.append(component)
            self.converters.append(convert)

    def values(self, raw: List[Any]) -> Dict[str, Any]:
        """The values to send. Raises `ValueError` for a missing or invalid one."""
        values, missing = {}, []
        for (name, required), convert, value in zip(self.names, self.converters, raw):
            converted = convert(value)
            if converted is None:
                if required:
                    missing.append(name)
                continue
            values[name] = converted
        if missing:
            raise ValueError(f"Fill in {', '.join(missing)}.")
        return values


def build_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """
    Build a Gradio Blocks app for the OpenEnv web interface.

    Args:
        web_manager: WebInterfaceManager (reset/step_environment, get_state).
        action_fields: Field dicts from _extract_action_fields(action_cls). The form is
            built from the action's JSON schema, so this is kept for `gradio_builder`s.
        metadata: Environment metadata for README/name.
        is_chat_env: If True, a single message box; else a form from the action schema.
        title: App title (overridden by metadata.name when present; see get_gradio_display_title).
        quick_start_md: Optional Quick Start markdown (class names already replaced).

    Returns:
        gr.Blocks to mount with gr.mount_gradio_app(app, blocks, path="/web").
    """
    env = web_manager.env
    tools = _list_tools(env) if isinstance(env, MCPEnvironment) else []
    is_mcp = bool(tools)
    kind = "MCP environment" if is_mcp else "environment"

    def render(data: Dict[str, Any]):
        """The env's drawing and its one-click actions for this observation."""
        obs = {
            **(data.get("observation") or {}),
            "reward": data.get("reward"),
            "done": data.get("done"),
        }
        drawing = env.render_web(obs)
        actions = env.web_actions(obs) if not obs["done"] else []
        return (
            gr.update(value=drawing or "", visible=bool(drawing)),
            gr.update(
                choices=[(label, str(i)) for i, (label, _) in enumerate(actions)],
                value=None,
                visible=bool(actions),
            ),
            [action for _, action in actions],
        )

    def steps_in(entries: List[List[str]]) -> int:
        return sum(1 for badge, _, _ in entries if badge != "R")

    def unchanged(message: str, entries: List[List[str]]):
        """Show an error without touching the episode."""
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            _error_html(message, steps_in(entries)),
            gr.update(),
            gr.update(),
            entries,
        )

    async def reset_env(entries):
        try:
            data = await web_manager.reset_environment()
        except Exception as e:
            # Keep the current episode and the last step's result.
            (
                visual_update,
                quick_update,
                actions,
                result_html,
                raw,
                episode_html,
                kept,
            ) = unchanged(f"reset() failed: {e}", entries)
            return (
                visual_update,
                quick_update,
                actions,
                result_html,
                raw,
                episode_html,
                kept,
                gr.update(),
            )
        error = _error(data.get("observation", {}) or {})
        entries = [
            ["R", "reset()", f"error: {_short(error)}" if error else "new episode"]
        ]
        visual_update, quick_update, actions = render(data)
        shown = _folded(data)
        return (
            visual_update,
            quick_update,
            actions,
            _result_html(shown, 0),
            json.dumps(shown, indent=2, default=str, ensure_ascii=False),
            _episode_html(entries),
            entries,
            "",
        )

    async def run_step(action: Dict[str, Any], call: str, entries):
        if not entries:
            try:
                await web_manager.reset_environment()
            except Exception as e:
                return unchanged(f"reset() failed: {e}", entries)
            entries = [["R", "reset()", "new episode"]]
        try:
            data = await web_manager.step_environment(action)
        except Exception as e:
            entries = entries + [
                [str(steps_in(entries) + 1), call, f"error: {_short(str(e))}"]
            ]
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                _error_html(str(e), steps_in(entries)),
                gr.update(),
                _episode_html(entries),
                entries,
            )
        obs = data.get("observation", {}) or {}
        error = _error(obs)
        summary = (
            f"error: {error}"
            if error
            else _output(obs)
            if obs.get("result") is not None
            else f"reward {data.get('reward')}"
        )
        entries = entries + [[str(steps_in(entries) + 1), call, _short(summary)]]
        visual_update, quick_update, actions = render(data)
        shown = _folded(data)
        return (
            visual_update,
            quick_update,
            actions,
            _result_html(shown, steps_in(entries)),
            json.dumps(shown, indent=2, default=str, ensure_ascii=False),
            _episode_html(entries),
            entries,
        )

    with gr.Blocks(title=get_gradio_display_title(metadata, fallback=title)) as demo:
        entries_state = gr.State([])
        gr.HTML(_header_html(metadata, kind))
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                with gr.Column(elem_classes="oe-card"):
                    with gr.Row(elem_classes="oe-row", equal_height=True):
                        gr.HTML(
                            _step_title(
                                1,
                                "Start an episode",
                                "<code>reset()</code> returns the first observation.",
                            )
                        )
                        reset_btn = gr.Button(
                            "Reset", variant="secondary", scale=0, min_width=110
                        )
                    reset_result = gr.HTML()

                with gr.Column(elem_classes="oe-card"):
                    if is_mcp:
                        gr.HTML(
                            _step_title(
                                2,
                                "Call a tool",
                                "<code>step()</code> sends a tool call. Pick one:",
                            )
                        )
                    else:
                        gr.HTML(
                            _step_title(
                                2,
                                "Take an action",
                                "<code>step()</code> sends an action.",
                            )
                        )
                    with gr.Row(equal_height=False):
                        visual = gr.HTML(visible=False, elem_classes="oe-visual")
                        with gr.Column(min_width=260):
                            quick = gr.Radio(
                                choices=[],
                                visible=False,
                                label="Legal actions",
                                elem_classes="oe-actions",
                            )
                            if is_mcp:
                                tool_choice = gr.Radio(
                                    choices=[(_tool_label(t), t.name) for t in tools],
                                    value=tools[0].name,
                                    show_label=False,
                                    elem_classes="oe-tools",
                                )
                                forms, groups = {}, []
                                for i, tool in enumerate(tools):
                                    with gr.Column(visible=i == 0) as group:
                                        forms[tool.name] = _Form(
                                            _params(tool.input_schema)
                                        )
                                    groups.append(group)
                                tool_choice.change(
                                    lambda name: [
                                        gr.update(visible=t.name == name) for t in tools
                                    ],
                                    inputs=tool_choice,
                                    outputs=groups,
                                )
                                all_inputs = [
                                    i for f in forms.values() for i in f.inputs
                                ]

                                async def step_fn(entries, name, *values):
                                    start = 0
                                    for tool_name, form in forms.items():
                                        if tool_name == name:
                                            break
                                        start += len(form.inputs)
                                    form = forms[name]
                                    try:
                                        arguments = form.values(
                                            list(
                                                values[start : start + len(form.inputs)]
                                            )
                                        )
                                    except ValueError as e:
                                        return unchanged(str(e), entries)
                                    action = {
                                        "type": "call_tool",
                                        "tool_name": name,
                                        "arguments": arguments,
                                    }
                                    return await run_step(
                                        action,
                                        f"{name}({_call_args(arguments)})",
                                        entries,
                                    )

                                step_inputs = [entries_state, tool_choice, *all_inputs]
                            else:
                                if is_chat_env:
                                    params = [
                                        (
                                            "message",
                                            {
                                                "type": "string",
                                                "description": "Your message to the model.",
                                            },
                                            True,
                                        )
                                    ]
                                else:
                                    params = [
                                        p
                                        for p in _params(
                                            web_manager.action_cls.model_json_schema()
                                        )
                                        if p[0] != "type"
                                    ]
                                form = _Form(params)

                                async def step_fn(entries, *values):
                                    try:
                                        action = form.values(list(values))
                                    except ValueError as e:
                                        return unchanged(str(e), entries)
                                    return await run_step(
                                        action, f"step({_call_args(action)})", entries
                                    )

                                step_inputs = [entries_state, *form.inputs]

                            step_btn = gr.Button(
                                "Run step", variant="primary", elem_classes="oe-run"
                            )
                    result = gr.HTML()
                    with gr.Accordion(
                        "raw JSON", open=False, elem_classes="oe-raw oe-json"
                    ):
                        raw_json = gr.Code(
                            language="json", interactive=False, show_label=False
                        )

            with gr.Column(scale=2, min_width=320):
                with gr.Column(elem_classes="oe-card"):
                    episode = gr.HTML(_episode_html([]))
                    with gr.Accordion(
                        "State", open=False, elem_classes="oe-raw oe-json"
                    ):
                        state_json = gr.Code(
                            language="json", interactive=False, show_label=False
                        )
                if quick_start_md:
                    with gr.Column(elem_classes="oe-card oe-code"):
                        gr.HTML(
                            '<div class="oe-step"><h2>Same call in Python</h2></div>'
                        )
                        gr.Markdown(
                            quick_start_md.replace(
                                "### Connect to this environment\n\n", ""
                            )
                        )

        if metadata and metadata.readme_content:
            with gr.Accordion("README", open=False, elem_classes="oe-raw"):
                gr.Markdown(_readme_body(metadata.readme_content))

        quick_state = gr.State([])

        def show_state():
            try:
                return json.dumps(web_manager.get_state(), indent=2, default=str)
            except Exception:
                return ""

        async def quick_step(entries, actions, choice):
            action = actions[int(choice)]
            return await run_step(action, f"step({_call_args(action)})", entries)

        outputs = [visual, quick, quick_state, result, raw_json, episode, entries_state]
        reset_outputs = [
            visual,
            quick,
            quick_state,
            reset_result,
            raw_json,
            episode,
            entries_state,
            result,
        ]
        # One call to the environment at a time, so a slow reset can't overlap a step.
        busy = dict(concurrency_id="env", concurrency_limit=1, trigger_mode="once")
        reset_btn.click(
            reset_env, inputs=[entries_state], outputs=reset_outputs, **busy
        ).then(show_state, outputs=state_json)
        # Enter submits a one-line text field, Shift+Enter a multi-line one.
        gr.on(
            [step_btn.click]
            + [i.submit for i in step_inputs if isinstance(i, gr.Textbox)],
            step_fn,
            inputs=step_inputs,
            outputs=outputs,
            **busy,
        ).then(show_state, outputs=state_json)
        quick.input(
            quick_step,
            inputs=[entries_state, quick_state, quick],
            outputs=outputs,
            **busy,
        ).then(show_state, outputs=state_json)

    return demo
