# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Web UI for the τ²-bench environment: browse tasks, run a model, play as the agent, compare runs.

Each session gets its own `Tau2Environment`, so conversations never mix, and
keeps its own history of runs. The domain, the split and the simulated
customer's model can be changed from the page.
"""

import json
import os
import re
import tempfile
import time
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import litellm
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from tau2.environment.toolkit import get_tool_types
from tau2.registry import registry

from .task_view import (
    compare_html,
    explorer_html,
    result_html,
    run_label,
    runs_html,
    status_html,
    summary_html,
    task_html,
    timeline_html,
    title,
)
from .tau2_environment import (
    DEFAULT_MODEL,
    HF_ROUTER,
    Tau2Environment,
    without_end_tokens,
)

DOMAINS = ["airline", "retail", "telecom"]
SPLITS = ["test", "train", "base"]
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "zai-org/GLM-5.3",
    "Qwen/Qwen3.8-27B",
    "moonshotai/Kimi-K3",
]
AGENT_INSTRUCTIONS = (
    "\n\nTalk to the user only through the respond_to_user tool, one message at a "
    "time. Call done when the user's request is handled."
)
MAX_AGENT_TURNS = 60
MAX_COMPARED = 4
RUN_PLACEHOLDER = "Pick a model and press <b>Run</b> to watch it handle this customer."
PLAY_PLACEHOLDER = (
    "Press <b>Start</b> to meet your customer. You are the agent: reply below, look "
    "things up with the tools on the right, and follow the policy."
)
ASSETS = Path(__file__).parent / "assets"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def skeleton(schema: dict[str, Any]) -> Any:
    """A placeholder of the right type for one JSON-schema property."""
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return skeleton(schema["anyOf"][0])
    return {"integer": 0, "number": 0, "boolean": False, "array": [], "object": {}}.get(
        schema.get("type"), ""
    )


@lru_cache(maxsize=1)
def model_choices() -> list[tuple[str, str]]:
    """Chat models on Inference Providers that call tools, as `(label, id)`, widest-served first."""
    try:
        with urllib.request.urlopen(f"{HF_ROUTER}/models", timeout=20) as response:
            data = json.loads(response.read())["data"]
    except (OSError, ValueError, KeyError):
        return [(m, m) for m in FALLBACK_MODELS]
    models = []
    for model in data:
        providers = [
            p
            for p in model.get("providers") or []
            if p.get("status") == "live" and p.get("supports_tools")
        ]
        if not providers:
            continue
        prices = [p["pricing"] for p in providers if p.get("pricing")]
        cheapest = min(prices, key=lambda p: p.get("input", 0), default=None)
        price = (
            f" · ${cheapest['input']:g} / ${cheapest['output']:g}" if cheapest else ""
        )
        label = f"{model['id']} · {plural(len(providers), 'provider')}{price}"
        models.append((len(providers), model["id"], label))
    models.sort(key=lambda m: (m[1] != DEFAULT_MODEL, -m[0]))
    return [(label, model_id) for _, model_id, label in models] or [
        (m, m) for m in FALLBACK_MODELS
    ]


class Catalog:
    """What the UI shows about one domain and split: tasks, tools and policy."""

    def __init__(self, env: Tau2Environment):
        self.domain, self.split = env.domain, env.split
        self.tasks = {
            t.id: t for t in registry.get_tasks_loader(self.domain)(self.split)
        }
        self.choices = [
            (f"#{t.id} · {title(t)[:80]}", t.id) for t in self.tasks.values()
        ]
        self.first = self.choices[0][1]
        self.tools = {t.name: t for t in env.step(ListToolsAction()).tools}
        self.domain_tools = [
            n for n in self.tools if n not in ("respond_to_user", "done")
        ]
        reads = [
            n
            for n in self.domain_tools
            if n.startswith(("get_", "find_", "list_", "search_"))
        ]
        self.default_tool = (
            "get_user_details"
            if "get_user_details" in self.tools
            else (reads or self.domain_tools)[0]
        )
        domain_env = registry.get_env_constructor(self.domain)()
        self.tool_types = {
            n: t.value for n, t in get_tool_types(domain_env.tools).items()
        }
        # The policy's own headings would dwarf the panel it is shown in.
        self.policy = re.sub(
            r"^(#+) ",
            lambda m: "#" * (len(m.group(1)) + 3) + " ",
            domain_env.get_policy(),
            flags=re.M,
        )

    def page(self, task_id: str) -> str:
        return task_html(self.tasks[task_id], self.tool_types, self.domain)

    def summary(self, task_id: str) -> str:
        return summary_html(self.tasks[task_id], self.tool_types)

    def explorer(self) -> str:
        return explorer_html(list(self.tasks.values()), self.tool_types)

    def tool_help(self, name: str) -> tuple[str, str]:
        tool = self.tools[name]
        props = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])
        lines = [tool.description or ""]
        for key, prop in props.items():
            flag = "" if key in required else " *(optional)*"
            lines.append(
                f"- `{key}` ({prop.get('type', 'object')}){flag}: {prop.get('description', '')}"
            )
        example = {key: skeleton(props.get(key, {})) for key in required}
        return "\n".join(lines), json.dumps(example, indent=2)


class Episode:
    """A conversation in progress: the environment and the events the UI draws."""

    def __init__(self, env: Tau2Environment, task_id: str):
        self.env = env
        self.task_id = task_id
        # e.g. the customer has no credential, or its provider rejects it
        try:
            self.observation = env.reset(task_id=task_id)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        self.events = [
            {"kind": "customer", "text": self.observation.metadata["user_message"]}
        ]
        self.started = time.time()
        self.messages = 0
        self.tool_calls = 0

    def act(self, name: str, arguments: dict):
        """Run one agent action and record what it shows in the conversation."""
        observation = self.env.step(CallToolAction(tool_name=name, arguments=arguments))
        if observation.error is not None:
            result = f"Error: {observation.error.message}"
        else:
            result = observation.result.content[0].text
        if name == "respond_to_user":
            self.messages += 1
            self.events.append({"kind": "agent", "text": arguments.get("message", "")})
            # A failed reply is shown as a failed call, not as the customer.
            if observation.error is not None:
                self.events.append(
                    {
                        "kind": "tool",
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
            elif reply := without_end_tokens(result):
                self.events.append({"kind": "customer", "text": reply})
        elif name != "done":
            self.tool_calls += 1
            self.events.append(
                {"kind": "tool", "name": name, "arguments": arguments, "result": result}
            )
        return observation, result

    def stats(self) -> str:
        return (
            f"{plural(self.messages, 'message')} to the customer · "
            f"{plural(self.tool_calls, 'tool call')} · "
            f"{time.time() - self.started:.0f}s"
        )

    def side(self, observation, running: bool = False) -> str:
        if observation is not None and observation.done:
            if not observation.metadata["reward_info"]:  # the simulated customer failed
                return status_html(f"Ended without a score · {self.stats()}", False)
            return result_html(
                observation.reward, observation.metadata["reward_info"], self.stats()
            )
        return status_html(self.stats(), running)

    def record(self, run_id: str, model: str, catalog: Catalog, observation) -> dict:
        """The finished run, as kept in the session's history and downloaded as JSON."""
        return {
            "id": run_id,
            "finished": time.strftime("%H:%M:%S"),
            "domain": catalog.domain,
            "split": catalog.split,
            "task_id": self.task_id,
            "title": title(catalog.tasks[self.task_id]),
            "model": model,
            "user_model": self.env.user_llm,
            "reward": observation.reward,
            "reward_info": observation.metadata["reward_info"],
            "stats": self.stats(),
            "events": self.events,
        }


def parse_arguments(raw: str | None) -> dict | None:
    """A tool call's arguments, or `None` when the model sent invalid JSON or not an object."""
    try:
        arguments = json.loads(raw or "{}")
    except ValueError:
        return None
    return arguments if isinstance(arguments, dict) else None


def build_ui(make_env: Callable[..., Tau2Environment]) -> Callable[..., gr.Blocks]:
    """
    Build the UI's Gradio builder.

    Args:
        make_env (`Callable[..., Tau2Environment]`):
            Environment factory. Called with no arguments for the server's defaults,
            or with `domain`, `split`, `user_provider`, `user_model` and `hf_token` to
            override them. The UI's conversations always run on Inference Providers.
    """

    @lru_cache(maxsize=None)
    def catalog(domain: str, split: str) -> Catalog:
        return Catalog(make_env(domain=domain, split=split))

    css = (ASSETS / "tau2.css").read_text()
    explorer_js = (ASSETS / "explorer.js").read_text()

    def header(cat: Catalog) -> str:
        return (
            '<div class="t2-head"><div><p>A customer-service agent solves the request of an '
            "LLM-simulated customer with the domain's tools while following its policy. "
            "τ²-bench scores the conversation when it ends.</p></div>"
            '<div class="t2-stats">'
            f'<div class="t2-stat"><b>{len(cat.tasks)}</b><span>tasks</span></div>'
            f'<div class="t2-stat"><b>{len(cat.domain_tools)}</b><span>tools</span></div>'
            "</div></div>"
        )

    def switch(domain: str, split: str):
        """Everything that depends on the domain and split, for the new ones."""
        cat = catalog(domain, split)
        help_text, example = cat.tool_help(cat.default_tool)
        tasks = gr.Dropdown(choices=cat.choices, value=cat.first)
        return (
            header(cat),
            cat.explorer(),
            cat.page(cat.first),
            cat.first,
            tasks,
            tasks,
            cat.summary(cat.first),
            cat.summary(cat.first),
            gr.Dropdown(choices=cat.domain_tools, value=cat.default_tool),
            help_text,
            example,
            cat.policy,
            cat.policy,
        )

    def hf_token(oauth_token: gr.OAuthToken | None) -> str:
        """The visitor's token on a Space, where signing in is required, else `HF_TOKEN`."""
        if os.environ.get("SPACE_ID"):
            # The Space's own secret, if any, stays for the API: visitors pay for their runs.
            token = oauth_token and oauth_token.token
        else:
            token = os.environ.get("HF_TOKEN")
        if not token:
            raise gr.Error(
                "Sign in with Hugging Face (or set HF_TOKEN) to run conversations. The "
                "models run on your Inference Providers credits."
            )
        return token

    # Run a model -------------------------------------------------------------
    def run_model(
        domain,
        split,
        user_model,
        task_id,
        model,
        runs,
        oauth_token: gr.OAuthToken | None,
    ):
        cat = catalog(domain, split)
        token = hf_token(oauth_token)
        episode = Episode(
            make_env(
                domain=domain,
                split=split,
                user_provider="hf",
                user_model=user_model,
                hf_token=token,
            ),
            task_id,
        )
        try:
            yield (
                timeline_html(episode.events, RUN_PLACEHOLDER),
                episode.side(None, True),
                runs,
            )
            schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": n,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for n, t in cat.tools.items()
            ]
            messages = [
                {
                    "role": "system",
                    "content": episode.observation.metadata["policy"]
                    + AGENT_INSTRUCTIONS,
                },
                {"role": "user", "content": episode.events[0]["text"]},
            ]
            for _ in range(MAX_AGENT_TURNS):
                reply = (
                    litellm.completion(
                        model=f"openai/{model}",
                        api_base=HF_ROUTER,
                        api_key=token,
                        messages=messages,
                        tools=schemas,
                        temperature=0.0,
                        num_retries=5,
                        timeout=120,
                    )
                    .choices[0]
                    .message
                )
                if reply.tool_calls:
                    messages.append(reply.model_dump(exclude_none=True))
                    calls = [
                        (c.id, c.function.name, parse_arguments(c.function.arguments))
                        for c in reply.tool_calls
                    ]
                else:  # plain text is a message to the user
                    call_id = f"text{len(messages)}"
                    args = {"message": reply.content or ""}
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "respond_to_user",
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ],
                        }
                    )
                    calls = [(call_id, "respond_to_user", args)]
                for call_id, name, args in calls:
                    if args is None:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": "Error: the arguments are not valid JSON.",
                            }
                        )
                        continue
                    observation, result = episode.act(name, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": result}
                    )
                    if observation.done:
                        # Only scored runs go to the history.
                        if observation.metadata["reward_info"]:
                            runs = runs + [
                                episode.record(
                                    f"run-{len(runs) + 1}", model, cat, observation
                                )
                            ]
                        yield (
                            timeline_html(episode.events, RUN_PLACEHOLDER),
                            episode.side(observation),
                            runs,
                        )
                        return
                    yield (
                        timeline_html(episode.events, RUN_PLACEHOLDER),
                        episode.side(observation, running=True),
                        runs,
                    )
            yield (
                timeline_html(episode.events, RUN_PLACEHOLDER),
                status_html(
                    f"Stopped after {MAX_AGENT_TURNS} agent turns · {episode.stats()}",
                    False,
                ),
                runs,
            )
        finally:  # also when Stop cancels the run
            episode.env.close()

    # Play as the agent -------------------------------------------------------
    def play_start(
        previous, domain, split, user_model, task_id, oauth_token: gr.OAuthToken | None
    ):
        token = hf_token(oauth_token)
        episode = Episode(
            make_env(
                domain=domain,
                split=split,
                user_provider="hf",
                user_model=user_model,
                hf_token=token,
            ),
            task_id,
        )
        close_episode(previous)  # only once the new one has started
        return (
            episode,
            timeline_html(episode.events, PLAY_PLACEHOLDER),
            episode.side(None),
        )

    def close_episode(episode: Episode | None) -> None:
        if episode is not None:
            episode.env.close()

    def play_act(episode: Episode, name: str, arguments: dict):
        if episode is None or episode.env.state.done:
            return gr.update(), gr.update()
        observation, _ = episode.act(name, arguments)
        return timeline_html(episode.events, PLAY_PLACEHOLDER), episode.side(
            observation
        )

    def play_say(episode: Episode, message: str):
        if not message.strip():
            return gr.update(), gr.update(), message
        view, side = play_act(episode, "respond_to_user", {"message": message})
        return view, side, ""

    def play_tool(episode: Episode, name: str, arguments: str):
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            return gr.update(), status_html("The arguments are not valid JSON.", False)
        return play_act(episode, name, args)

    # Runs --------------------------------------------------------------------
    def runs_changed(runs):
        choices = [(run_label(r), r["id"]) for r in reversed(runs)]
        return runs_html(runs), gr.CheckboxGroup(choices=choices, value=[])

    def open_run(runs, evt: gr.SelectData):
        run = next(r for r in runs if r["id"] == evt.value)
        # A directory per download, so visitors' runs with the same name don't collide.
        path = (
            Path(tempfile.mkdtemp(prefix="tau2-run-"))
            / f"tau2-{run['domain']}-{run['task_id']}-{run['id']}.json"
        )
        path.write_text(json.dumps(run, indent=2, default=str))
        return (
            result_html(run["reward"], run["reward_info"], run["stats"])
            + timeline_html(run["events"], ""),
            gr.DownloadButton(value=str(path), visible=True),
        )

    def compare(runs, picked):
        chosen = [r for r in runs if r["id"] in picked][:MAX_COMPARED]
        return compare_html(chosen)

    html = {"apply_default_css": False, "padding": False}

    def builder(
        web_manager, action_fields, metadata, is_chat_env, display_title, quick_start_md
    ):
        default = make_env()
        home = catalog(default.domain, default.split)
        help_text, example = home.tool_help(home.default_tool)
        models = model_choices()
        default_user_model = (
            default.user_llm.split("/", 1)[-1]
            if default.user_provider == "hf"
            else DEFAULT_MODEL
        )

        with gr.Blocks() as demo, gr.Column(elem_classes="t2"):
            gr.HTML(f"<style>{css}</style>", elem_classes="t2-style", **html)
            head = gr.HTML(header(home), **html)
            with gr.Row(elem_classes="t2-controls"):
                domain = gr.Dropdown(
                    DOMAINS, value=home.domain, label="Domain", scale=1
                )
                split = gr.Dropdown(SPLITS, value=home.split, label="Split", scale=1)
                user_model = gr.Dropdown(
                    models,
                    value=default_user_model,
                    allow_custom_value=True,
                    label="Simulated customer · Hugging Face Inference Providers",
                    scale=3,
                )
                # Off a Space, Gradio's sign-in needs a local HF login, so it is shown on Spaces only.
                if os.environ.get("SPACE_ID"):
                    gr.LoginButton(scale=1)
            selected = gr.State(home.first)
            runs = gr.State([])
            with gr.Tabs() as tabs:
                with gr.Tab("Tasks", id="tasks"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=280):
                            explorer = gr.HTML(
                                home.explorer(), js_on_load=explorer_js, **html
                            )
                        with gr.Column(scale=3):
                            with gr.Row(elem_classes="t2-actions"):
                                to_run = gr.Button("Run a model", variant="primary")
                                to_play = gr.Button("Play as the agent")
                            detail = gr.HTML(home.page(home.first), **html)
                            with gr.Accordion(
                                "Domain policy (what the agent is told)", open=False
                            ):
                                tasks_policy = gr.Markdown(home.policy)

                with gr.Tab("Run a model", id="run"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            run_view = gr.HTML(
                                timeline_html([], RUN_PLACEHOLDER),
                                autoscroll=True,
                                max_height=760,
                                **html,
                            )
                        with gr.Column(scale=2, elem_classes="t2-card"):
                            gr.HTML('<p class="t2-card-title">Run a model</p>', **html)
                            run_summary = gr.HTML(home.summary(home.first), **html)
                            run_task = gr.Dropdown(
                                home.choices, value=home.first, label="Task"
                            )
                            run_agent = gr.Dropdown(
                                models,
                                value=models[0][1],
                                allow_custom_value=True,
                                label="Agent model · Hugging Face Inference Providers",
                            )
                            with gr.Row():
                                run_go = gr.Button("Run", variant="primary")
                                run_stop = gr.Button("Stop")
                            run_side = gr.HTML("", **html)

                with gr.Tab("Play as the agent", id="play"):
                    episode = gr.State(None, delete_callback=close_episode)
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            play_view = gr.HTML(
                                timeline_html([], PLAY_PLACEHOLDER),
                                autoscroll=True,
                                max_height=620,
                                **html,
                            )
                            with gr.Row(elem_classes="t2-composer"):
                                play_message = gr.Textbox(
                                    placeholder="Reply to the customer…",
                                    show_label=False,
                                    scale=5,
                                )
                                play_send = gr.Button(
                                    "Send", variant="primary", scale=1
                                )
                            play_side = gr.HTML("", **html)
                        with gr.Column(scale=2):
                            with gr.Column(elem_classes="t2-card"):
                                play_summary = gr.HTML(home.summary(home.first), **html)
                                play_task = gr.Dropdown(
                                    home.choices, value=home.first, label="Task"
                                )
                                play_go = gr.Button("Start", variant="primary")
                            with gr.Column(elem_classes="t2-card"):
                                gr.HTML(
                                    '<p class="t2-card-title">Call a tool</p>', **html
                                )
                                tool_name = gr.Dropdown(
                                    home.domain_tools,
                                    value=home.default_tool,
                                    show_label=False,
                                )
                                tool_doc = gr.Markdown(help_text)
                                tool_args = gr.Code(
                                    example, language="json", label="Arguments"
                                )
                                with gr.Row():
                                    tool_call = gr.Button("Call tool")
                                    play_end = gr.Button("End conversation")
                            with gr.Accordion("Policy", open=False):
                                play_policy = gr.Markdown(home.policy)

                with gr.Tab("Runs", id="runs"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=280):
                            runs_list = gr.HTML(
                                runs_html([]), js_on_load=explorer_js, **html
                            )
                        with gr.Column(scale=3):
                            run_detail = gr.HTML(
                                '<div class="t2-timeline t2-empty">Open a run to see it here.</div>',
                                **html,
                            )
                            download = gr.DownloadButton(
                                "Download run (JSON)", visible=False
                            )
                    with gr.Accordion("Compare runs", open=False):
                        picked = gr.CheckboxGroup(
                            [], label="Runs to compare (two to four)"
                        )
                        compare_go = gr.Button("Compare")
                        compared = gr.HTML(compare_html([]), **html)

            # Wiring ----------------------------------------------------------
            for control in (domain, split):
                control.change(
                    switch,
                    [domain, split],
                    [
                        head,
                        explorer,
                        detail,
                        selected,
                        run_task,
                        play_task,
                        run_summary,
                        play_summary,
                        tool_name,
                        tool_doc,
                        tool_args,
                        tasks_policy,
                        play_policy,
                    ],
                )

            def open_task(domain, split, evt: gr.SelectData):
                page = catalog(domain, split).page(evt.value)
                return evt.value, page, evt.value, evt.value

            explorer.select(
                open_task, [domain, split], [selected, detail, run_task, play_task]
            )
            to_run.click(
                lambda t: (t, gr.Tabs(selected="run")), selected, [run_task, tabs]
            )
            to_play.click(
                lambda t: (t, gr.Tabs(selected="play")), selected, [play_task, tabs]
            )

            running = run_go.click(
                run_model,
                [domain, split, user_model, run_task, run_agent, runs],
                [run_view, run_side, runs],
            )
            run_stop.click(None, cancels=[running])
            run_task.change(
                lambda d, s, t: catalog(d, s).summary(t),
                [domain, split, run_task],
                run_summary,
            )

            play_go.click(
                play_start,
                [episode, domain, split, user_model, play_task],
                [episode, play_view, play_side],
            )
            play_task.change(
                lambda d, s, t: catalog(d, s).summary(t),
                [domain, split, play_task],
                play_summary,
            )
            for trigger in (play_send.click, play_message.submit):
                trigger(
                    play_say,
                    [episode, play_message],
                    [play_view, play_side, play_message],
                )
            tool_name.change(
                lambda d, s, n: catalog(d, s).tool_help(n),
                [domain, split, tool_name],
                [tool_doc, tool_args],
            )
            tool_call.click(
                play_tool, [episode, tool_name, tool_args], [play_view, play_side]
            )
            play_end.click(
                lambda ep: play_act(ep, "done", {}), episode, [play_view, play_side]
            )

            runs.change(runs_changed, runs, [runs_list, picked])
            runs_list.select(open_run, runs, [run_detail, download])
            compare_go.click(compare, [runs, picked], compared)
        return demo

    return builder
