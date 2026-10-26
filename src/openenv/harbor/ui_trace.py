"""How a rollout reads in the UI: its trajectory as a timeline of events, and its result.

A coding agent works in a loop: the model says something, acts, and gets something back. The capture
records that as conversations of messages in the agent's own dialect, which is accurate but hard to
read: a tool call is a JSON blob, a terminal is a user message, and some agents (terminus-2) write
their actions as a JSON object in plain text. So before rendering, this module turns a conversation
into **events**, one line each, in the order they happened:

  * the prompt the harness sent (folded: it is long and the same on every rollout of a task);
  * the model's thinking, when the endpoint returns it;
  * each **action**, paired with what came back: a native tool call (chat-completions or Anthropic
    shape) with its result, or a batch of commands an agent wrote as structured text with the
    terminal output that followed;
  * what the model said, and its final answer.

Every line opens to the full input and output. Everything here is model or task output, so every
string is escaped before it becomes HTML, and markdown is rendered with raw HTML disabled.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from .ui_icons import icon

_MD: Any = None


def _markdown(text: str) -> str:
    """Render text written by a model or a task. Raw HTML in it is escaped, never rendered, and so
    are images: one pointing at someone's server would report every reader of the page to it."""
    global _MD
    if _MD is None:
        from markdown_it import MarkdownIt

        _MD = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
    return _MD.render(text or "")


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _clip(text: Any, limit: int = 400) -> str:
    """Escape and shorten a value for display, keeping the head where the meaning usually is."""
    body = text if isinstance(text, str) else json.dumps(text, default=str)
    body = body.strip()
    return html.escape(body[:limit]) + ("…" if len(body) > limit else "")


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


# ── reading messages, whatever the dialect ───────────────────────────────────────────────────────


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls on a message, normalised across the dialects.

    Chat-completions puts them in `tool_calls`; Anthropic puts them in the content block list as
    `tool_use`. Reading only the former shows claude-code as a stream of text with no actions.
    """
    out: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name") or call.get("name")
        if name:
            out.append(
                {
                    "id": call.get("id"),
                    "name": str(name),
                    "arguments": function.get("arguments", call.get("arguments", "")),
                }
            )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name")
            ):
                out.append(
                    {
                        "id": block.get("id"),
                        "name": str(block["name"]),
                        "arguments": block.get("input", ""),
                    }
                )
    return out


def _message_text(message: dict[str, Any]) -> str:
    """Readable text of a message, without its tool-call, tool-result and thinking blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            str(b["text"])
            for b in content
            if isinstance(b, dict)
            and b.get("type") not in ("tool_use", "tool_result", "thinking")
            and b.get("text")
        )
    return ""


def _results(message: dict[str, Any]) -> list[dict[str, Any]]:
    """What came back from tools, in either the chat-completions or the Anthropic shape."""
    if message.get("role") == "tool":
        content = message.get("content")
        text = (
            _message_text(message)
            if isinstance(content, (str, list))
            else ""
            if content is None
            else json.dumps(content, default=str)
        )
        return [
            {
                "id": message.get("tool_call_id"),
                "text": text,
                "error": False,
                "name": message.get("name"),
            }
        ]
    content = message.get("content")
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            body = block.get("content")
            if isinstance(body, list):
                body = "\n".join(
                    str(b.get("text", "")) for b in body if isinstance(b, dict)
                )
            out.append(
                {
                    "id": block.get("tool_use_id"),
                    "text": str(body if body is not None else ""),
                    "error": bool(block.get("is_error")),
                }
            )
    return out


def _thinking(message: dict[str, Any]) -> str:
    """The model's reasoning, where the endpoint returned it."""
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = message.get("content")
    if isinstance(content, list):
        return "\n\n".join(
            str(b.get("thinking") or b.get("text") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ).strip()
    return ""


def _structured(text: str) -> dict[str, Any] | None:
    """A reply that is one JSON object, which is how terminus-2 and its kin write their actions."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        value = json.loads(s)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _commands(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Keystroke or command batches in a structured reply."""
    out = []
    for c in value.get("commands") or []:
        if isinstance(c, dict):
            keys = c.get("keystrokes", c.get("command", c.get("cmd")))
            if isinstance(keys, str):
                out.append({"keys": keys, "duration": c.get("duration")})
        elif isinstance(c, str):
            out.append({"keys": c, "duration": None})
    return out


def split_steps(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A conversation as its opening (everything before the first reply) and one step per reply.

    Each step is the assistant message and everything that came back before the next one.
    """
    opening: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "assistant":
            steps.append({"assistant": m, "back": []})
        elif steps:
            steps[-1]["back"].append(m)
        else:
            opening.append(m)
    return opening, steps


def count_actions(result: dict[str, Any]) -> int:
    """Tool calls, or for an agent that writes its actions as text, the commands it wrote."""
    calls = sum(len(t.get("tool_calls") or []) for t in result.get("turns") or [])
    if calls:
        return calls
    n = 0
    for convo in result.get("conversations") or []:
        if convo.get("role", "agent") != "agent":
            continue
        for m in convo.get("messages") or []:
            if m.get("role") == "assistant":
                s = _structured(_message_text(m))
                n += len(_commands(s)) if s else 0
    return n


# ── rendering pieces ─────────────────────────────────────────────────────────────────────────────

# `user@host:/dir# ` or `$ ` at the start of a terminal line: drawn apart from the command after it.
_PROMPT = re.compile(r"^([\w.-]+@[\w.-]+:[^\s]*\s?[#$]\s|\$\s)")
_TERMINAL_PREFIX = re.compile(
    r"^\s*(New Terminal Output|Terminal output|Current terminal state)\s*:\s*\n?", re.I
)
_SHOW_LINES = 40
_MAX_CHARS = 30000

_COMMAND_KEYS = ("command", "cmd", "script", "code")
_PATH_KEYS = ("file_path", "filePath", "path", "filename", "file", "notebook_path")
_QUERY_KEYS = ("pattern", "query", "url", "glob", "regex", "description", "prompt")
_BODY_KEYS = ("content", "patch", "diff", "text", "new_source")
_OLD_KEYS = ("old_string", "oldString", "old_str")
_NEW_KEYS = ("new_string", "newString", "new_str")

# The icon for a tool, by what it does. Agents name the same tool differently (Bash, bash,
# run_shell_command, execute), so the match is on words in the name.
_TOOL_ICONS = (
    (("bash", "shell", "exec", "terminal", "command", "run"), "terminal"),
    (("edit", "replace", "patch", "str_replace", "apply"), "pencil"),
    (("write", "create"), "filePlus"),
    (("read", "view", "cat", "open"), "file"),
    (("grep", "search", "find", "glob", "rg"), "search"),
    (("ls", "list", "dir", "tree"), "folder"),
    (("todo", "plan", "task"), "list"),
    (("fetch", "web", "http", "browse", "url"), "globe"),
)


def _tool_icon(name: str) -> str:
    low = name.lower()
    for words, ic in _TOOL_ICONS:
        if any(w in low for w in words):
            return ic
    return "tool"


def _term_lines(lines: list[str]) -> str:
    out = []
    for line in lines:
        m = _PROMPT.match(line)
        out.append(
            f'<span class="ps">{_e(m.group(0))}</span>{_e(line[m.end() :])}'
            if m
            else _e(line)
        )
    return "\n".join(out)


def _terminal(text: str, *, error: bool = False) -> str:
    """Output as a terminal shows it. Long output folds after forty lines; very long is cut."""
    cut = len(text) > _MAX_CHARS
    lines = text[:_MAX_CHARS].rstrip("\n").split("\n")
    cls = "term err" if error else "term"
    out = f'<pre class="{cls}">{_term_lines(lines[:_SHOW_LINES])}</pre>'
    if len(lines) > _SHOW_LINES:
        rest = len(lines) - _SHOW_LINES
        out += (
            f'<details class="term-more"><summary>{_plural(rest, "more line")}</summary>'
            f'<pre class="{cls}">{_term_lines(lines[_SHOW_LINES:])}</pre></details>'
        )
    if cut:
        out += f'<div class="hb-fine">Cut at {_MAX_CHARS:,} characters; the full text is in the result JSON.</div>'
    return out


def _capped(text: Any) -> tuple[str, str]:
    """`(text, note)`: text past `_MAX_CHARS` is left out of the page, which is redrawn every two
    seconds while a run is live; the result JSON keeps all of it."""
    text = str(text)
    if len(text) <= _MAX_CHARS:
        return text, ""
    return text[:_MAX_CHARS], (
        f'<div class="hb-fine">Cut at {_MAX_CHARS:,} of {len(text):,} characters; '
        "the full text is in the result JSON.</div>"
    )


def _code(text: str, cls: str = "hb-code") -> str:
    text, note = _capped(text)
    lines = text.rstrip("\n").split("\n")
    out = f'<pre class="{cls}">{_e(chr(10).join(lines[:_SHOW_LINES]))}</pre>'
    if len(lines) > _SHOW_LINES:
        out += (
            f'<details class="term-more"><summary>{_plural(len(lines) - _SHOW_LINES, "more line")}</summary>'
            f'<pre class="{cls}">{_e(chr(10).join(lines[_SHOW_LINES:]))}</pre></details>'
        )
    return out + note


def _diff(old: str, new: str) -> str:
    rows = [f'<span class="d">{_e(line)}</span>' for line in old.split("\n")]
    rows += [f'<span class="a">{_e(line)}</span>' for line in new.split("\n")]
    return f'<pre class="diff">{"".join(rows)}</pre>'


def _parse_args(arguments: Any) -> Any:
    if isinstance(arguments, str):
        s = arguments.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except ValueError:
                return arguments
    return arguments


def _first(args: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    return next(
        (args[k] for k in keys if isinstance(args.get(k), str) and args[k].strip()),
        None,
    )


def _peek(args: Any) -> str:
    """What a tool call is about, on one line: its command, its file, or its query."""
    if isinstance(args, dict):
        value = (
            _first(args, _COMMAND_KEYS)
            or _first(args, _PATH_KEYS)
            or _first(args, _QUERY_KEYS)
        )
        if value is None:
            value = next(
                (v for v in args.values() if isinstance(v, str) and v.strip()), None
            )
        if value is None:
            value = json.dumps(args, ensure_ascii=False, default=str)
    else:
        value = str(args or "")
    return " ".join(str(value).split())[:200]


def _input_html(args: Any) -> str:
    """A tool call's input, as what it does: a command, an edit, a file body, or its arguments."""
    if not isinstance(args, dict):
        return (
            _code(
                args
                if isinstance(args, str)
                else json.dumps(args, indent=2, default=str)
            )
            if args not in (None, "", {})
            else ""
        )
    used: set[str] = set()
    parts = []
    command = _first(args, _COMMAND_KEYS)
    if command:
        used |= set(_COMMAND_KEYS)
        parts.append(
            f'<pre class="cmd"><span class="ps">$ </span>{_e(command.rstrip())}</pre>'
        )
    # The agents disagree on spelling: old_string (claude-code), oldString (opencode), old_str.
    old = next((args[k] for k in _OLD_KEYS if isinstance(args.get(k), str)), None)
    new = next((args[k] for k in _NEW_KEYS if isinstance(args.get(k), str)), None)
    if isinstance(old, str) and isinstance(new, str):
        used |= set(_OLD_KEYS) | set(_NEW_KEYS)
        parts.append(_diff(old, new))
    else:
        for key in _BODY_KEYS:
            if isinstance(args.get(key), str) and args[key]:
                used.add(key)
                parts.append(_code(args[key]))
                break
    rest = {k: v for k, v in args.items() if k not in used}
    if rest:
        if all(
            not isinstance(v, (dict, list)) and len(str(v)) <= 160
            for v in rest.values()
        ):
            parts.append(
                '<dl class="args">'
                + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rest.items())
                + "</dl>"
            )
        else:
            parts.append(
                _code(json.dumps(rest, indent=2, ensure_ascii=False, default=str))
            )
    return "".join(parts)


def _output_html(text: str, *, error: bool = False, label: str = "Output") -> str:
    text = text or ""
    if not text.strip():
        return f'<div><h5>{_e(label)}</h5><div class="hb-fine">no output</div></div>'
    lines = text.count("\n") + 1
    return f"<div><h5>{_e(label)}<span>{'error · ' if error else ''}{_plural(lines, 'line')}</span></h5>{_terminal(text, error=error)}</div>"


def _ev(kind: str, ic: str, body: str, ts: str = "") -> str:
    return f'<li class="ev {kind}"><span class="ei">{icon(ic, 15)}</span><div class="eb">{body}</div><span class="ts">{_e(ts)}</span></li>'


def _fold(summary: str, inner: str, *, open_: bool = False) -> str:
    return f"<details{' open' if open_ else ''}><summary>{icon('chevronRight', 13, 'chev')}{summary}</summary>{inner}</details>"


def _tool_event(call: dict[str, Any], result: dict[str, Any] | None, ts: str) -> str:
    args = _parse_args(call.get("arguments"))
    error = bool(result and result.get("error"))
    peek = _peek(args)
    lines = ""
    if result is not None and (result.get("text") or "").strip():
        lines = f'<span class="ms">{"error" if error else _plural(result["text"].count(chr(10)) + 1, "line")}</span>'
    elif result is None:
        lines = '<span class="ms">no result</span>'
    summary = (
        f"<b>{_e(call['name'])}</b>"
        + (f"<code>{_e(peek)}</code>" if peek else "")
        + lines
    )
    io = _input_html(args)
    inner = '<div class="io">' + (f"<div><h5>Input</h5>{io}</div>" if io else "")
    if result is not None:
        inner += _output_html(result.get("text") or "", error=error)
    inner += "</div>"
    return _ev(
        f"ev-tool{' bad' if error else ''}",
        _tool_icon(call["name"]),
        _fold(summary, inner),
        ts,
    )


def _keys_event(commands: list[dict[str, Any]], output: str | None, ts: str) -> str:
    """A batch of keystrokes typed into a terminal, with the terminal as it looked afterwards."""
    first = commands[0]["keys"].strip().split("\n")[0] if commands else ""
    more = f'<span class="ms">+{len(commands) - 1}</span>' if len(commands) > 1 else ""
    summary = f"<b>terminal</b><code>{_e(first[:200])}</code>{more}"
    cmds = "".join(
        f'<span class="ps">$ </span>{_e(c["keys"].rstrip(chr(10)))}'
        + (
            f'  <span class="ps"># waits {_e(c["duration"])}s</span>'
            if c.get("duration") not in (None, "", 0.1, 0)
            else ""
        )
        + "\n"
        for c in commands
    )
    inner = f'<div class="io"><div><h5>Keystrokes<span>{_plural(len(commands), "command")}</span></h5><pre class="cmd">{cmds.rstrip()}</pre></div>'
    if output is not None:
        inner += _output_html(output, label="Terminal")
    inner += "</div>"
    return _ev("ev-tool", "terminal", _fold(summary, inner), ts)


def _thought_event(label: str, text: str, ts: str) -> str:
    peek = " ".join(text.split())[:220]
    summary = f'<b>{_e(label)}</b><span class="peek">{_e(peek)}</span>'
    return _ev(
        "ev-think",
        "brain",
        _fold(summary, f'<div class="thought">{_e(text.strip())}</div>'),
        ts,
    )


def _prompt_events(opening: list[dict[str, Any]]) -> list[str]:
    """The system prompt and the task: long and the same on every rollout of a task, so folded."""
    out = []
    first_user = True
    for m in opening:
        text = _message_text(m)
        if not text.strip():
            continue
        role = m.get("role")
        if role == "system":
            label = "System prompt"
        else:
            label = (
                "Instructions to the model" if first_user else "Message to the model"
            )
            first_user = False
        summary = f'<b>{_e(label)}</b><span class="peek">{_plural(len(text), "character")}</span>'
        out.append(
            _ev(
                "ev-prompt",
                "list" if role == "system" else "file",
                _fold(summary, f'<div class="prompt hb-prose">{_markdown(text)}</div>'),
            )
        )
    return out


def _step_events(
    n: int, step: dict[str, Any], *, final: bool, structured_run: bool
) -> list[str]:
    """One model call as events: its thinking, its actions with their results, what it said."""
    a = step["assistant"]
    ts = f"#{n}"
    out: list[str] = []
    thinking = _thinking(a)
    if thinking:
        out.append(_thought_event("Thinking", thinking, ts))
        ts = ""
    calls = _tool_calls(a)
    text = _message_text(a)
    structured = _structured(text) if not calls else None
    # Results by id where the dialect carries one, else in order.
    results: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for m in step["back"]:
        got = _results(m)
        if not got:
            others.append(m)
            continue
        results.extend(got)
        # Anthropic puts text beside the results in the same message (claude-code's reminders).
        beside = _message_text(m)
        if beside.strip():
            others.append({"role": m.get("role"), "content": beside})
    by_id = {r["id"]: r for r in results if r.get("id")}
    unmatched = [
        r
        for r in results
        if not r.get("id") or r["id"] not in {c.get("id") for c in calls}
    ]

    if structured is not None:
        said = "\n\n".join(
            f"{label}: {structured[key].strip()}"
            for key, label in (
                ("analysis", "Analysis"),
                ("plan", "Plan"),
                ("thought", "Thought"),
                ("reasoning", "Reasoning"),
            )
            if isinstance(structured.get(key), str) and structured[key].strip()
        )
        if said:
            out.append(_thought_event("Plan", said, ts))
            ts = ""
        terminal = None
        for m in others:
            body = _message_text(m)
            if body.strip():
                match = _TERMINAL_PREFIX.match(body)
                terminal = (terminal + "\n" if terminal else "") + (
                    body[match.end() :] if match else body
                )
        others = []
        cmds = _commands(structured)
        if cmds:
            out.append(_keys_event(cmds, terminal, ts))
            ts = ""
        elif terminal is not None:
            out.append(
                _ev(
                    "ev-tool",
                    "terminal",
                    _fold(
                        "<b>terminal</b>",
                        f'<div class="io">{_output_html(terminal, label="Terminal")}</div>',
                    ),
                    ts,
                )
            )
            ts = ""
        extra = {
            k: v
            for k, v in structured.items()
            if k
            not in (
                "analysis",
                "plan",
                "thought",
                "reasoning",
                "commands",
                "task_complete",
            )
        }
        if extra:
            out.append(
                _ev(
                    "ev-think",
                    "code",
                    _fold(
                        "<b>Other fields</b>",
                        f'<div class="io">{_code(json.dumps(extra, indent=2, ensure_ascii=False, default=str))}</div>',
                    ),
                    ts,
                )
            )
            ts = ""
        if structured.get("task_complete") is True:
            out.append(_ev("ev-done", "check", "Marked the task complete.", ts))
            ts = ""
    elif text.strip():
        cls = "ev-text ev-final" if final else "ev-text"
        out.append(
            _ev(
                cls,
                "check" if final else "message",
                f'<div class="msg hb-prose">{_markdown(_capped(text)[0])}</div>{_capped(text)[1]}',
                ts,
            )
        )
        ts = ""

    queue = list(unmatched)
    for c in calls:
        r = by_id.get(c.get("id")) if c.get("id") else None
        if r is None and queue:
            r = queue.pop(0)
        out.append(_tool_event(c, r, ts))
        ts = ""
    for m in others:
        body = _message_text(m)
        if not body.strip():
            continue
        match = _TERMINAL_PREFIX.match(body)
        if match or structured_run:
            out.append(
                _ev(
                    "ev-tool",
                    "terminal",
                    _fold(
                        "<b>terminal</b>",
                        f'<div class="io">{_output_html(body[match.end() :] if match else body, label="Terminal")}</div>',
                    ),
                    ts,
                )
            )
        else:
            peek = " ".join(body.split())[:200]
            out.append(
                _ev(
                    "ev-prompt",
                    "user",
                    _fold(
                        f'<b>{_e((m.get("role") or "user").capitalize())}</b><span class="peek">{_e(peek)}</span>',
                        f'<div class="prompt hb-prose">{_markdown(body)}</div>',
                    ),
                    ts,
                )
            )
        ts = ""
    if not out:
        out.append(
            _ev("ev-prompt", "message", '<span class="hb-fine">Empty reply.</span>', ts)
        )
    return out


def trajectory_html(messages: list[dict[str, Any]], *, live: bool = False) -> str:
    """A conversation as a timeline: the prompt, then every model call's events in order.

    Args:
        messages (`list[dict]`):
            The conversation, in the agent's dialect.
        live (`bool`, *optional*, defaults to `False`):
            The rollout is still running: the last reply is not a final answer, and a row says so.

    Returns:
        `str`: HTML.
    """
    opening, steps = split_steps(messages)
    structured_run = any(
        _structured(_message_text(s["assistant"])) is not None for s in steps[:3]
    )
    items = _prompt_events(opening)
    for i, step in enumerate(steps, 1):
        final = (
            not live
            and i == len(steps)
            and not _tool_calls(step["assistant"])
            and not step["back"]
        )
        items += _step_events(i, step, final=final, structured_run=structured_run)
    if live:
        items.append(
            _ev(
                "ev-live",
                "clock",
                '<span class="hb-pulse"></span>Waiting for the next model call…',
            )
        )
    return f'<ol class="tl">{"".join(items)}</ol>'


def _conversation_label(convo: dict[str, Any], seen_agents: int, n_agents: int) -> str:
    role = convo.get("role", "agent")
    if role == "agent":
        return (
            "main conversation"
            if n_agents == 1
            else f"conversation {seen_agents} of {n_agents}"
        )
    return {"auxiliary": "auxiliary call", "discarded": "discarded branch"}.get(
        role, str(role)
    )


def _conversation_html(r: dict[str, Any]) -> str:
    """Every conversation of a finished rollout, the agent's first.

    Each root is a separate conversation. An auxiliary one (a next-speaker check, a summariser) is
    labelled as such and folded, so it is not mistaken for the agent working on the task.
    """
    conversations = [c for c in r.get("conversations") or [] if c.get("messages")]
    if not conversations:
        return '<div class="tl-empty">No conversation was captured.</div>'
    if len(conversations) == 1:
        return trajectory_html(conversations[0]["messages"])
    n_agents = sum(1 for c in conversations if c.get("role", "agent") == "agent")
    blocks, seen = [], 0
    for convo in conversations:
        agent = convo.get("role", "agent") == "agent"
        seen += agent
        label = _conversation_label(convo, seen, n_agents)
        steps = sum(1 for m in convo["messages"] if m.get("role") == "assistant")
        head = f"<b>{_e(label)}</b><span>{_plural(steps, 'model call')}</span>"
        body = trajectory_html(convo["messages"])
        if agent:
            blocks.append(
                f'<section class="tl-conv"><div class="tl-conv-h">{head}</div>{body}</section>'
            )
        else:
            blocks.append(
                f'<details class="tl-conv"><summary class="tl-conv-h">{icon("chevronRight", 13, "chev")}{head}</summary>{body}</details>'
            )
    return "".join(blocks)


def _transcript_html(session: Any) -> str:
    """The trajectory of a rollout still running, from its newest captured call.

    That call's `request_messages` already holds the whole conversation the harness assembled, tool
    results included, so it plus the reply is the trajectory so far, with nothing rebuilt from deltas.
    The newest call that offered tools, when any did: an agent's side calls (opencode's title
    generator) offer none, and are newer than the agent's own call while they run. This is the rule
    `export._assign_roles` uses to tell the two apart.
    """
    nodes = sorted(session.graph.nodes(), key=lambda n: n.index)
    if not nodes:
        return ""
    working = [n for n in nodes if n.n_tools]
    latest = (working or nodes)[-1]
    messages = list(latest.request_messages or [])
    if latest.response_message:
        messages.append({**latest.response_message, "role": "assistant"})
    return trajectory_html(messages, live=True)


# ── the result panels ────────────────────────────────────────────────────────────────────────────

_LEVELS = (("FATAL", "bad", "x"), ("WARN", "warn", "alert"), ("INFO", "", "info"))


def _findings_html(findings: list[str]) -> str:
    """Findings, worst first: a FATAL means unusable, a WARN means read it before training on it."""
    if not findings:
        return ""
    buckets: dict[str, list[str]] = {"FATAL": [], "WARN": [], "INFO": []}
    for raw in findings:
        level = (
            "FATAL"
            if raw.startswith("[FATAL")
            else "WARN"
            if raw.startswith("[WARN")
            else "INFO"
        )
        buckets[level].append(
            raw.split("]", 1)[-1].strip() if raw.startswith("[") else raw
        )
    rows = [
        f'<li class="{cls}"><span class="mk">{icon(ic, 14)}</span><b>{html.escape(item[:400])}</b><span class="sc">{level}</span></li>'
        for level, cls, ic in _LEVELS
        for item in buckets[level]
    ]
    return f'<ul class="rp-list">{"".join(rows)}</ul>'


_ATIF = {
    "match": "Harbor's own trajectory agrees with the capture, call for call.",
    "MISMATCH": "Harbor's trajectory and the capture disagree. Read the findings before using this rollout.",
    "none": "This agent writes no trajectory file, so there is nothing to cross-check.",
}


def _verdict(r: dict[str, Any]) -> tuple[str, str, str, str]:
    """`(status tone, status word, big number, its class)` for a finished rollout."""
    reward = r.get("reward")
    if not r.get("ok"):
        return "bad", "Failed", "–", "none"
    if reward is None and len(r.get("rewards") or {}) > 1:
        return (
            "warn",
            "Graded",
            "–",
            "none",
        )  # every reward is listed below; none is the headline
    if reward is None:
        return "warn", "Not graded", "–", "none"
    value = float(reward)
    if value > 0:
        return "ok", "Solved", f"{value:.2f}", "full" if value >= 0.999 else "part"
    return "bad", "Not solved", f"{value:.2f}", "zero"


def _result_html(r: dict[str, Any]) -> str:
    """The result, and what to check before trusting it, as side panels.

    Figures that mean nothing for this kind of rollout (token counts from an endpoint that returns no
    token ids) are left out rather than shown as zeros, because a zero reads as a measurement.
    """
    tone, word, big, cls = _verdict(r)
    reward = r.get("reward")
    if not r.get("ok"):
        caption = _e(r.get("exception_type") or "the rollout did not finish")
    elif reward is None and len(r.get("rewards") or {}) > 1:
        caption = "several rewards, none named reward: <code>serve --reward-key</code> picks one"
    elif reward is None:
        caption = "the verifier did not produce a reward"
    else:
        caption = "reward"
    turns = r.get("turns") or []
    is_eval = r.get("rollout_type", "train") == "eval"
    generated = sum(len(t.get("completion_token_ids") or []) for t in turns)
    facts = [
        ("Model calls", f"{r.get('n_turns', 0):,}"),
        ("Actions", f"{count_actions(r):,}"),
        ("Conversations", f"{r.get('n_roots', 0):,}"),
        ("Duration", _duration(r.get("wall_s"))),
    ]
    if not is_eval:
        facts.append(("Trainable tokens", f"{r.get('n_trainable_tokens', 0):,}"))
    if generated:
        facts.append(("Generated tokens", f"{generated:,}"))
    meter = ""
    if isinstance(reward, (int, float)) and 0 <= reward <= 1 and r.get("ok"):
        meter = f'<div class="rp-meter" style="color:var(--{"ok" if cls == "full" else "warn" if cls == "part" else "err"})"><i style="width:{max(reward, 0.02) * 100:.0f}%"></i></div>'
    grade = [
        '<section class="hb-panel rp-grade">'
        f'<div class="hb-panel-h"><h3>{icon("target", 15)}Result</h3><span class="aside"><span class="hb-status {tone}">{word}</span></span></div>'
        '<div class="hb-panel-b">'
        f'<div class="rp-score"><span class="rp-big {cls}">{_e(big)}</span><span class="of">{caption}</span></div>{meter}'
    ]
    rewards = r.get("rewards") or {}
    if len(rewards) > 1:
        chosen = r.get("reward_key", "")
        grade.append(
            '<ul class="rp-list">'
            + "".join(
                f'<li><span class="mk">{icon("check" if k == chosen else "more", 14)}</span><b>{html.escape(k)}</b><span class="sc">{v:.3f}</span></li>'
                for k, v in sorted(rewards.items())
            )
            + "</ul>"
        )
    for step in r.get("step_results") or []:
        vals = ", ".join(f"{k} {v:.2f}" for k, v in (step.get("rewards") or {}).items())
        grade.append(
            f'<div class="hb-fine">Step {html.escape(str(step.get("name", "")))}: {html.escape(vals)}</div>'
        )
    grade.append(
        '<dl class="hb-kv">'
        + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts)
        + "</dl>"
    )
    if r.get("error"):
        grade.append(f'<pre class="rp-err">{html.escape(str(r["error"])[:1500])}</pre>')
    grade.append("</div></section>")

    atif = str(r.get("atif") or "none")
    checks = [
        f'<li class="{"ok" if atif == "match" else "bad" if atif == "MISMATCH" else ""}"><span class="mk">{icon("check" if atif == "match" else "x" if atif == "MISMATCH" else "more", 14)}</span>'
        f"<div><b>Trace check: {html.escape(atif)}</b><p>{html.escape(_ATIF.get(atif, ''))}</p></div><span></span></li>"
    ]
    for fix in r.get("param_fixes") or []:
        checks.append(
            f'<li class="warn"><span class="mk">{icon("alert", 14)}</span><div><b>Changed upstream: {html.escape(str(fix))}</b>'
            "<p>The request sent differs from what the agent sent.</p></div><span></span></li>"
        )
    panel = (
        f'<section class="hb-panel"><div class="hb-panel-h"><h3>{icon("shield", 15)}Checks</h3></div><div class="hb-panel-b" style="display:grid;gap:10px">'
        f'<ul class="rp-list">{"".join(checks)}</ul>{_findings_html(r.get("findings") or [])}'
    )
    if r.get("agent_log_tail"):
        panel += (
            f'<details><summary class="hb-disclose">{icon("chevronRight", 13, "chev")}Agent log</summary>'
            f'<pre class="term">{html.escape(str(r["agent_log_tail"])[:4000])}</pre></details>'
        )
    panel += "</div></section>"
    return "".join(grade) + panel


def _duration(seconds: Any) -> str:
    if seconds in (None, ""):
        return "–"
    s = int(float(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


# ── tokens, for a trainable rollout ──────────────────────────────────────────────────────────────


def _confidence(mean_logp: float) -> str:
    """A bar for mean logprob. Closer to 0 is more confident; -1.0 is the practical floor here."""
    pct = max(0.0, min(1.0, 1.0 + mean_logp))
    return f'<span class="rp-conf" title="mean logprob {mean_logp:.3f}"><i style="width:{pct * 100:.0f}%"></i></span>'


def _turns_html(r: dict[str, Any]) -> str:
    """Per model call: what it did, how much it wrote, how sure it was. Only meaningful with token ids."""
    turns = r.get("turns") or []
    if not turns:
        return '<div class="hb-fine">No model calls were captured.</div>'
    rows = []
    for t in turns:
        lp = t.get("per_token_logps") or []
        mean = sum(lp) / len(lp) if lp else 0.0
        gen = len(t.get("completion_token_ids") or [])
        calls = t.get("tool_calls") or []
        if calls:
            action = " ".join(
                f"<code>{html.escape(str(c.get('name', 'tool')))}</code>" for c in calls
            )
        elif t.get("finish_reason") == "stop":
            action = '<span class="faint">final answer</span>'
        else:
            action = '<span class="faint">text only</span>'
        note = ' <span class="hb-chip">discarded</span>' if t.get("discarded") else ""
        rows.append(
            f'<tr><td class="num">{t.get("turn")}</td><td>{action}{note}</td><td class="num">{gen:,}</td>'
            f'<td>{_confidence(mean) if lp else ""}</td><td class="faint">{html.escape(str(t.get("finish_reason") or ""))}</td></tr>'
        )
    return (
        '<div class="hb-tbl"><table><thead><tr><th class="num">#</th><th>Action</th><th class="num">Tokens</th><th>Confidence</th>'
        "<th>Stopped because</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        '<p class="hb-fine" style="margin-top:8px">Confidence is the mean logprob of the sampled tokens. Discarded calls '
        "were generated and billed but lead nowhere, so training paths leave them out.</p>"
    )
