# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the evaluation loop without loading model weights or a server."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_play():
    tree = ast.parse((ROOT / "examples/wordle/evaluate.py").read_text())
    names = {"PROMPT", "TOOL_CALL", "THINK", "FEEDBACK"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "play")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    namespace = dict(json=json, re=re, TextArenaAction=SimpleNamespace, TOOLS=[])
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "evaluate.py", "exec"),
        namespace,
    )
    return namespace["play"]


class Inputs(dict):
    input_ids = SimpleNamespace(shape=(1, 0))

    def to(self, device):
        return self


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, *args, **kwargs):
        return Inputs()

    def decode(self, *args, **kwargs):
        return '<tool_call>{"arguments": {"guess": "[apple]"}}</tool_call>'


class Environment:
    def __init__(self, invalid_turns=(), terminal_at=None, win_at=None):
        self.invalid_turns = invalid_turns
        self.terminal_at = terminal_at
        self.win_at = win_at
        self.calls = 0
        self.full = ""

    def reset(self):
        return SimpleNamespace(
            observation=SimpleNamespace(messages=[SimpleNamespace(content=self.full)])
        )

    def step(self, action):
        self.calls += 1
        invalid = self.calls in self.invalid_turns
        self.full += (
            "Invalid Move: rejected" if invalid else "Feedback:\nA P P L E\nX X X X X"
        )
        return SimpleNamespace(
            observation=self.reset().observation,
            reward=1.0 if self.calls == self.win_at else 0.0,
            done=self.calls == self.terminal_at,
        )

    def state(self):
        return SimpleNamespace(raw_state={"game_state": {"secret_word": "apple"}})


@pytest.mark.parametrize(
    "invalid_turns,terminal_at,win_at,expected_calls,won",
    [
        ({1}, 7, 7, 7, True),
        ({1, 3, 5, 7, 9, 11}, 12, 12, 12, True),
        (set(), None, None, 6, False),
        (set(range(1, 20)), None, None, 12, False),
        ({1, 2}, 2, 2, 2, False),
    ],
)
def test_guess_budget(invalid_turns, terminal_at, win_at, expected_calls, won):
    env = Environment(invalid_turns, terminal_at, win_at)
    model = SimpleNamespace(device="cpu", generate=lambda **kwargs: [[]])
    args = SimpleNamespace(thinking=False, temperature=0, max_new_tokens=16)
    result = load_play()(model, Tokenizer(), env, args)
    assert env.calls == expected_calls
    assert len(result["rows"]) == expected_calls
    assert result["won"] is won


def test_notebook_commands_use_configured_arguments(monkeypatch):
    import subprocess

    notebook = json.loads((ROOT / "examples/wordle/wordle.ipynb").read_text())
    answers = iter(["example-org", "https://example-org-wordle.hf.space"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    namespace = {}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "subprocess.run" in source:
            # Skip the download magic; all publication/job calls are intercepted.
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("!")
            )
            exec(compile(source, "wordle.ipynb", "exec"), namespace)
    assert len(calls) == 3
    for argv, kwargs in calls:
        assert kwargs == {"check": True}
        assert not any("<user>" in arg for arg in argv)
    assert calls[0][0][-1] == "example-org/wordle-tool-calls-sft"
    assert calls[1][0][-1] == "example-org/Qwen3-1.7B-Wordle-SFT"
    assert calls[2][0][-1] == "https://example-org-wordle.hf.space"


@pytest.mark.parametrize(
    "answers", [["<user>"], ["example-org", "https://<user>-wordle.hf.space"]]
)
def test_notebook_rejects_unconfigured_placeholders(monkeypatch, answers):
    import subprocess

    notebook = json.loads((ROOT / "examples/wordle/wordle.ipynb").read_text())
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("Command ran before validation")
    )
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "hub_namespace = input" in "".join(c["source"])
    )
    source = "\n".join(line for line in source.splitlines() if not line.startswith("!"))
    with pytest.raises(ValueError):
        exec(compile(source, "wordle.ipynb", "exec"), {})
