# SPDX-License-Identifier: BSD-3-Clause
"""Run the training entry points with small fakes, without model or network access."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def run_training(monkeypatch, script, trainer, client=None):
    monkeypatch.setattr(
        sys,
        "argv",
        [script, "--model", "example/model", "--hub-model-id", "example/output"]
        + (["--dataset", "example/games"] if script == "sft.py" else []),
    )
    modules = {
        "torch": SimpleNamespace(bfloat16="bfloat16", float32="float32"),
        "datasets": SimpleNamespace(
            load_dataset=lambda name, split: [(name, split)],
            Dataset=SimpleNamespace(from_dict=lambda data: data),
        ),
        "trl": SimpleNamespace(
            SFTConfig=SimpleNamespace,
            SFTTrainer=trainer,
            GRPOConfig=SimpleNamespace,
            GRPOTrainer=trainer,
        ),
        "huggingface_hub": SimpleNamespace(get_token=lambda: None),
        "textarena_env": SimpleNamespace(
            TextArenaEnv=lambda **kwargs: client, TextArenaAction=SimpleNamespace
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    runpy.run_path(str(ROOT / "examples/wordle" / script), run_name="__main__")


def test_sft_explicit_fp32_weights_and_bf16_compute_then_upload(monkeypatch):
    events = []

    class Trainer:
        def __init__(self, *, model, train_dataset, args):
            assert model == "example/model"
            assert train_dataset == [("example/games", "train")]
            assert args.model_init_kwargs == {"dtype": "float32"}
            assert args.bf16 is True
            self.model = SimpleNamespace(
                to=lambda dtype: events.append(("cast", dtype))
            )

        def train(self):
            events.append("train")

        def push_to_hub(self, **kwargs):
            events.append(("upload", kwargs))

    run_training(monkeypatch, "sft.py", Trainer)
    assert events == [
        "train",
        ("cast", "bfloat16"),
        ("upload", {"dataset_name": "example/games"}),
    ]


class WordleClient:
    """Only legal guesses enter history; six legal guesses or repeated invalids end play."""

    def __init__(self, invalid_turns, *, win=False, terminal_invalid=False):
        self.invalid_turns = invalid_turns
        self.win = win
        self.terminal_invalid = terminal_invalid
        self.calls = self.legal = self.consecutive_invalids = 0
        self.full = "Wordle: six legal guesses."

    def result(self, *, done=False, reward=0.0):
        return SimpleNamespace(
            observation=SimpleNamespace(messages=[SimpleNamespace(content=self.full)]),
            done=done,
            reward=reward,
        )

    def reset(self):
        return self.result()

    def step(self, action):
        self.calls += 1
        if self.calls in self.invalid_turns:
            self.consecutive_invalids += 1
            self.full += "\nInvalid move: rejected."
            return self.result(
                done=self.terminal_invalid and self.consecutive_invalids == 2
            )
        self.consecutive_invalids = 0
        self.legal += 1
        self.full += "\nFeedback:\nA P P L E\nG X X X X"
        return self.result(
            done=self.legal == 6, reward=float(self.win and self.legal == 6)
        )


@pytest.mark.parametrize("calls_per_round", [1, 2])
@pytest.mark.parametrize(
    "invalid_turns,win,terminal_invalid,expected_calls,expected_legal",
    [
        ({1}, True, False, 7, 6),
        ({1, 3, 5, 7, 9, 11}, True, False, 12, 6),
        (set(), False, False, 6, 6),
        ({1, 2}, False, True, 2, 0),
        (set(range(1, 20)), False, False, 12, 0),
    ],
)
def test_grpo_rollout_budget(
    monkeypatch,
    invalid_turns,
    win,
    terminal_invalid,
    expected_calls,
    expected_legal,
    calls_per_round,
):
    client = WordleClient(invalid_turns, win=win, terminal_invalid=terminal_invalid)

    class Trainer:
        def __init__(self, *, environment_factory, args, **kwargs):
            self.env = environment_factory()
            self.args = args
            self.model = SimpleNamespace(to=lambda dtype: None)

        def train(self):
            self.env.reset()
            # TRL 1.16.0 counts tool rounds, even for invalid results. It can keep
            # calling tools after done; the adapter must reject those calls.
            for _ in range(self.args.max_tool_calling_iterations):
                for _ in range(calls_per_round):
                    try:
                        self.env.guess("[apple]")
                    except ValueError as error:
                        assert str(error) == "Game over."
            assert client.calls == expected_calls
            assert client.legal == self.env.num_guesses == expected_legal
            assert self.env.invalid_moves == expected_calls - expected_legal
            assert self.env.won is win
            assert self.args.max_tool_calling_iterations == 12
            assert self.args.bf16 is True

        def push_to_hub(self):
            pass

    run_training(monkeypatch, "grpo.py", Trainer, client)
