# SPDX-License-Identifier: BSD-3-Clause

"""Turning Harbor rollouts into conversational SFT rows.

A wrong row trains silently: a tool call whose arguments are a JSON string breaks templates that iterate over them,
an auxiliary call (a session title) becomes a training target, or a filter applied by default drops data the caller
wanted to keep.
"""

from __future__ import annotations

import json

import pytest

sft = pytest.importorskip("openenv.harbor.sft")
models = pytest.importorskip("openenv.harbor.models")

SYSTEM = {"role": "system", "content": "You are a data agent."}
TASK = {"role": "user", "content": "How many rows are in data.csv?"}
TOOLS = [
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
]


def call(name, arguments, call_id=None, nested=True):
    fn = {"name": name, "arguments": json.dumps(arguments)}
    if not nested:
        return {**fn, **({"id": call_id} if call_id else {})}
    return {"id": call_id, "type": "function", "function": fn}


def rollout(
    reward=1.0,
    rewards=None,
    last_reply="There are 42 rows.",
    finish_reason="stop",
    extra_turns=(),
):
    """An OpenCode-like rollout: one tool call, its output, then the answer."""
    first_request = [SYSTEM, TASK]
    tool_turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call("bash", {"command": "wc -l data.csv"}, call_id="c1")],
    }
    final_request = first_request + [
        tool_turn,
        {"role": "tool", "tool_call_id": "c1", "content": "43 data.csv"},
    ]
    turns = [
        {
            "turn": 0,
            "role": "agent",
            "finish_reason": "tool_calls",
            "request_messages": first_request,
            "request_tools": TOOLS,
            "tool_calls": [
                call("bash", {"command": "wc -l data.csv"}, call_id="c1", nested=False)
            ],
        },
        *extra_turns,
        {
            "turn": 1,
            "role": "agent",
            "finish_reason": finish_reason,
            "request_messages": final_request,
            "request_tools": TOOLS,
            "text": last_reply,
        },
    ]
    return {
        "task_name": "0000_001_qa_1",
        "dataset": "org/tasks",
        "harness": "opencode",
        "reward": reward,
        "rewards": rewards
        if rewards is not None
        else ({"reward": reward} if reward is not None else {}),
        "rollout_type": "eval",
        "turns": turns,
        "n_turns": len(turns),
    }


def test_row_is_the_last_request_plus_its_reply():
    row = sft.to_sft_row(models.HarborRolloutResult.model_validate(rollout()))
    assert [m["role"] for m in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["messages"][-1] == {"role": "assistant", "content": "There are 42 rows."}
    assert row["tools"] == TOOLS
    assert row["reward"] == 1.0
    assert (
        row["rollout_type"] == "eval"
    )  # hosted-provider rollouts are exportable as text


def test_tool_call_arguments_become_dicts():
    row = sft.to_sft_row(models.HarborRolloutResult.model_validate(rollout()))
    (tool_call,) = row["messages"][2]["tool_calls"]
    assert tool_call == {
        "type": "function",
        "function": {"name": "bash", "arguments": {"command": "wc -l data.csv"}},
        "id": "c1",
    }


def test_flat_reply_tool_calls_are_nested_and_parsed():
    # A rollout whose last call is a tool call: the reply's flat {name, arguments} must be nested too.
    result = rollout(last_reply="", finish_reason="tool_calls")
    result["turns"][-1]["tool_calls"] = [call("bash", {"command": "ls"}, nested=False)]
    row = sft.to_sft_row(models.HarborRolloutResult.model_validate(result))
    assert row["messages"][-1]["tool_calls"] == [
        {
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "ls"}},
        }
    ]


def test_discarded_retry_is_never_the_row():
    result = rollout()
    result["turns"].append(  # after the real last agent turn
        {
            "turn": 2,
            "role": "agent",
            "discarded": True,
            "request_messages": [TASK],
            "text": "retry",
        }
    )
    row = sft.to_sft_row(models.HarborRolloutResult.model_validate(result))
    assert row["messages"][-1]["content"] == "There are 42 rows."


def test_unusable_capture_is_skipped_but_a_nonzero_exit_is_not():
    fatal, mismatch, exited = rollout(), rollout(), rollout()
    fatal["findings"] = ["[FATAL] no model calls captured"]
    mismatch["atif"] = "MISMATCH"
    # older OpenEnv marked graded OpenCode runs not ok because OpenCode exits 1: the conversation is sound
    exited.update(ok=False, exception_type="NonZeroAgentExitCodeError")
    rows, stats = sft.export_sft([fatal, mismatch, exited], min_reward=1.0)
    assert stats.capture_failed == 2 and len(rows) == 1


def test_empty_reply_and_malformed_tool_calls_are_skipped():
    empty = rollout(last_reply="")
    bad_json, not_object = rollout(), rollout()
    for r, raw in ((bad_json, "{not json"), (not_object, "[1, 2]")):
        # the earlier assistant turn, as the last request carries it
        r["turns"][-1]["request_messages"][2] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": raw},
                }
            ],
        }
    rows, stats = sft.export_sft([empty, bad_json, not_object])
    assert rows == []
    assert (stats.empty_reply, stats.malformed_tool_call) == (1, 2)


def test_no_filter_by_default():
    rows, stats = sft.export_sft(
        [rollout(reward=0.0), rollout(reward=1.0), rollout(reward=None)]
    )
    assert [r["reward"] for r in rows] == [0.0, 1.0, None]
    assert stats.rows == 3


def test_min_reward_keeps_solved_and_drops_ungraded():
    rows, stats = sft.export_sft(
        [rollout(reward=0.0), rollout(reward=1.0), rollout(reward=None)], min_reward=1.0
    )
    assert [r["reward"] for r in rows] == [1.0]
    assert (stats.below_min_reward, stats.ungraded) == (1, 1)


def test_reward_from_named_rewards():
    several = {"correctness": 1.0, "submission": 1.0}
    unpicked = rollout(reward=None, rewards=several)
    assert (
        sft.export_sft([unpicked], min_reward=1.0)[1].ungraded == 1
    )  # ambiguous without a key
    rows, _ = sft.export_sft([unpicked], min_reward=1.0, reward_key="correctness")
    assert len(rows) == 1
    single = rollout(reward=None, rewards={"correctness": 1.0})
    assert sft.export_sft([single], min_reward=1.0)[0][0]["reward"] == 1.0


def test_reward_key_preference_order():
    # a split mixing graders: some tasks write `correctness`, others only `reward`
    multi = rollout(reward=None, rewards={"correctness": 0.0, "submission": 1.0})
    single = rollout(reward=None, rewards={"reward": 1.0, "tool_efficiency": 0.5})
    rows, stats = sft.export_sft([multi, single], reward_key="correctness,reward")
    assert [r["reward"] for r in rows] == [0.0, 1.0]
    assert stats.ungraded == 0
    assert (
        sft.export_sft([multi], min_reward=0.0, reward_key="missing")[1].ungraded == 1
    )


def test_rollout_cut_short_is_reported():
    _, stats = sft.export_sft([rollout(last_reply="", finish_reason="tool_calls")])
    assert stats.ended_on_tool_call == ["0000_001_qa_1"]


def test_rollout_without_agent_turn_is_skipped():
    empty = {**rollout(), "turns": []}
    rows, stats = sft.export_sft([empty])
    assert rows == [] and stats.no_agent_turn == 1


def test_cli_writes_rows_from_a_results_directory(tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    harbor_cli = pytest.importorskip("openenv.cli.commands.harbor")
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(json.dumps([rollout(reward=1.0)]))
    (results / "b.json").write_text(json.dumps(rollout(reward=0.0)))
    out = tmp_path / "rows.jsonl"
    outcome = typer_testing.CliRunner().invoke(
        harbor_cli.app,
        ["export-sft", str(results), "--out", str(out), "--min-reward", "1"],
    )
    assert outcome.exit_code == 0, outcome.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["messages"][-1]["content"] == "There are 42 rows."


# Without --private the namespace's default visibility applies, as with `hf upload`: passing
# False would make a dataset public in an organization whose repos are private by default.
@pytest.mark.parametrize("flag, private", [(["--private"], True), ([], None)])
def test_cli_reads_reward_key_skips_other_json_and_pushes(
    tmp_path, monkeypatch, flag, private
):
    typer_testing = pytest.importorskip("typer.testing")
    harbor_cli = pytest.importorskip("openenv.cli.commands.harbor")
    hub = pytest.importorskip("huggingface_hub")
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(
        json.dumps(
            rollout(reward=None, rewards={"correctness": 1.0, "submission": 1.0})
        )
    )
    (results / "summary.json").write_text(json.dumps({"solved": 1}))
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, **kwargs):
            calls.append(("create_repo", repo_id, kwargs.get("private")))

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs["repo_id"], kwargs["path_in_repo"]))

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    out = tmp_path / "rows.jsonl"
    args = ["export-sft", str(results), "--out", str(out), "--min-reward", "1"]
    args += [
        "--reward-key",
        "correctness,reward",
        "--push-to",
        "me/traces",
        *flag,
    ]
    outcome = typer_testing.CliRunner().invoke(harbor_cli.app, args)
    assert outcome.exit_code == 0, outcome.output
    assert len(out.read_text().splitlines()) == 1
    assert calls == [
        ("create_repo", "me/traces", private),
        ("upload_file", "me/traces", "data/rows.jsonl"),
    ]


def test_cli_rejects_a_missing_path(tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    harbor_cli = pytest.importorskip("openenv.cli.commands.harbor")
    outcome = typer_testing.CliRunner().invoke(
        harbor_cli.app, ["export-sft", str(tmp_path / "nope.json")]
    )
    assert outcome.exit_code == 2  # a usage error, not a traceback
