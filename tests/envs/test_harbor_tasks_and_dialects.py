# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task discovery and wire-dialect detection.

Task order is identity: a task's *index* is what a trainer's dataset row, a `run_rollout` argument
and a result all refer to, so an unstable order silently changes which task an index means between
runs. Dialect detection decides which transformer parses a request, and misclassifying it produces a
valid-looking response in the wrong envelope, which harnesses report as nothing at all.
"""

from __future__ import annotations

import pytest

detection = pytest.importorskip("openenv.core.harness.capture.detection")
tasks = pytest.importorskip("openenv.harbor.tasks")

APIType = detection.APIType
detect = detection.detect


def make_task(root, name: str) -> None:
    """A directory shaped like a Harbor task, enough for discovery to list it."""
    task = root / "tasks" / name
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('[task]\nname = "%s"\n' % name)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")


# --- dialect detection ------------------------------------------------------
@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/chat/completions", APIType.OPENAI_CHAT),
        ("/v1/messages", APIType.ANTHROPIC),
        ("/v1/responses", APIType.OPENAI_RESPONSES),
        ("/v1beta/models/gemini:generateContent", APIType.GOOGLE),
        ("/v1beta/models/gemini:streamGenerateContent?alt=sse", APIType.GOOGLE),
    ],
)
def test_path_decides_the_dialect(path, expected):
    assert detect(path, {}, {}) is expected


def test_path_wins_over_a_misleading_body():
    """A body key must never override an explicit route."""
    assert detect("/v1/messages", {}, {"contents": []}) is APIType.ANTHROPIC


def test_anthropic_header_is_used_when_the_path_is_unhelpful():
    assert (
        detect("/proxy", {"anthropic-version": "2023-06-01"}, {}) is APIType.ANTHROPIC
    )


def test_header_matching_is_case_insensitive():
    assert (
        detect("/proxy", {"Anthropic-Version": "2023-06-01"}, {}) is APIType.ANTHROPIC
    )


def test_body_shape_is_the_last_resort():
    assert detect("/x", {}, {"contents": [{"parts": []}]}) is APIType.GOOGLE
    assert detect("/x", {}, {"input": "hi", "instructions": "be brief"}) is (
        APIType.OPENAI_RESPONSES
    )


def test_unknown_requests_default_to_chat_completions():
    """The most common dialect, and the one the proxy normalises everything else into."""
    assert detect("/", {}, {}) is APIType.OPENAI_CHAT
    assert detect("/x", {}, {"messages": []}) is APIType.OPENAI_CHAT


# --- task discovery ---------------------------------------------------------
def test_tasks_subdirectory_is_preferred_when_present(tmp_path):
    make_task(tmp_path, "b_task")
    make_task(tmp_path, "a_task")
    found = tasks._task_dirs_from_directory(tmp_path)
    assert [p.name for p in found] == ["a_task", "b_task"]


def test_a_directory_of_tasks_works_without_a_tasks_subdir(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name / "environment").mkdir(parents=True)
        (tmp_path / name / "task.toml").write_text("[task]\n")
    assert [p.name for p in tasks._task_dirs_from_directory(tmp_path)] == ["one", "two"]


def test_order_is_stable_and_sorted(tmp_path):
    """Index is identity. Any ordering that depends on filesystem iteration is a data bug."""
    for name in ("zebra", "alpha", "mike", "0001", "0010", "0002"):
        make_task(tmp_path, name)
    first = [p.name for p in tasks._task_dirs_from_directory(tmp_path)]
    second = [p.name for p in tasks._task_dirs_from_directory(tmp_path)]
    assert first == second == sorted(first)


def test_hidden_directories_are_skipped(tmp_path):
    make_task(tmp_path, "real")
    (tmp_path / "tasks" / ".ipynb_checkpoints").mkdir(parents=True)
    assert [p.name for p in tasks._task_dirs_from_directory(tmp_path)] == ["real"]


def test_grouped_task_folders_are_found(tmp_path):
    """terminal-bench-science nests tasks as `tasks/<field>/<subfield>/<task>/`."""
    grouped = tmp_path / "tasks"
    for rel in ("earth/ocean/b-task", "earth/geo/a-task", "life/c-task"):
        (grouped / rel).mkdir(parents=True)
        (grouped / rel / "task.toml").write_text("")
    found = tasks._task_dirs_from_directory(tmp_path)
    assert [p.name for p in found] == ["a-task", "b-task", "c-task"]


def test_a_flat_folder_without_task_toml_does_not_trigger_the_nested_search(tmp_path):
    """One broken task must not reshuffle the indexes of the others."""
    for name in ("t2", "t1", "broken"):
        (tmp_path / "tasks" / name).mkdir(parents=True)
    for name in ("t1", "t2"):
        (tmp_path / "tasks" / name / "task.toml").write_text("")
    assert [p.name for p in tasks._task_dirs_from_directory(tmp_path)] == [
        "broken",
        "t1",
        "t2",
    ]


def test_a_grouped_layout_does_not_follow_symlinks(tmp_path):
    """A symlinked folder can point outside the dataset, or back into it (a loop); neither is walked."""
    (tmp_path / "tasks" / "group" / "a").mkdir(parents=True)
    (tmp_path / "tasks" / "group" / "a" / "task.toml").write_text("")
    (tmp_path / "outside" / "b").mkdir(parents=True)
    (tmp_path / "outside" / "b" / "task.toml").write_text("")
    (tmp_path / "tasks" / "linked").symlink_to(tmp_path / "outside")
    (tmp_path / "tasks" / "group" / "loop").symlink_to(tmp_path / "tasks")
    found = tasks._task_dirs_from_directory(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["tasks/group/a"]


def test_discovery_does_not_validate_by_default(tmp_path):
    """Validation costs a file read per task, and discovery is on every /splits call.

    A malformed task surfaces as a failed rollout with Harbor's own error rather than vanishing from
    the listing, which would also shift the index of every task after it.
    """
    make_task(tmp_path, "good")
    (tmp_path / "tasks" / "malformed").mkdir(parents=True)  # no task.toml
    listed = tasks._task_dirs_from_directory(tmp_path, validate=False)
    assert [p.name for p in listed] == ["good", "malformed"]


def test_validation_can_be_requested_explicitly(tmp_path):
    pytest.importorskip("harbor.models.task.task")
    make_task(tmp_path, "good")
    (tmp_path / "tasks" / "malformed").mkdir(parents=True)
    listed = tasks._task_dirs_from_directory(tmp_path, validate=True)
    assert "malformed" not in [p.name for p in listed]


def test_an_empty_dataset_raises_rather_than_serving_zero_tasks(tmp_path):
    (tmp_path / "tasks").mkdir()
    with pytest.raises(ValueError, match="no Harbor tasks"):
        tasks.resolve_task_dirs(str(tmp_path), refresh=True)


def test_resolve_caches_and_refresh_bypasses_it(tmp_path):
    make_task(tmp_path, "one")
    first = tasks.resolve_task_dirs(str(tmp_path), refresh=True)
    make_task(tmp_path, "two")
    assert tasks.resolve_task_dirs(str(tmp_path)) == first  # cached
    assert len(tasks.resolve_task_dirs(str(tmp_path), refresh=True)) == 2


# --- the symlink regression -------------------------------------------------
def test_has_symlinks_detects_an_outward_pointing_link(tmp_path):
    """Modal graded 0/5 while E2B graded 4/5 because of exactly this.

    The HF cache is a symlink farm, Harbor uploads a task's `tests/` by tarring it, and tar preserves
    symlinks, so the sandbox received dangling links. Bash reports those as "No such file or
    directory", which reads as a failed upload rather than a symlink problem.
    """
    outside = tmp_path / "blobs" / "sha"
    outside.parent.mkdir()
    outside.write_text("#!/bin/sh\necho ok\n")
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").symlink_to(outside)

    found = tasks.has_symlinks(task)
    assert [p.name for p in found] == ["test.sh"]


def test_has_symlinks_is_empty_for_real_files(tmp_path):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    assert tasks.has_symlinks(task) == []


# --- dataset spec classification --------------------------------------------
@pytest.mark.parametrize(
    "spec,is_repo",
    [
        ("AdithyaSK/data_agent_rl_environment_train", True),
        ("org/name", True),
        ("terminal-bench@1.0", False),
        ("/abs/path", False),
        ("./relative", False),
        ("~/home", False),
        ("org/team/name", False),
        ("bare", False),
    ],
)
def test_hf_repo_specs_are_distinguished_from_paths_and_registry_names(spec, is_repo):
    assert tasks._is_hf_repo(spec) is is_repo


# --- trace fallback for harnesses that emit no ATIF --------------------------
def test_pi_session_log_is_used_when_there_is_no_trajectory(tmp_path):
    """Three of sixteen harnesses write no `trajectory.json`, leaving no independent check.

    pi records the same thing under another name: every assistant record in its session log carries
    `usage.output`, the completion-token count for that call, which is what ATIF calls
    `metrics.completion_tokens`. So the cross-check exists after all.
    """
    import json as _json

    atif = pytest.importorskip("openenv.harbor.atif")
    sessions = tmp_path / "agent" / "pi" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "s.jsonl").write_text(
        "\n".join(
            _json.dumps(r)
            for r in [
                {"type": "session", "id": "x"},
                {
                    "type": "message",
                    "message": {"role": "assistant", "usage": {"output": 63}},
                },
                {"type": "message", "message": {"role": "toolResult", "content": "ok"}},
                {
                    "type": "message",
                    "message": {"role": "assistant", "usage": {"output": 41}},
                },
            ]
        )
    )
    trace, source = atif.load_trace(tmp_path)
    assert source == "pi_session"
    assert atif.atif_turn_lengths(trace) == [63, 41]


def test_a_real_trajectory_wins_over_any_fallback(tmp_path):
    import json as _json

    atif = pytest.importorskip("openenv.harbor.atif")
    agent = tmp_path / "agent"
    (agent / "pi" / "sessions").mkdir(parents=True)
    (agent / "pi" / "sessions" / "s.jsonl").write_text(
        _json.dumps(
            {
                "type": "message",
                "message": {"role": "assistant", "usage": {"output": 9}},
            }
        )
    )
    (agent / "trajectory.json").write_text(
        _json.dumps(
            {"steps": [{"source": "agent", "metrics": {"completion_tokens": 77}}]}
        )
    )
    trace, source = atif.load_trace(tmp_path)
    assert source == "atif"
    assert atif.atif_turn_lengths(trace) == [77]


def test_no_trace_at_all_is_reported_honestly(tmp_path):
    """hermes writes a zero-byte session file and openclaw only echoes its config back."""
    atif = pytest.importorskip("openenv.harbor.atif")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "hermes-session.jsonl").write_text("")
    trace, source = atif.load_trace(tmp_path)
    assert trace is None and source == ""
