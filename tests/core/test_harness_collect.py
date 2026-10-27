# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rollout collection, serialization, and resume."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    ModelStep,
    ModelStepResult,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    ToolTraceEntry,
    VerifyResult,
)
from openenv.core.harness.collect import (
    build_dataset_readme,
    build_model_step,
    CollectResult,
    CollectRunner,
    EpisodeRecord,
    push_to_hf_hub,
    README_FILENAME,
    RolloutSerializer,
)
from openenv.core.llm_client import LLMClient, LLMResponse, ToolCall


def _fake_rollout(
    *,
    reward: float = 1.0,
    done: bool = True,
    turns: int = 1,
) -> HarnessRolloutResult:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "play"},
    ]
    trace: list[ToolTraceEntry] = []
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"move {i}"})
        trace.append(
            ToolTraceEntry(
                tool_name="play_move",
                arguments={"cell": i},
                result=ToolResult(
                    data={"board": "...", "reward": 0.0 if i < turns - 1 else reward},
                    done=(i == turns - 1) and done,
                    metadata={"reward": 0.0 if i < turns - 1 else reward},
                ),
            )
        )
        messages.append(
            {
                "role": "tool",
                "name": "play_move",
                "content": f"cell={i}",
            }
        )

    return HarnessRolloutResult(
        messages=messages,
        tool_trace=trace,
        events=[],
        done=done,
        metrics={"turns": turns, "tool_calls": turns},
        prompt_ids=[],
        completion_ids=[],
        logprobs=[],
    )


def _fake_verify(reward: float = 1.0) -> VerifyResult:
    return VerifyResult(
        env_reward=reward,
        done=True,
        metrics={"step_count": 1},
        artifacts={"final_state": {"winner": "X"}},
    )


class TestEpisodeRecord:
    def test_from_rollout_populates_core_fields(self):
        rollout = _fake_rollout(reward=1.0)
        verify = _fake_verify(reward=1.0)

        record = EpisodeRecord.from_rollout("ep1", rollout, verify)

        assert record.episode_id == "ep1"
        assert record.messages == rollout.messages
        assert record.reward == 1.0
        assert record.done is True
        assert record.metrics == {"turns": 1, "tool_calls": 1}

    def test_from_rollout_flattens_tool_trace(self):
        rollout = _fake_rollout(turns=2)
        verify = _fake_verify()

        record = EpisodeRecord.from_rollout("ep1", rollout, verify)

        assert len(record.tool_trace) == 2
        first = record.tool_trace[0]
        assert first["tool_name"] == "play_move"
        assert first["arguments"] == {"cell": 0}
        assert "result" in first
        assert first["result"]["done"] is False

    def test_from_rollout_uses_resolve_env_reward(self):
        """Reward must be derived through _resolve_env_reward to respect the
        'rewards in env' invariant enforced by the harness runtime."""
        rollout = _fake_rollout(reward=1.0)
        # Verify carries a mismatching reward — must raise, NOT be silently ignored.
        verify = VerifyResult(env_reward=0.0)

        with pytest.raises(ValueError):
            EpisodeRecord.from_rollout("ep1", rollout, verify)

    def test_from_rollout_accepts_task_and_extra(self):
        rollout = _fake_rollout()
        verify = _fake_verify()

        record = EpisodeRecord.from_rollout(
            "ep1",
            rollout,
            verify,
            task={"game": "tic_tac_toe", "seed": 42},
            extra={"teacher": "gpt-5-mini"},
        )

        assert record.task == {"game": "tic_tac_toe", "seed": 42}
        assert record.extra == {"teacher": "gpt-5-mini"}

    def test_to_dict_returns_json_safe_payload(self):
        record = EpisodeRecord.from_rollout(
            "ep1",
            _fake_rollout(),
            _fake_verify(),
            task=Path("task.json"),
            extra={"artifact": Path("artifact.txt")},
        )

        payload = record.to_dict()

        json.dumps(payload)
        assert payload["task"] == "task.json"
        assert payload["extra"]["artifact"] == "artifact.txt"


class TestRolloutSerializer:
    def test_writes_one_jsonl_line_per_episode(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)

        serializer.write_episode(
            EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
        )
        serializer.write_episode(
            EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
        )

        lines = serializer.results_path.read_text().strip().splitlines()
        assert [json.loads(line)["episode_id"] for line in lines] == ["ep1", "ep2"]

    def test_line_contains_trl_ready_messages_column(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        rollout = _fake_rollout(turns=2)

        serializer.write_episode(
            EpisodeRecord.from_rollout("ep1", rollout, _fake_verify())
        )

        payload = json.loads(serializer.results_path.read_text().strip())
        assert payload["messages"] == rollout.messages
        assert payload["messages"][0]["role"] == "user"

    def test_record_schema_has_required_keys(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.write_episode(
            EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
        )

        payload = json.loads(serializer.results_path.read_text().strip())
        for key in (
            "episode_id",
            "messages",
            "reward",
            "done",
            "tool_trace",
            "metrics",
            "verify_metrics",
            "artifacts",
        ):
            assert key in payload, f"missing {key}"

    def test_writes_metadata_sidecar(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.write_metadata(
            {"env_id": "tic_tac_toe", "teacher_model": "gpt-5-mini", "num_episodes": 10}
        )

        metadata = json.loads(serializer.metadata_path.read_text())
        assert metadata["env_id"] == "tic_tac_toe"
        assert metadata["teacher_model"] == "gpt-5-mini"

    def test_collected_episode_ids_from_prior_run(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.write_episode(
            EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
        )
        serializer.write_episode(
            EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
        )

        reopened = RolloutSerializer(tmp_path)
        assert reopened.collected_episode_ids() == {"ep1", "ep2"}

    def test_collected_episode_ids_empty_when_no_file(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path / "brand-new")
        assert serializer.collected_episode_ids() == set()

    def test_creates_output_dir_on_first_write(self, tmp_path: Path):
        target = tmp_path / "nested" / "out"
        serializer = RolloutSerializer(target)

        serializer.write_episode(
            EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
        )

        assert target.exists()
        assert serializer.results_path.exists()

    def test_append_mode_survives_new_serializer_instance(self, tmp_path: Path):
        """Serializer must append, not truncate, when a previous run exists."""
        first = RolloutSerializer(tmp_path)
        first.write_episode(
            EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
        )

        second = RolloutSerializer(tmp_path)
        second.write_episode(
            EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
        )

        lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
        ids = [json.loads(line)["episode_id"] for line in lines]
        assert ids == ["ep1", "ep2"]

    def test_append_separates_complete_record_without_trailing_newline(
        self, tmp_path: Path
    ):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text(json.dumps({"episode_id": "ep1"}))

        serializer.write_episode(
            EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
        )

        lines = serializer.results_path.read_text().splitlines()
        assert [json.loads(line)["episode_id"] for line in lines] == ["ep1", "ep2"]

    def test_append_does_not_read_the_entire_results_file(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text('{"episode_id": "ep1"}\n')

        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("full-file read")
        ):
            serializer.write_episode(
                EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
            )

        lines = serializer.results_path.read_text().splitlines()
        assert [json.loads(line)["episode_id"] for line in lines] == ["ep1", "ep2"]

    def test_append_scans_a_long_final_record_across_chunks(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text(
            json.dumps({"episode_id": "ep1", "payload": "x" * 10_000})
        )

        serializer.write_episode(
            EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
        )

        lines = serializer.results_path.read_text().splitlines()
        assert [json.loads(line)["episode_id"] for line in lines] == ["ep1", "ep2"]

    def test_append_after_unterminated_blank_line(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text('{"episode_id": "ep1"}\n   ')

        serializer.write_episode(
            EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
        )

        records = [
            json.loads(line)
            for line in serializer.results_path.read_text().splitlines()
            if line.strip()
        ]
        assert [record["episode_id"] for record in records] == ["ep1", "ep2"]

    def test_append_rejects_partial_trailing_record_without_modifying_file(
        self, tmp_path: Path
    ):
        serializer = RolloutSerializer(tmp_path)
        original = b'{"episode_id": "ep1"}\n{"episode_id":'
        serializer.results_path.write_bytes(original)

        with pytest.raises(
            ValueError, match=r"results\.jsonl:final line: invalid JSON"
        ):
            serializer.write_episode(
                EpisodeRecord.from_rollout("ep2", _fake_rollout(), _fake_verify())
            )

        assert serializer.results_path.read_bytes() == original

    def test_resume_rejects_malformed_interior_record(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text(
            '{"episode_id": "ep1"}\n{"episode_id":\n{"episode_id": "ep3"}\n'
        )

        with pytest.raises(ValueError, match=r"results\.jsonl:2: invalid JSON"):
            serializer.collected_episode_ids()

    def test_resume_rejects_non_object_record(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text('{"episode_id": "ep1"}\n["ep2"]\n')

        with pytest.raises(ValueError, match=r"results\.jsonl:2: expected an object"):
            serializer.collected_episode_ids()

    def test_resume_rejects_record_without_episode_id(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.results_path.write_text('{"episode_id": "ep1"}\n{"messages": []}\n')

        with pytest.raises(
            ValueError, match=r"results\.jsonl:2: expected a string episode_id"
        ):
            serializer.collected_episode_ids()


class _FakeSession(ResourceSession):
    """Minimal session that returns a fixed reward on verify()."""

    def __init__(self, *, task: Any = None, reward: float = 1.0):
        self.task = task
        self._reward = reward
        self.closed = False

    def initial_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": f"task={self.task}"}]

    def list_tools(self) -> list[Tool]:
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            data={"ack": name},
            done=True,
            metadata={"reward": self._reward},
        )

    def verify(
        self,
        transcript: list[dict[str, Any]],
        final_state: Any | None = None,
    ) -> VerifyResult:
        return VerifyResult(
            env_reward=self._reward,
            done=True,
            metrics={"step_count": 1},
            artifacts={"task": self.task},
        )

    def close(self) -> None:
        self.closed = True


class _FakeFactory(ResourceSessionFactory):
    """Records create() calls and hands out deterministic sessions."""

    def __init__(self, reward_map: dict[Any, float] | None = None):
        self.created: list[dict[str, Any]] = []
        self._reward_map = reward_map or {}
        self.sessions: list[_FakeSession] = []

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> _FakeSession:
        self.created.append({"task": task, "seed": seed, "episode_id": episode_id})
        reward = self._reward_map.get(task, 1.0) if self._reward_map else 1.0
        session = _FakeSession(task=task, reward=reward)
        self.sessions.append(session)
        return session


class _FakeAdapter(HarnessAdapter):
    """Produces a scripted rollout keyed off the session's task."""

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        fake_session = session  # type: ignore[assignment]
        reward = fake_session._reward  # type: ignore[attr-defined]
        messages = list(session.initial_messages())
        messages.append({"role": "assistant", "content": "move"})
        tool_result = session.call_tool("play_move", {"cell": 0})
        trace = [
            ToolTraceEntry(
                tool_name="play_move",
                arguments={"cell": 0},
                result=tool_result,
            )
        ]
        messages.append({"role": "tool", "name": "play_move", "content": "ok"})
        return HarnessRolloutResult(
            messages=messages,
            tool_trace=trace,
            events=[],
            done=True,
            metrics={"turns": 1, "reward": reward},
        )

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError


def _noop_model_step(
    messages: list[dict[str, Any]],
    tools: list[Tool],
    sampling: dict[str, Any],
) -> ModelStepResult:
    return ModelStepResult(response=LLMResponse(content="ok", tool_calls=[]))


class TestCollectRunner:
    def test_writes_requested_number_of_episodes(self, tmp_path: Path):
        runner = CollectRunner(
            session_factory=_FakeFactory(),
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )

        result = runner.run(model_step=_noop_model_step, num_episodes=3)

        assert result.num_collected == 3
        assert result.num_skipped == 0
        lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_assigns_deterministic_episode_ids(self, tmp_path: Path):
        runner = CollectRunner(
            session_factory=_FakeFactory(),
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )

        result = runner.run(
            model_step=_noop_model_step,
            num_episodes=2,
            episode_id_prefix="ttt",
        )

        assert result.episode_ids == ["ttt-000000", "ttt-000001"]

    def test_resume_skips_already_collected(self, tmp_path: Path):
        """Running twice on the same dir with resume=True (default) must
        pick up where the last run left off."""
        factory = _FakeFactory()
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )
        first = runner.run(model_step=_noop_model_step, num_episodes=2)
        assert first.num_collected == 2

        # Second runner instance, same output dir, asks for 5 episodes total.
        factory2 = _FakeFactory()
        runner2 = CollectRunner(
            session_factory=factory2,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )
        second = runner2.run(model_step=_noop_model_step, num_episodes=5)

        assert second.num_collected == 3
        assert second.num_skipped == 2
        # Only the 3 new episodes should have triggered a factory.create().
        assert len(factory2.created) == 3

    def test_resume_rejects_corrupt_file_before_starting_sessions(self, tmp_path: Path):
        original = b'{"episode_id": "ep-000000"}\n{"episode_id":'
        results_path = tmp_path / "results.jsonl"
        results_path.write_bytes(original)
        factory = _FakeFactory()
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )

        with pytest.raises(ValueError, match=r"results\.jsonl:2: invalid JSON"):
            runner.run(model_step=_noop_model_step, num_episodes=2)

        assert factory.created == []
        assert results_path.read_bytes() == original

    def test_resume_false_forces_full_rerun(self, tmp_path: Path):
        runner = CollectRunner(
            session_factory=_FakeFactory(),
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )
        runner.run(model_step=_noop_model_step, num_episodes=2)

        # Same output dir, resume=False should overwrite and collect 2 again.
        with pytest.warns(RuntimeWarning, match="moved existing results file"):
            runner.run(
                model_step=_noop_model_step,
                num_episodes=2,
                resume=False,
            )

        lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        backups = list(tmp_path.glob("results.*.bak.jsonl"))
        assert len(backups) == 1
        assert len(backups[0].read_text().strip().splitlines()) == 2

    def test_closes_session_after_each_episode(self, tmp_path: Path):
        factory = _FakeFactory()
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
        )

        runner.run(model_step=_noop_model_step, num_episodes=2)

        assert all(session.closed for session in factory.sessions)

    def test_should_keep_filter_drops_episodes(self, tmp_path: Path):
        factory = _FakeFactory(reward_map={0: 1.0, 1: 0.0, 2: 1.0})
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
            tasks=iter([0, 1, 2]),
        )

        result = runner.run(
            model_step=_noop_model_step,
            num_episodes=3,
            should_keep=lambda record: record.reward > 0.0,
        )

        assert result.num_collected == 2
        assert result.num_dropped == 1
        lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_passes_tasks_from_iterable_to_factory(self, tmp_path: Path):
        factory = _FakeFactory()
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
            tasks=iter(["a", "b", "c"]),
        )

        runner.run(model_step=_noop_model_step, num_episodes=3)

        assert [call["task"] for call in factory.created] == ["a", "b", "c"]

    def test_resume_consumes_task_slots_for_skipped_episodes(self, tmp_path: Path):
        first_runner = CollectRunner(
            session_factory=_FakeFactory(),
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
            tasks=iter(["a", "b"]),
        )
        first_runner.run(model_step=_noop_model_step, num_episodes=2)

        resumed_factory = _FakeFactory()
        resumed_runner = CollectRunner(
            session_factory=resumed_factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
            tasks=iter(["a", "b", "c", "d"]),
        )
        resumed_runner.run(model_step=_noop_model_step, num_episodes=4)

        assert [call["task"] for call in resumed_factory.created] == ["c", "d"]
        payloads = [
            json.loads(line)
            for line in (tmp_path / "results.jsonl").read_text().strip().splitlines()
        ]
        assert payloads[2]["episode_id"] == "ep-000002"
        assert payloads[2]["task"] == "c"
        assert payloads[3]["episode_id"] == "ep-000003"
        assert payloads[3]["task"] == "d"

    def test_collect_result_reports_success_rate(self, tmp_path: Path):
        factory = _FakeFactory(reward_map={0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0})
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=_FakeAdapter(),
            serializer=RolloutSerializer(tmp_path),
            tasks=iter([0, 1, 2, 3]),
        )

        result = runner.run(model_step=_noop_model_step, num_episodes=4)

        assert isinstance(result, CollectResult)
        assert result.num_collected == 4
        assert result.success_rate == pytest.approx(0.75)
        assert result.avg_reward == pytest.approx(0.75)

    def test_episode_failure_is_counted_and_collection_continues(
        self,
        tmp_path: Path,
    ):
        class FailingAdapter(_FakeAdapter):
            def run_white_box(
                self,
                model_step: ModelStep,
                session: ResourceSession,
                limits: HarnessRunLimits | None = None,
            ) -> HarnessRolloutResult:
                if getattr(session, "task") == "bad":
                    raise RuntimeError("episode failed")
                return super().run_white_box(model_step, session, limits)

        factory = _FakeFactory()
        runner = CollectRunner(
            session_factory=factory,
            harness_adapter=FailingAdapter(),
            serializer=RolloutSerializer(tmp_path),
            tasks=iter(["ok-1", "bad", "ok-2"]),
        )

        result = runner.run(model_step=_noop_model_step, num_episodes=3)

        assert result.num_collected == 2
        assert result.num_failed == 1
        assert all(session.closed for session in factory.sessions)
        payloads = [
            json.loads(line)
            for line in (tmp_path / "results.jsonl").read_text().strip().splitlines()
        ]
        assert [payload["task"] for payload in payloads] == ["ok-1", "ok-2"]


class _RecordingLLMClient(LLMClient):
    """Captures the args passed to complete_with_tools and returns a canned response."""

    def __init__(self, response: LLMResponse):
        super().__init__(endpoint="http://recorder", port=0)
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "tools": tools, "sampling": dict(kwargs)}
        )
        return self._response


class _PooledHttpLLMClient(LLMClient):
    """Holds one pooled `httpx.AsyncClient` across calls, like the provider SDKs do."""

    def __init__(self, url: str):
        super().__init__(endpoint=url, port=None)
        self._client = httpx.AsyncClient()

    async def complete(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        response = await self._client.post(self.endpoint, json={"messages": messages})
        return LLMResponse(content=response.json()["content"], tool_calls=[])


class _OkHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"content": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestBuildModelStep:
    def _tool(self) -> Tool:
        return Tool(
            name="play_move",
            description="Play a move.",
            input_schema={
                "type": "object",
                "properties": {"action_id": {"type": "integer"}},
                "required": ["action_id"],
            },
        )

    def test_returns_callable_matching_model_step_protocol(self):
        client = _RecordingLLMClient(LLMResponse(content="ok", tool_calls=[]))
        step = build_model_step(client)

        result = step([{"role": "user", "content": "hi"}], [], {})

        assert isinstance(result, ModelStepResult)
        assert result.response.content == "ok"

    def test_converts_tools_to_mcp_dict_shape(self):
        client = _RecordingLLMClient(LLMResponse(content="", tool_calls=[]))
        step = build_model_step(client)

        step([{"role": "user", "content": "go"}], [self._tool()], {})

        tools = client.calls[0]["tools"]
        assert tools[0]["name"] == "play_move"
        assert tools[0]["description"] == "Play a move."
        assert tools[0]["inputSchema"]["properties"]["action_id"]["type"] == "integer"

    def test_forwards_messages_verbatim(self):
        client = _RecordingLLMClient(LLMResponse(content="", tool_calls=[]))
        step = build_model_step(client)

        messages = [
            {"role": "user", "content": "play 0"},
            {"role": "assistant", "content": "ok"},
        ]
        step(messages, [], {})

        assert client.calls[0]["messages"] == messages

    def test_prepends_system_prompt_when_missing(self):
        client = _RecordingLLMClient(LLMResponse(content="", tool_calls=[]))
        step = build_model_step(client, system_prompt="You are a TTT expert.")

        step([{"role": "user", "content": "play"}], [], {})

        sent = client.calls[0]["messages"]
        assert sent[0] == {"role": "system", "content": "You are a TTT expert."}
        assert sent[1]["role"] == "user"

    def test_does_not_double_prepend_system_prompt(self):
        client = _RecordingLLMClient(LLMResponse(content="", tool_calls=[]))
        step = build_model_step(client, system_prompt="A")

        step(
            [
                {"role": "system", "content": "pre-existing"},
                {"role": "user", "content": "play"},
            ],
            [],
            {},
        )

        sent = client.calls[0]["messages"]
        system_roles = [m for m in sent if m["role"] == "system"]
        assert len(system_roles) == 1
        assert system_roles[0]["content"] == "pre-existing"

    def test_forwards_sampling_kwargs(self):
        client = _RecordingLLMClient(LLMResponse(content="", tool_calls=[]))
        step = build_model_step(client)

        with pytest.warns(
            RuntimeWarning,
            match="Dropping unsupported sampling keys: unknown",
        ):
            step(
                [{"role": "user", "content": "go"}],
                [],
                {"temperature": 0.7, "max_tokens": 50, "unknown": "drop"},
            )

        sampling = client.calls[0]["sampling"]
        assert sampling["temperature"] == 0.7
        assert sampling["max_tokens"] == 50
        # Unknown keys should NOT leak to the client (provider SDKs reject them).
        assert "unknown" not in sampling

    def test_passes_through_tool_calls(self):
        response = LLMResponse(
            content="Playing cell 0.",
            tool_calls=[ToolCall(id="abc", name="play_move", args={"action_id": 0})],
        )
        client = _RecordingLLMClient(response)
        step = build_model_step(client)

        result = step([{"role": "user", "content": "play"}], [self._tool()], {})

        assert len(result.response.tool_calls) == 1
        assert result.response.tool_calls[0].name == "play_move"
        assert result.response.tool_calls[0].args == {"action_id": 0}

    @pytest.mark.asyncio
    async def test_runs_safely_inside_existing_event_loop(self):
        client = _RecordingLLMClient(LLMResponse(content="ok", tool_calls=[]))
        step = build_model_step(client)

        result = step([{"role": "user", "content": "play"}], [], {})

        assert result.response.content == "ok"
        assert client.calls[0]["messages"] == [{"role": "user", "content": "play"}]

    def test_reuses_pooled_client_across_steps(self):
        # Regression: a fresh event loop per step broke the client's connection
        # pool, bound to the first loop, with "Event loop is closed".
        server = ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = _PooledHttpLLMClient(f"http://127.0.0.1:{server.server_port}")
            step = build_model_step(client)

            for _ in range(3):
                result = step([{"role": "user", "content": "play"}], [], {})
                assert result.response.content == "ok"
        finally:
            server.shutdown()
            server.server_close()

    def test_repeated_builds_share_one_loop_thread(self):
        client = _RecordingLLMClient(LLMResponse(content="ok", tool_calls=[]))
        threads_before = threading.active_count()

        for _ in range(5):
            step = build_model_step(client)
            assert step([], [], {}).response.content == "ok"

        assert threading.active_count() <= threads_before + 1


def _populated_output_dir(tmp_path: Path) -> Path:
    """Write a minimal results.jsonl + metadata.json into tmp_path."""
    serializer = RolloutSerializer(tmp_path)
    serializer.write_metadata({"env_id": "tic_tac_toe", "num_episodes": 1})
    serializer.write_episode(
        EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
    )
    return tmp_path


class TestPushToHfHub:
    def test_creates_dataset_repo_and_uploads(self, tmp_path: Path):
        output = _populated_output_dir(tmp_path)
        mock_api = MagicMock()

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            url = push_to_hf_hub(output, "user/ttt-sft-v1")

        mock_api.create_repo.assert_called_once()
        create_kwargs = mock_api.create_repo.call_args.kwargs
        assert create_kwargs["repo_id"] == "user/ttt-sft-v1"
        assert create_kwargs["repo_type"] == "dataset"
        assert create_kwargs["exist_ok"] is True

        mock_api.upload_folder.assert_called_once()
        upload_kwargs = mock_api.upload_folder.call_args.kwargs
        assert upload_kwargs["repo_id"] == "user/ttt-sft-v1"
        assert upload_kwargs["repo_type"] == "dataset"
        assert Path(upload_kwargs["folder_path"]) == output

        assert url == "https://huggingface.co/datasets/user/ttt-sft-v1"

    def test_private_flag_propagates(self, tmp_path: Path):
        output = _populated_output_dir(tmp_path)
        mock_api = MagicMock()

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            push_to_hf_hub(output, "user/private-ds", private=True)

        assert mock_api.create_repo.call_args.kwargs["private"] is True

    def test_custom_commit_message(self, tmp_path: Path):
        output = _populated_output_dir(tmp_path)
        mock_api = MagicMock()

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            push_to_hf_hub(output, "user/ds", commit_message="seed: 200 TTT games")

        assert (
            mock_api.upload_folder.call_args.kwargs["commit_message"]
            == "seed: 200 TTT games"
        )

    def test_default_commit_message_mentions_episodes(self, tmp_path: Path):
        output = _populated_output_dir(tmp_path)
        mock_api = MagicMock()

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            push_to_hf_hub(output, "user/ds")

        msg = mock_api.upload_folder.call_args.kwargs["commit_message"]
        # Default message includes episode count so the Hub history is informative.
        assert "1" in msg and "episode" in msg.lower()

    def test_errors_when_results_jsonl_missing(self, tmp_path: Path):
        # Empty dir — no results.jsonl written.
        with pytest.raises(FileNotFoundError):
            push_to_hf_hub(tmp_path, "user/ds")

    def test_token_forwarded_to_hfapi(self, tmp_path: Path):
        output = _populated_output_dir(tmp_path)
        mock_api = MagicMock()
        api_cls = MagicMock(return_value=mock_api)

        with patch("huggingface_hub.HfApi", api_cls):
            push_to_hf_hub(output, "user/ds", token="hf_xxx")

        assert api_cls.call_args.kwargs["token"] == "hf_xxx"

    def test_writes_readme_with_viewer_frontmatter(self, tmp_path: Path):
        """HF Dataset Viewer needs YAML front-matter to know which file is the
        dataset — otherwise it tries to cast ``metadata.json`` into the same
        schema as ``results.jsonl`` and errors out."""
        output = _populated_output_dir(tmp_path)
        mock_api = MagicMock()

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            push_to_hf_hub(output, "user/ds")

        readme = (output / README_FILENAME).read_text(encoding="utf-8")
        assert readme.startswith("---")
        assert "configs:" in readme
        assert 'path: "results.jsonl"' in readme
        assert "split: train" in readme


class TestBuildDatasetReadme:
    def test_includes_episode_count(self, tmp_path: Path):
        _populated_output_dir(tmp_path)

        readme = build_dataset_readme(tmp_path)

        assert "Episodes:** 1" in readme

    def test_includes_metadata_table(self, tmp_path: Path):
        _populated_output_dir(tmp_path)
        # overwrite metadata with a known payload
        (tmp_path / "metadata.json").write_text(
            json.dumps({"provider": "openai", "model": "gpt-5-mini"})
        )

        readme = build_dataset_readme(tmp_path)

        assert "`provider`" in readme
        assert "`openai`" in readme
        assert "`model`" in readme
        assert "`gpt-5-mini`" in readme

    def test_resilient_to_missing_metadata(self, tmp_path: Path):
        serializer = RolloutSerializer(tmp_path)
        serializer.write_episode(
            EpisodeRecord.from_rollout("ep1", _fake_rollout(), _fake_verify())
        )
        # No metadata.json written.

        readme = build_dataset_readme(tmp_path)

        assert "OpenEnv rollouts" in readme
        assert "Schema" in readme
