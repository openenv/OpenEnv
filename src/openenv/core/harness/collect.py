# SPDX-License-Identifier: BSD-3-Clause

"""Rollout collection helpers for synthetic dataset generation.

Complements ``openenv.core.harness`` with:

- ``EpisodeRecord`` — serializable view of one rollout + its verification.
- ``RolloutSerializer`` — append-only JSONL writer with metadata sidecar.
- ``CollectRunner`` — orchestrates repeated rollouts, supports resume, and
  optional per-record filtering via ``should_keep``.

The serialized schema is designed to be consumed directly by TRL's
``SFTTrainer`` (``messages`` column) or by ``datasets.Dataset.from_json``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator

from ..env_server.mcp_types import Tool
from ..llm_client import LLMClient
from .rollout import (
    _resolve_env_reward,
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    ModelStep,
    ModelStepResult,
    ResourceSessionFactory,
    VerifyResult,
)

_SUPPORTED_SAMPLING_KEYS = frozenset({"temperature", "max_tokens", "top_p"})

RESULTS_FILENAME = "results.jsonl"
METADATA_FILENAME = "metadata.json"


def _tool_trace_entry_to_plain(entry: Any) -> dict[str, Any]:
    return {
        "tool_name": entry.tool_name,
        "arguments": dict(entry.arguments),
        "result": {
            "data": entry.result.data,
            "done": entry.result.done,
            "metadata": dict(entry.result.metadata),
            "error": entry.result.error,
        },
    }


def _tool_trace_to_plain(rollout: HarnessRolloutResult) -> list[dict[str, Any]]:
    return [_tool_trace_entry_to_plain(entry) for entry in rollout.tool_trace]


@dataclass
class EpisodeRecord:
    """Serializable view of one collected episode."""

    episode_id: str
    messages: list[dict[str, Any]]
    reward: float
    done: bool
    tool_trace: list[dict[str, Any]]
    metrics: dict[str, Any]
    verify_metrics: dict[str, Any]
    artifacts: dict[str, Any]
    task: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rollout(
        cls,
        episode_id: str,
        rollout: HarnessRolloutResult,
        verify: VerifyResult,
        task: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> "EpisodeRecord":
        """Build a record from a harness rollout and its verification.

        Uses ``_resolve_env_reward`` so that any disagreement between the
        reward emitted inside the environment and the one forwarded by
        ``verify()`` raises — preserving the "rewards in env" invariant.
        """
        reward = _resolve_env_reward(rollout, verify)
        return cls(
            episode_id=episode_id,
            messages=[dict(m) for m in rollout.messages],
            reward=reward,
            done=bool(rollout.done or verify.done),
            tool_trace=_tool_trace_to_plain(rollout),
            metrics=dict(rollout.metrics),
            verify_metrics=dict(verify.metrics),
            artifacts=dict(verify.artifacts),
            task=task,
            extra=dict(extra or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))


class RolloutSerializer:
    """Append-only JSONL writer for collected episodes."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def results_path(self) -> Path:
        return self._output_dir / RESULTS_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self._output_dir / METADATA_FILENAME

    def write_episode(self, record: EpisodeRecord) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record.to_dict(), default=str) + "\n").encode("utf-8")
        with self.results_path.open("a+b") as handle:
            separator = self._append_separator(handle)
            handle.write(separator + line)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

    def collected_episode_ids(self) -> set[str]:
        """Return episode ids already persisted on disk. Used for resume.

        Raises:
            `ValueError`: If a non-empty line is not a valid episode record.
        """
        if not self.results_path.exists():
            return set()

        ids: set[str] = set()
        with self.results_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                ids.add(self._episode_id_from_line(raw, line_number))
        return ids

    def _append_separator(self, handle: BinaryIO) -> bytes:
        """Return a newline when an existing complete final record lacks one."""
        end = handle.seek(0, 2)
        if not end:
            return b""
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return b""

        chunks: list[bytes] = []
        cursor = end
        while cursor:
            read_size = min(cursor, 8192)
            cursor -= read_size
            handle.seek(cursor)
            chunk = handle.read(read_size)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                chunks.append(chunk[newline + 1 :])
                break
            chunks.append(chunk)
        final_line = b"".join(reversed(chunks))
        if final_line.strip():
            self._episode_id_from_line(final_line, "final line")
        return b"\n"

    def _episode_id_from_line(self, raw: bytes, line_number: int | str) -> str:
        """Parse and validate one non-empty JSONL episode record."""
        location = f"{self.results_path}:{line_number}"
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{location}: invalid UTF-8: {exc}") from exc
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{location}: expected an object")
        episode_id = payload.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError(f"{location}: expected a string episode_id")
        return episode_id


def _rollout_final_state(rollout: HarnessRolloutResult) -> dict[str, Any]:
    """Build the structured payload passed to ``session.verify()``.

    Mirrors the shape used by ``build_harness_rollout_func`` so verifiers
    can rely on the same contract regardless of caller (TRL vs. collector).
    """
    return {
        "done": rollout.done,
        "metrics": dict(rollout.metrics),
        "events": [
            {"type": event.type, "payload": dict(event.payload)}
            for event in rollout.events
        ],
        "tool_trace": _tool_trace_to_plain(rollout),
    }


@dataclass
class CollectResult:
    """Summary returned by ``CollectRunner.run()``."""

    num_collected: int
    num_skipped: int
    num_dropped: int
    num_failed: int
    episode_ids: list[str]
    avg_reward: float
    success_rate: float


class CollectRunner:
    """Drive repeated rollouts into an append-only JSONL dataset.

    The runner is stateless across ``run()`` calls — resume state lives on
    disk via the ``RolloutSerializer``. A single runner instance can be
    reused across runs targeting different output dirs.
    """

    def __init__(
        self,
        *,
        session_factory: ResourceSessionFactory,
        harness_adapter: HarnessAdapter,
        serializer: RolloutSerializer,
        tasks: Iterable[Any] | None = None,
        limits: HarnessRunLimits | None = None,
    ):
        self._session_factory = session_factory
        self._harness_adapter = harness_adapter
        self._serializer = serializer
        self._tasks: Iterator[Any] | None = iter(tasks) if tasks is not None else None
        self._limits = limits

    def _next_task(self) -> Any:
        if self._tasks is None:
            return None
        try:
            return next(self._tasks)
        except StopIteration:
            return None

    def run(
        self,
        *,
        model_step: ModelStep,
        num_episodes: int,
        episode_id_prefix: str = "ep",
        resume: bool = True,
        should_keep: Callable[[EpisodeRecord], bool] | None = None,
    ) -> CollectResult:
        planned_ids = [f"{episode_id_prefix}-{i:06d}" for i in range(num_episodes)]

        if resume:
            already = self._serializer.collected_episode_ids()
        else:
            self._reset_results_file()
            already = set()

        num_skipped = 0
        num_collected = 0
        num_dropped = 0
        num_failed = 0
        rewards: list[float] = []

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(
                "collected={task.fields[collected]} reward={task.fields[avg_reward]:.2f}"
            ),
            TimeElapsedColumn(),
        )
        task_id = progress.add_task(
            "Collecting", total=len(planned_ids), collected=0, avg_reward=0.0
        )

        with progress:
            for episode_id in planned_ids:
                # Preserve task-to-episode alignment even when resume skips ids.
                task = self._next_task()
                if episode_id in already:
                    num_skipped += 1
                    progress.advance(task_id)
                    continue

                session = None
                try:
                    session = self._session_factory.create(
                        task=task,
                        episode_id=episode_id,
                    )
                    rollout = self._harness_adapter.run_white_box(
                        model_step=model_step,
                        session=session,
                        limits=self._limits,
                    )
                    verify = session.verify(
                        transcript=rollout.messages,
                        final_state=_rollout_final_state(rollout),
                    )
                    record = EpisodeRecord.from_rollout(
                        episode_id=episode_id,
                        rollout=rollout,
                        verify=verify,
                        task=task,
                    )
                except Exception as exc:
                    num_failed += 1
                    progress.console.print(
                        f"[red]failed[/red] {episode_id}: {type(exc).__name__}: {exc}"
                    )
                    progress.advance(task_id)
                    continue
                finally:
                    if session is not None:
                        session.close()

                if should_keep is not None and not should_keep(record):
                    num_dropped += 1
                    progress.advance(task_id)
                    continue

                self._serializer.write_episode(record)
                num_collected += 1
                rewards.append(record.reward)
                avg = sum(rewards) / len(rewards)
                progress.update(
                    task_id, advance=1, collected=num_collected, avg_reward=avg
                )

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_rate = (
            sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        )

        return CollectResult(
            num_collected=num_collected,
            num_skipped=num_skipped,
            num_dropped=num_dropped,
            num_failed=num_failed,
            episode_ids=planned_ids,
            avg_reward=avg_reward,
            success_rate=success_rate,
        )

    def _reset_results_file(self) -> None:
        if self._serializer.results_path.exists():
            results_path = self._serializer.results_path
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = results_path.with_name(
                f"{results_path.stem}.{timestamp}.bak{results_path.suffix}"
            )
            counter = 1
            while backup_path.exists():
                backup_path = results_path.with_name(
                    f"{results_path.stem}.{timestamp}.{counter}.bak"
                    f"{results_path.suffix}"
                )
                counter += 1
            results_path.replace(backup_path)
            warnings.warn(
                f"resume=False moved existing results file to {backup_path}",
                RuntimeWarning,
                stacklevel=2,
            )


def _tool_to_mcp_dict(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


@functools.cache
def _model_step_loop() -> asyncio.AbstractEventLoop:
    """Start the event loop shared by every model step, on its first use."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def build_model_step(
    llm_client: LLMClient,
    *,
    system_prompt: str | None = None,
) -> ModelStep:
    """Adapt any OpenEnv ``LLMClient`` into a ``ModelStep`` for the harness.

    Works with every provider the ``LLMClient`` hierarchy already covers —
    OpenAI and any OpenAI-compatible endpoint (vLLM, TGI, Ollama, HF
    Inference, Together, Groq, Fireworks, ...), Anthropic natively, and
    any future subclass that implements ``complete_with_tools``. Provider
    -specific schema conversion lives inside the client; this helper only
    handles the sync/async adaptation and tool dict shape.
    """
    # Async clients (e.g. `AsyncOpenAI`) bind their connection pool to the loop of
    # the first request, so every call must run on the same long-lived loop. All
    # model steps share one loop, so building one per episode doesn't add threads.
    loop = _model_step_loop()

    def model_step(
        messages: list[dict[str, Any]],
        tools: list[Tool],
        sampling: dict[str, Any],
    ) -> ModelStepResult:
        effective_messages = list(messages)
        if system_prompt and not any(
            m.get("role") == "system" for m in effective_messages
        ):
            effective_messages.insert(0, {"role": "system", "content": system_prompt})

        tool_dicts = [_tool_to_mcp_dict(t) for t in tools]
        dropped_keys = sorted(set(sampling) - _SUPPORTED_SAMPLING_KEYS)
        if dropped_keys:
            warnings.warn(
                "Dropping unsupported sampling keys: " + ", ".join(dropped_keys),
                RuntimeWarning,
                stacklevel=2,
            )
        filtered_sampling = {
            k: v for k, v in sampling.items() if k in _SUPPORTED_SAMPLING_KEYS
        }

        response = asyncio.run_coroutine_threadsafe(
            llm_client.complete_with_tools(
                messages=effective_messages,
                tools=tool_dicts,
                **filtered_sampling,
            ),
            loop,
        ).result()
        return ModelStepResult(response=response)

    return model_step


def _count_episodes(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    with results_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


README_FILENAME = "README.md"

_DATASET_CARD_FRONTMATTER = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: "results.jsonl"
---
"""


def build_dataset_readme(output_dir: Path) -> str:
    """Build the README.md written alongside ``results.jsonl`` on the Hub.

    The YAML front-matter pins ``results.jsonl`` as the only dataset file
    so the HF Dataset Viewer does not try to cast ``metadata.json`` into
    the same schema.
    """
    output_dir = Path(output_dir)
    results_path = output_dir / RESULTS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}

    lines: list[str] = [
        _DATASET_CARD_FRONTMATTER.strip(),
        "",
        "# OpenEnv rollouts",
        "",
        "Collected with [OpenEnv](https://github.com/huggingface/OpenEnv) "
        "(`openenv collect`).",
        "",
        f"- **Episodes:** {_count_episodes(results_path)}",
    ]
    if metadata:
        lines.append("- **Run metadata:**")
        lines.append("")
        lines.append("  | key | value |")
        lines.append("  | --- | --- |")
        for key, value in metadata.items():
            lines.append(f"  | `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Schema",
            "",
            "Each line of `results.jsonl` is one episode:",
            "",
            "- `episode_id` (string)",
            "- `messages` (chat transcript; TRL `SFTTrainer`-compatible)",
            "- `reward` (float)",
            "- `done` (bool)",
            "- `tool_trace` (list of `{tool_name, arguments, result}`)",
            "- `metrics`, `verify_metrics`, `artifacts`",
            "- `task`, `extra` (optional call-site annotations)",
            "",
            "## Load",
            "",
            "```python",
            "from datasets import load_dataset",
            'ds = load_dataset("<user>/<repo>", split="train")',
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def push_to_hf_hub(
    output_dir: Path | str,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload a collected rollout directory to the Hugging Face Hub as a dataset.

    Uploads ``results.jsonl``, ``metadata.json`` (if present), and an
    auto-generated ``README.md`` whose YAML front-matter tells the HF
    Dataset Viewer to treat ``results.jsonl`` as the sole dataset file.
    Consumers load the result with ``load_dataset("<user>/<repo>",
    split="train")``.

    Args:
        output_dir: Directory previously populated by ``RolloutSerializer``.
        repo_id: Destination repo in ``"user/name"`` form.
        private: Create the dataset repo as private.
        commit_message: Override the default commit message.
        token: HF token. Defaults to the token resolved by ``huggingface_hub``.

    Returns:
        The public URL of the dataset on the Hub.

    Raises:
        FileNotFoundError: If ``results.jsonl`` is missing from ``output_dir``.
    """
    output_dir = Path(output_dir)
    results_path = output_dir / RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(
            f"No {RESULTS_FILENAME} found in {output_dir}. "
            "Run a collect job before pushing to the Hub."
        )

    readme_path = output_dir / README_FILENAME
    readme_path.write_text(build_dataset_readme(output_dir), encoding="utf-8")

    num_episodes = _count_episodes(results_path)
    message = commit_message or f"Add {num_episodes} collected episode(s)"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


__all__ = [
    "build_dataset_readme",
    "build_model_step",
    "CollectResult",
    "CollectRunner",
    "EpisodeRecord",
    "push_to_hf_hub",
    "RolloutSerializer",
    "METADATA_FILENAME",
    "README_FILENAME",
    "RESULTS_FILENAME",
]
