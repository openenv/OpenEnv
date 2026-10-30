# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for deterministic OpenEnv environment import."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from openenv.cli.__main__ import app
from openenv.cli.importers.agentenv import detect_agentenv_bundles
from openenv.cli.importers.base import safe_vendor_dir_name
from openenv.cli.importers.ors import detect_ors_dependencies, detect_ors_environments
from openenv.cli.importers.verifiers import detect_verifiers_environments
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.mcp_client import MCPToolClient
from packaging.requirements import Requirement
from typer.testing import CliRunner


runner = CliRunner()


def _write_fake_ors_sdk(root: Path) -> None:
    ors_dir = root / "ors"
    ors_dir.mkdir(parents=True)
    (ors_dir / "__init__.py").write_text(
        "from .environment import Environment, ListToolsOutput, Split, TextBlock, "
        "ToolOutput, ToolSpec\n",
        encoding="utf-8",
    )
    (ors_dir / "environment.py").write_text(
        """
class _Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class Split(_Model):
    pass


class ToolSpec(_Model):
    pass


class ListToolsOutput(_Model):
    pass


class TextBlock(_Model):
    def __init__(self, text, detail=None, type="text"):
        super().__init__(text=text, detail=detail, type=type)


class ToolOutput(_Model):
    pass


class _RunToolSuccess:
    ok = True

    def __init__(self, output):
        self.output = output


class RunToolOutput:
    def __init__(self, output):
        self.root = _RunToolSuccess(output)


class Environment:
    def __init__(self, task_spec=None, secrets=None):
        self.task_spec = task_spec or {}
        self.secrets = secrets or {}
        self.setup_called = False
        self.teardown_called = False

    def setup(self):
        self.setup_called = True

    def teardown(self):
        self.teardown_called = True
""".lstrip(),
        encoding="utf-8",
    )


def _write_single_fake_ors_env(root: Path) -> None:
    _write_fake_ors_sdk(root)
    (root / "demo_env.py").write_text(
        """
from ors import Environment, ListToolsOutput, Split, TextBlock, ToolOutput, ToolSpec


class DemoEnvironment(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split):
        return [{"id": "alpha", "goal": "answer"}]

    @classmethod
    def num_tasks(cls, split):
        return 1

    @classmethod
    def get_task(cls, split, index):
        return cls.list_tasks(split)[index]

    @classmethod
    def get_task_range(cls, split, start=None, stop=None):
        return cls.list_tasks(split)[slice(start, stop)]

    @classmethod
    def list_tools(cls):
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="answer",
                    description="Submit an answer",
                    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                )
            ]
        )

    def list_task_tools(self):
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="hint",
                    description="Get a hint",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )

    def get_prompt(self):
        return [TextBlock(text=f"Task: {self.task_spec['id']}")]

    def _call_tool(self, name, input):
        return __import__("ors.environment").environment.RunToolOutput(
            ToolOutput(
                blocks=[TextBlock(text=f"{name}:{input.get('value', '')}")],
                metadata={"tool": name},
                reward=1.0,
                finished=True,
            )
        )
""".lstrip(),
        encoding="utf-8",
    )


def _write_fake_verifiers_sdk(root: Path) -> None:
    verifiers_dir = root / "verifiers"
    verifiers_dir.mkdir(parents=True)
    (verifiers_dir / "__init__.py").write_text(
        """
class Environment:
    pass


class Rubric:
    async def score_rollout(self, state):
        answer = state.get("answer") or state.get("task", {}).get("answer")
        completion = state.get("completion") or []
        text = completion[-1].get("content", "") if completion else ""
        state["reward"] = 1.0 if answer and answer in text else 0.0
        state["metrics"] = {"contains_answer": state["reward"]}


class SingleTurnEnv(Environment):
    def __init__(self, dataset, eval_dataset=None, rubric=None):
        self._dataset = dataset
        self._eval_dataset = eval_dataset or dataset
        self.rubric = rubric or Rubric()

    def get_dataset(self):
        return self._dataset

    def get_eval_dataset(self):
        return self._eval_dataset
""".lstrip(),
        encoding="utf-8",
    )


def _write_single_fake_verifiers_env(root: Path) -> None:
    _write_fake_verifiers_sdk(root)
    (root / "vf_demo.py").write_text(
        """
import verifiers as vf


def load_environment() -> vf.Environment:
    train = [
        {"prompt": [{"role": "user", "content": "Say alpha"}], "answer": "alpha", "example_id": 0},
        {"prompt": [{"role": "user", "content": "Say beta"}], "answer": "beta", "example_id": 1},
    ]
    eval_rows = [
        {"prompt": [{"role": "user", "content": "Say gamma"}], "answer": "gamma", "example_id": 0}
    ]
    return vf.SingleTurnEnv(dataset=train, eval_dataset=eval_rows)
""".lstrip(),
        encoding="utf-8",
    )


def _write_agentenv_bundle(root: Path) -> None:
    (root / "envs" / "echo").mkdir(parents=True)
    (root / "envs" / "echo" / "server.py").write_text(
        'raise RuntimeError("should not import")\n', encoding="utf-8"
    )
    (root / "tasks").mkdir()
    steps = [
        {"id": "env", "type": "deploy_env", "env_id": "echo"},
        {"id": "policy", "type": "deploy_agent", "agent_name": "policy"},
        {"id": "play", "type": "prompt_agent", "agent_name": "policy"},
    ]
    (root / "tasks" / "ping.json").write_text(json.dumps(steps), encoding="utf-8")


def _write_fake_agentenv_sdk(root: Path) -> None:
    package = root / "agent_env"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "openenv.py").write_text(
        """
from fastapi import FastAPI

served = []


def build_app(bundle, max_concurrent_envs=8):
    served.append((bundle, max_concurrent_envs))
    return FastAPI()
""".lstrip(),
        encoding="utf-8",
    )


def test_ors_detector_finds_environment_class_without_importing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "envs").mkdir()
    (source / "envs" / "sample.py").write_text(
        """
from ors import Environment as ORSEnvironment

SIDE_EFFECT = 0


class SampleEnv(ORSEnvironment):
    pass
""".lstrip(),
        encoding="utf-8",
    )

    matches = detect_ors_environments(source)

    assert len(matches) == 1
    assert matches[0].class_name == "SampleEnv"
    assert matches[0].module_path == "envs.sample"
    assert matches[0].source_type == "ors"


def test_ors_detector_finds_openreward_environments_import_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.py").write_text(
        """
from openreward.environments import Environment


class SampleEnv(Environment):
    pass
""".lstrip(),
        encoding="utf-8",
    )

    matches = detect_ors_environments(source)

    assert len(matches) == 1
    assert matches[0].class_name == "SampleEnv"
    assert detect_ors_dependencies(source) == ["openreward"]


def test_ors_detector_returns_no_matches_for_unrelated_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "plain.py").write_text("class Plain: pass\n", encoding="utf-8")

    assert detect_ors_environments(source) == []


def test_verifiers_detector_finds_load_environment_without_importing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "demo.py").write_text(
        """
import verifiers as vf

SIDE_EFFECT = 0


def load_environment() -> vf.Environment:
    raise RuntimeError("should not import")
""".lstrip(),
        encoding="utf-8",
    )

    matches = detect_verifiers_environments(source)

    assert len(matches) == 1
    assert matches[0].source_type == "verifiers"
    assert matches[0].class_name == "load_environment"
    assert matches[0].module_path == "demo"


def test_import_command_requires_env_class_when_multiple_ors_classes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text(
        "from ors import Environment\nclass First(Environment): pass\n",
        encoding="utf-8",
    )
    (source / "b.py").write_text(
        "from ors import Environment\nclass Second(Environment): pass\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import",
            str(source),
            "--name",
            "imported_env",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Multiple environment entrypoints" in result.output
    assert "env" in result.output
    assert "class" in result.output


def test_import_command_detects_ors_and_generates_working_wrapper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_single_fake_ors_env(source)
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "imported_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "imported_env"
    assert (env_dir / "server" / "imported_env_environment.py").exists()
    assert (env_dir / "vendor" / "source" / "demo_env.py").exists()
    assert (env_dir / "vendor" / "source" / "ors" / "environment.py").exists()

    sys.path.insert(0, str(output_dir))
    try:
        from imported_env.server.imported_env_environment import (  # type: ignore
            ImportedEnvironment,
        )

        env = ImportedEnvironment()
        assert env.list_splits() == [{"name": "train", "type": "train"}]
        assert env.get_task("train", 0) == {"id": "alpha", "goal": "answer"}
        with pytest.raises(RuntimeError, match="reset"):
            ImportedEnvironment().step(
                CallToolAction(tool_name="answer", arguments={"value": "42"})
            )

        reset_obs = env.reset(split="train", index=0)
        assert reset_obs.metadata["task_spec"] == {"id": "alpha", "goal": "answer"}
        assert reset_obs.metadata["prompt"][0]["text"] == "Task: alpha"

        tools_obs = env.step(ListToolsAction())
        assert [tool.name for tool in tools_obs.tools] == ["answer", "hint"]

        call_obs = env.step(
            CallToolAction(tool_name="answer", arguments={"value": "42"})
        )
        assert call_obs.reward == 1.0
        assert call_obs.done is True
        assert call_obs.result["blocks"][0]["text"] == "answer:42"

        from imported_env.server.app import app as generated_app  # type: ignore

        client = TestClient(generated_app)
        assert client.get("/list_environments").json() == ["imported_env"]
        assert client.get("/imported_env/splits").status_code == 200
        mcp_tools = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1,
            },
        ).json()
        assert mcp_tools["result"]["tools"][0]["name"] == "answer"
        mcp_call = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "answer", "arguments": {"value": "42"}},
                "id": 2,
            },
        ).json()
        assert "reset" in mcp_call["error"]["message"]
    finally:
        sys.path.remove(str(output_dir))


def test_imported_environment_usage_example_runs(tmp_path: Path) -> None:
    example_dir = Path(__file__).resolve().parents[2] / "examples/imported_environment"
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(example_dir / "source"),
                "--name",
                "imported_ors_demo",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output

    spec = importlib.util.spec_from_file_location(
        "imported_environment_usage_example",
        example_dir / "use_imported_environment.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.run_imported_environment(output_dir) == {
        "prompt": "What is 2 + 2?",
        "tools": ["answer"],
        "result": "correct",
        "reward": 1.0,
        "done": True,
    }


def test_import_command_handles_source_module_matching_generated_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_fake_ors_sdk(source)
    (source / "collision_env.py").write_text(
        """
from ors import Environment, Split


class CollisionEnvironment(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "collision_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    sys.path.insert(0, str(output_dir))
    try:
        from collision_env.server.collision_env_environment import (  # type: ignore
            CollisionEnvironment,
        )

        env = CollisionEnvironment()
        assert env.list_splits() == [{"name": "train", "type": "train"}]
    finally:
        sys.path.remove(str(output_dir))


def test_import_command_excludes_common_secret_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_single_fake_ors_env(source)
    (source / "data").mkdir()
    (source / "data" / "prompt.txt").write_text("public task data\n", encoding="utf-8")
    (source / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (source / "secrets.yaml").write_text("token: secret\n", encoding="utf-8")
    (source / "private.pem").write_text("secret\n", encoding="utf-8")
    (source / "native.so").write_text("binary\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "secret_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    vendor_dir = output_dir / "secret_env" / "vendor" / "source"
    assert (vendor_dir / "data" / "prompt.txt").read_text(encoding="utf-8") == (
        "public task data\n"
    )
    assert not (vendor_dir / ".env").exists()
    assert not (vendor_dir / "secrets.yaml").exists()
    assert not (vendor_dir / "private.pem").exists()
    assert not (vendor_dir / "native.so").exists()


def test_safe_vendor_dir_name_avoids_normalization_collision(tmp_path: Path) -> None:
    hyphen_source = tmp_path / "my-source"
    underscore_source = tmp_path / "my_source"
    hyphen_source.mkdir()
    underscore_source.mkdir()

    assert safe_vendor_dir_name(underscore_source) == "my_source"
    assert safe_vendor_dir_name(hyphen_source).startswith("my_source_")
    assert safe_vendor_dir_name(hyphen_source) != safe_vendor_dir_name(
        underscore_source
    )


def test_import_command_uses_detected_ors_dependency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "demo.py").write_text(
        """
from openreward.environments import Environment


class DemoEnvironment(Environment):
    pass
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "openreward_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    requirements = (
        output_dir / "openreward_env" / "server" / "requirements.txt"
    ).read_text(encoding="utf-8")
    pyproject = (output_dir / "openreward_env" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "openreward" in requirements
    assert "openreward" in pyproject
    assert "ors-sdk" not in requirements
    assert "ors-sdk" not in pyproject


def test_import_command_carries_source_dependencies_and_vendor_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_single_fake_ors_env(source)
    (source / "pyproject.toml").write_text(
        """
[project]
name = "source-env"
dependencies = [
    "numpy>=1.26",
]
""".lstrip(),
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text(
        """
# Runtime dependencies for the source env
datasets>=2.19  # reads dataset files
-r dev-requirements.txt
./local-package
""".lstrip(),
        encoding="utf-8",
    )
    (source / "fixtures").mkdir()
    (source / "fixtures" / "tasks.jsonl").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "dependency_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "dependency_env"
    requirements = (env_dir / "server" / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (env_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert "numpy>=1.26" in requirements
    assert "datasets>=2.19" in requirements
    assert "numpy>=1.26" in pyproject
    assert "datasets>=2.19" in pyproject
    assert "vendor/**/*" in pyproject
    assert (env_dir / "vendor" / "source" / "fixtures" / "tasks.jsonl").exists()


def test_import_command_detects_verifiers_and_generates_working_wrapper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vf_source"
    source.mkdir()
    _write_single_fake_verifiers_env(source)
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "vf_imported_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "vf_imported_env"
    assert (env_dir / "server" / "vf_imported_env_environment.py").exists()
    assert (env_dir / "vendor" / "vf_source" / "vf_demo.py").exists()
    assert (env_dir / "vendor" / "vf_source" / "verifiers" / "__init__.py").exists()

    sys.modules.pop("verifiers", None)
    sys.path.insert(0, str(output_dir))
    try:
        from vf_imported_env.server.vf_imported_env_environment import (  # type: ignore
            VfImportedEnvironment,
        )

        env = VfImportedEnvironment()
        assert env.list_splits() == [
            {"name": "train", "type": "train"},
            {"name": "eval", "type": "validation"},
        ]
        with pytest.raises(RuntimeError, match="reset"):
            VfImportedEnvironment().step(
                CallToolAction(tool_name="submit", arguments={"completion": "alpha"})
            )
        assert env.num_tasks("train") == 2
        assert env.get_task("train", 1)["answer"] == "beta"

        reset_obs = env.reset(split="train", index=0)
        assert reset_obs.metadata["prompt"][0]["content"] == "Say alpha"

        tools_obs = env.step(ListToolsAction())
        assert [tool.name for tool in tools_obs.tools] == ["submit"]

        call_obs = env.step(
            CallToolAction(tool_name="submit", arguments={"completion": "alpha"})
        )
        assert call_obs.reward == 1.0
        assert call_obs.done is True
        assert call_obs.result["reward"] == 1.0

        from vf_imported_env.server.app import app as generated_app  # type: ignore

        client = TestClient(generated_app)
        assert client.get("/list_environments").json() == ["vf_imported_env"]
        assert client.get("/vf_imported_env/splits").status_code == 200
    finally:
        sys.path.remove(str(output_dir))
        sys.modules.pop("verifiers", None)


def test_verifiers_wrapper_logs_state_initialization_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "vf_harness_source"
    source.mkdir()
    verifiers_dir = source / "verifiers"
    verifiers_dir.mkdir()
    (verifiers_dir / "__init__.py").write_text(
        """
class Environment:
    pass


class State(dict):
    @classmethod
    def for_task(cls, task):
        raise RuntimeError("state construction failed")
""".lstrip(),
        encoding="utf-8",
    )
    (source / "vf_harness.py").write_text(
        """
import verifiers as vf


class Taskset:
    def rows(self):
        return [{"prompt": "Say alpha", "answer": "alpha", "example_id": 0}]

    def task(self, row):
        return dict(row)

    def to_task(self, row):
        return dict(row)


class Harness:
    def score_group(self, tasks, states):
        states[0]["reward"] = 0.5
        states[0]["metrics"] = {"scored": True}


class HarnessEnvironment(vf.Environment):
    def __init__(self):
        self.taskset = Taskset()
        self.harness = Harness()


def load_environment() -> vf.Environment:
    return HarnessEnvironment()
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "vf_harness_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    sys.modules.pop("verifiers", None)
    sys.path.insert(0, str(output_dir))
    try:
        from vf_harness_env.server.vf_harness_env_environment import (  # type: ignore
            VfHarnessEnvironment,
        )

        env = VfHarnessEnvironment()
        env.reset(split="train", index=0)
        with caplog.at_level(logging.ERROR):
            observation = env.step(
                CallToolAction(tool_name="submit", arguments={"completion": "alpha"})
            )

        assert observation.reward == 0.5
        assert "Failed to initialize Verifiers State for task" in caplog.text
        assert "state construction failed" in caplog.text
    finally:
        sys.path.remove(str(output_dir))
        sys.modules.pop("verifiers", None)


@pytest.mark.parametrize(
    "write_source", [_write_single_fake_ors_env, _write_single_fake_verifiers_env]
)
def test_imported_server_main_parses_host_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_source
) -> None:
    """Test that the generated `server` entry point honors --host and --port."""
    import uvicorn

    source = tmp_path / "source"
    source.mkdir()
    write_source(source)
    env_name = f"port_{write_source.__name__.strip('_')}"
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                env_name,
                "--output-dir",
                str(output_dir),
            ],
        )
    assert result.exit_code == 0, result.output

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["server", "--host", "127.0.0.1", "--port", "8001"]
    )
    monkeypatch.delitem(sys.modules, "verifiers", raising=False)
    monkeypatch.syspath_prepend(str(output_dir))

    importlib.import_module(f"{env_name}.server.app").main()

    assert calls == [{"host": "127.0.0.1", "port": 8001}]


def test_agentenv_detector_finds_bundles_without_importing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_agentenv_bundle(source / "bundles" / "desk")
    _write_agentenv_bundle(source / ".venv" / "bundles" / "installed")
    (source / "notes" / "tasks").mkdir(parents=True)
    (source / "notes" / "tasks" / "todo.json").write_text(
        '{"title": "not a task"}', encoding="utf-8"
    )
    (source / "jobs" / "tasks").mkdir(parents=True)
    (source / "jobs" / "tasks" / "queue.json").write_text(
        '[{"type": "build"}, {"type": "deploy"}]', encoding="utf-8"
    )

    matches = detect_agentenv_bundles(source)

    assert [(m.source_type, m.class_name, m.module_path) for m in matches] == [
        ("agentenv", "desk", "bundles/desk")
    ]


def test_import_command_detects_agentenv_bundle_and_generates_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    source = tmp_path / "echo-desk"
    _write_agentenv_bundle(source)
    (source / "requirements.txt").write_text(
        "numpy>=1.26\nagentenv-framework\n", encoding="utf-8"
    )
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "desk_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "desk_env"
    bundle = env_dir / "vendor" / "echo-desk"
    assert (bundle / "tasks" / "ping.json").exists()
    assert not (env_dir / "server" / "desk_env_environment.py").exists()
    pyproject = (env_dir / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (env_dir / "server" / "requirements.txt").read_text(encoding="utf-8")
    for text in (pyproject, requirements):
        assert "agentenv-framework[openenv]>=0.9.1366" in text
        assert text.count("agentenv-framework") == 1
        assert "numpy>=1.26" in text
    assert 'requires-python = ">=3.11.4"' in pyproject
    assert "vendor/**/*" in pyproject

    sdk = tmp_path / "sdk"
    _write_fake_agentenv_sdk(sdk)
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["server", "--port", "8001"])
    monkeypatch.setenv("MAX_CONCURRENT_ENVS", "64")
    sys.path[:0] = [str(sdk), str(output_dir)]
    try:
        generated = importlib.import_module("desk_env.server.app")
        served = importlib.import_module("agent_env.openenv").served

        assert served == [(bundle.resolve(), 64)]
        package = importlib.import_module("desk_env")
        assert issubclass(package.DeskEnv, MCPToolClient)
        assert package.__all__ == ["DeskEnv", "CallToolAction", "ListToolsAction"]
        assert TestClient(generated.app).get("/docs").status_code == 200
        generated.main()
        assert calls == [{"host": "0.0.0.0", "port": 8001}]
    finally:
        sys.path.remove(str(sdk))
        sys.path.remove(str(output_dir))
        for name in list(sys.modules):
            if name.split(".", 1)[0] in {"agent_env", "desk_env"}:
                sys.modules.pop(name)


@pytest.mark.parametrize("dependency_file", ["requirements.txt", "pyproject.toml"])
@pytest.mark.parametrize(
    "dependency",
    [
        "agentenv-framework @ https://example.invalid/agentenv_framework-0.9.1000-py3-none-any.whl",
        "agentenv-framework @ https://example.invalid/agentenv_framework-0.9.1400-py3-none-any.whl",
        "agentenv-framework @ https://example.invalid/framework.tar.gz",
        "agentenv-framework @ git+https://example.invalid/framework.git@v0.9.1400",
        "agentenv-framework @ file:///tmp/agentenv_framework-0.9.1400-py3-none-any.whl",
        'AgentEnv_Framework[tools] @ https://user:secret@example.invalid/framework.whl ; python_version >= "3.11"',
    ],
)
def test_import_command_rejects_agentenv_direct_references(
    tmp_path: Path, dependency_file: str, dependency: str
) -> None:
    source = tmp_path / "bundle"
    _write_agentenv_bundle(source)
    content = (
        "[project]\ndependencies = " + json.dumps([dependency]) + "\n"
        if dependency_file == "pyproject.toml"
        else dependency + "\n"
    )
    (source / dependency_file).write_text(content, encoding="utf-8")
    output_dir = tmp_path / "out"

    with (
        patch("openenv.cli.commands.import_env._generate_uv_lock") as lock,
        patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
    ):
        result = runner.invoke(
            app,
            ["import", str(source), "--name", "desk_env", "-o", str(output_dir)],
        )

    assert result.exit_code == 1, result.output
    assert "Cannot verify agentenv-framework>=0.9.1366" in result.output
    assert "direct reference" in result.output
    assert "version constraint" in result.output
    assert "secret" not in result.output
    lock.assert_not_called()
    assert not (output_dir / "desk_env").exists()


@pytest.mark.parametrize("dependency_file", ["requirements.txt", "pyproject.toml"])
@pytest.mark.parametrize("version_range", [">=0.9.1000,<1", ">=0.9.1400"])
def test_import_command_merges_agentenv_floor_preserving_extras_and_marker(
    tmp_path: Path, dependency_file: str, version_range: str
) -> None:
    import tomllib

    source = tmp_path / "bundle"
    _write_agentenv_bundle(source)
    dependency = f'agentenv-framework[tools]{version_range}; python_version >= "3.11"'
    content = (
        "[project]\ndependencies = " + json.dumps([dependency]) + "\n"
        if dependency_file == "pyproject.toml"
        else dependency + "\n"
    )
    (source / dependency_file).write_text(content, encoding="utf-8")
    output_dir = tmp_path / "out"

    with (
        patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True),
        patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
    ):
        result = runner.invoke(
            app,
            ["import", str(source), "--name", "desk_env", "-o", str(output_dir)],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "desk_env"
    project = tomllib.loads((env_dir / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = (env_dir / "server" / "requirements.txt").read_text(encoding="utf-8")
    for dependencies in (project["project"]["dependencies"], requirements.splitlines()):
        matches = [
            Requirement(dep) for dep in dependencies if "agentenv-framework" in dep
        ]
        assert len(matches) == 1
        requirement = matches[0]
        assert requirement.url is None
        assert requirement.extras == {"tools", "openenv"}
        assert requirement.marker == Requirement(dependency).marker
        assert (
            requirement.specifier
            == Requirement(f"agentenv-framework{version_range},>=0.9.1366").specifier
        )
        assert "0.9.1000" not in requirement.specifier


def test_import_command_vendors_only_the_agentenv_bundle_it_picks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plugin"
    _write_agentenv_bundle(source / "bundles" / "train")
    _write_agentenv_bundle(source / "bundles" / "eval")
    (source / "bundles" / "train" / "requirements.txt").write_text(
        "agentenv-framework>=0.9.1400\nnumpy>=1.26\n", encoding="utf-8"
    )
    (source / "requirements.txt").write_text("torch>=2.0\n", encoding="utf-8")
    (source / ".agentenv").mkdir()
    (source / ".agentenv" / "config.toml").write_text("[stores]\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    args = ["import", str(source), "--name", "picked_env", "--output-dir"]

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        ambiguous = runner.invoke(app, [*args, str(output_dir)])
        picked = runner.invoke(app, [*args, str(output_dir), "--env-class", "train"])

    assert ambiguous.exit_code != 0
    assert "Multiple environment entrypoints" in ambiguous.output
    assert picked.exit_code == 0, picked.output
    env_dir = output_dir / "picked_env"
    vendored = sorted(
        path.relative_to(env_dir / "vendor").as_posix()
        for path in (env_dir / "vendor").rglob("*")
        if path.is_file()
    )
    assert vendored == [
        "__init__.py",
        "train/__init__.py",
        "train/envs/echo/server.py",
        "train/requirements.txt",
        "train/tasks/ping.json",
    ]
    app_source = (env_dir / "server" / "app.py").read_text(encoding="utf-8")
    assert '"vendor" / "train"' in app_source
    requirements = (env_dir / "server" / "requirements.txt").read_text(encoding="utf-8")
    assert "agentenv-framework[openenv]>=0.9.1366,>=0.9.1400" in requirements
    assert "numpy>=1.26" in requirements
    assert "torch" not in requirements


@pytest.mark.parametrize(
    ("bundle_name", "env_name"), [("echo-desk", "desk_env"), ("train", "picked_env")]
)
def test_agentenv_generated_readme_matches_paths_and_prerequisites(
    tmp_path: Path, bundle_name: str, env_name: str
) -> None:
    import re

    import tomli

    source = tmp_path / bundle_name
    _write_agentenv_bundle(source)
    output_dir = tmp_path / "out"
    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                env_name,
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / env_name
    readme = (env_dir / "README.md").read_text(encoding="utf-8")
    # Check the published file inventory against the output, including the bundle
    # whose name differs from the package and may contain a hyphen.
    inventory = readme.split("## Generated files\n", 1)[1]
    paths = re.findall(r"`([^`]+)`", inventory)
    assert f"vendor/{bundle_name}" in paths
    assert "server/app.py" in paths
    for path in paths:
        assert (env_dir / path).exists(), path
    assert f"{env_name}_environment.py" not in readme
    assert "from_docker_image" not in readme
    assert "echoed_message" not in readme

    project = tomli.loads((env_dir / "pyproject.toml").read_text())["project"]
    assert (
        "uv sync\nMAX_CONCURRENT_ENVS=8 uv run server --host 127.0.0.1 --port 8000"
        in readme
    )
    assert project["scripts"]["server"] == f"{env_name}.server.app:main"
    assert project["requires-python"] == ">=3.11.4"
    assert "Python >=3.11.4" in readme
    requirement = "agentenv-framework[openenv]>=0.9.1366"
    assert requirement in project["dependencies"]
    assert requirement in (env_dir / "server/requirements.txt").read_text()
    assert requirement in readme
    assert "agent_env.openenv.build_app" in readme
    adapter = "https://github.com/scaleapi/agentenv-framework/blob/v0.9.1366/src/agent_env/openenv"
    assert f"({adapter}/environment.py)" in readme
    assert f"({adapter}/app.py)" in readme
    client = {"desk_env": "DeskEnv", "picked_env": "PickedEnv"}[env_name]
    assert f"from {env_name} import {client}" in readme
    assert f"class {client}(MCPToolClient):" in (env_dir / "client.py").read_text()
    assert not (env_dir / "models.py").exists()
    assert "scaffolding" not in readme
    for text in (readme, result.output):
        normalized = " ".join(text.split())
        assert "Docker CLI and a running daemon accessible to the server" in normalized
        assert "openenv build" in text and "openenv push" in text
        assert "do not produce a working deployment" in normalized
    assert "server/Dockerfile" in result.output
    assert "Warning:" in result.output
