# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``openenv collect`` CLI command."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openenv.cli.__main__ import app
from openenv.cli.commands.collect import _extract_legal_actions
from typer.testing import CliRunner

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def mock_pipeline():
    """Patch the collect pipeline internals so tests don't hit network or disk."""
    with (
        patch("openenv.cli.commands.collect.OpenSpielEnv") as env_cls,
        patch("openenv.cli.commands.collect.OpenSpielSessionFactory") as factory_cls,
        patch("openenv.cli.commands.collect.CollectRunner") as runner_cls,
        patch("openenv.cli.commands.collect.RolloutSerializer") as serializer_cls,
        patch("openenv.cli.commands.collect.MCPHarnessAdapter") as adapter_cls,
    ):
        collect_result = MagicMock()
        collect_result.num_collected = 3
        collect_result.num_skipped = 0
        collect_result.num_dropped = 0
        collect_result.num_failed = 0
        collect_result.avg_reward = 0.6
        collect_result.success_rate = 0.6
        collect_result.episode_ids = ["ep-000000", "ep-000001", "ep-000002"]

        runner_instance = MagicMock()
        runner_instance.run.return_value = collect_result
        runner_cls.return_value = runner_instance

        yield {
            "env_cls": env_cls,
            "factory_cls": factory_cls,
            "runner_cls": runner_cls,
            "runner_instance": runner_instance,
            "serializer_cls": serializer_cls,
            "adapter_cls": adapter_cls,
        }


def test_scripted_provider_requires_no_llm_client(tmp_path: Path, mock_pipeline):
    with patch("openenv.cli.commands.collect.create_llm_client") as create_client:
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "-n",
                "3",
                "--provider",
                "scripted",
            ],
        )

    assert result.exit_code == 0, result.output
    create_client.assert_not_called()
    mock_pipeline["runner_instance"].run.assert_called_once()


def test_openai_provider_builds_llm_client(tmp_path: Path, mock_pipeline):
    llm_client = MagicMock()
    with (
        patch(
            "openenv.cli.commands.collect.create_llm_client", return_value=llm_client
        ) as create_client,
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
    ):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "-n",
                "5",
                "--provider",
                "openai",
                "--model",
                "gpt-5-mini",
            ],
        )

    assert result.exit_code == 0, result.output
    create_client.assert_called_once()
    kwargs = create_client.call_args.kwargs
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["api_key"] == "sk-test"


def test_llm_endpoint_uses_llm_teacher_with_default_provider(
    tmp_path: Path, mock_pipeline
):
    llm_step = MagicMock(name="llm_step")
    scripted_step = MagicMock(name="scripted_step")

    with (
        patch(
            "openenv.cli.commands.collect._build_llm_model_step",
            return_value=llm_step,
        ) as build_llm,
        patch(
            "openenv.cli.commands.collect._build_scripted_model_step",
            return_value=scripted_step,
        ) as build_scripted,
    ):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "--llm-endpoint",
                "http://localhost",
                "--model",
                "Qwen/Qwen2.5-7B-Instruct",
            ],
        )

    assert result.exit_code == 0, result.output
    build_llm.assert_called_once()
    build_scripted.assert_not_called()
    assert (
        mock_pipeline["runner_instance"].run.call_args.kwargs["model_step"] is llm_step
    )
    metadata = mock_pipeline[
        "serializer_cls"
    ].return_value.write_metadata.call_args.args[0]
    assert metadata["provider"] == "openai-compatible"
    assert metadata["llm_endpoint"] == "http://localhost"


def test_push_to_hub_triggers_upload(tmp_path: Path, mock_pipeline):
    with (
        patch("openenv.cli.commands.collect.push_to_hf_hub") as push_mock,
    ):
        push_mock.return_value = "https://huggingface.co/datasets/user/ttt"
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "-n",
                "3",
                "--provider",
                "scripted",
                "--push-to-hub",
                "user/ttt",
                "--private",
            ],
        )

    assert result.exit_code == 0, result.output
    push_mock.assert_called_once()
    push_kwargs = push_mock.call_args.kwargs
    assert push_kwargs["repo_id"] == "user/ttt"
    assert push_kwargs["private"] is True


def test_unknown_env_id_exits_nonzero(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "collect",
            "unknown:foo",
            "--base-url",
            "http://example",
            "--output-dir",
            str(tmp_path),
            "--provider",
            "scripted",
        ],
    )

    assert result.exit_code != 0
    assert (
        "unknown" in result.output.lower() or "not supported" in result.output.lower()
    )


def test_openai_provider_errors_without_model(tmp_path: Path, mock_pipeline):
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "--provider",
                "openai",
            ],
        )

    assert result.exit_code != 0
    assert "model" in result.output.lower()


def test_extract_legal_actions_from_json_observation():
    actions = _extract_legal_actions(
        [
            {"content": "ignored"},
            {"content": '{"legal_actions": [2, 4, 6]}'},
        ]
    )

    assert actions == [2, 4, 6]


def test_extract_legal_actions_from_text_observation():
    actions = _extract_legal_actions(
        [
            {"content": "Legal actions: [1, 3, 5]"},
        ]
    )

    assert actions == [1, 3, 5]


def test_keep_losses_disables_filter(tmp_path: Path, mock_pipeline):
    result = runner.invoke(
        app,
        [
            "collect",
            "openspiel:tic_tac_toe",
            "--base-url",
            "https://example.hf.space",
            "--output-dir",
            str(tmp_path),
            "--provider",
            "scripted",
            "--keep-losses",
        ],
    )

    assert result.exit_code == 0, result.output
    run_kwargs = mock_pipeline["runner_instance"].run.call_args.kwargs
    # When --keep-losses is passed, no filter should be installed.
    assert run_kwargs.get("should_keep") is None


def test_default_filters_losing_rollouts(tmp_path: Path, mock_pipeline):
    result = runner.invoke(
        app,
        [
            "collect",
            "openspiel:tic_tac_toe",
            "--base-url",
            "https://example.hf.space",
            "--output-dir",
            str(tmp_path),
            "--provider",
            "scripted",
        ],
    )

    assert result.exit_code == 0, result.output
    run_kwargs = mock_pipeline["runner_instance"].run.call_args.kwargs
    should_keep = run_kwargs.get("should_keep")
    assert should_keep is not None
    # Sanity: a winning record should be kept, a losing one dropped.
    winning = MagicMock(reward=1.0)
    losing = MagicMock(reward=-1.0)
    assert should_keep(winning) is True
    assert should_keep(losing) is False


@pytest.mark.parametrize(
    ("llm_args", "expected_base_url"),
    [
        (["--llm-endpoint", "http://localhost:8000"], "http://localhost:8000/v1"),
        (["--llm-endpoint", "http://localhost:8000/v1"], "http://localhost:8000/v1"),
        (
            ["--llm-endpoint", "http://localhost", "--llm-port", "8001"],
            "http://localhost:8001/v1",
        ),
        (["--llm-endpoint", "http://localhost:11434"], "http://localhost:11434/v1"),
        (["--llm-endpoint", "http://localhost"], "http://localhost/v1"),
        (["--llm-endpoint", "http://gw/openai/v1"], "http://gw/openai/v1"),
    ],
)
def test_llm_endpoint_url_forms_reach_openai_client(
    tmp_path: Path, mock_pipeline, llm_args, expected_base_url
):
    with patch("openenv.core.llm_client.AsyncOpenAI") as openai_cls:
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "--model",
                "Qwen/Qwen3-1.7B",
                *llm_args,
            ],
        )

    assert result.exit_code == 0, result.output
    openai_cls.assert_called_once()
    assert openai_cls.call_args.kwargs["base_url"] == expected_base_url


@pytest.mark.parametrize(
    ("llm_args", "expected_message"),
    [
        (
            ["--llm-endpoint", "http://localhost:8000:8000"],
            "Invalid endpoint URL",
        ),
        (
            ["--llm-endpoint", "http://localhost:11434", "--llm-port", "8000"],
            "conflicts with port=8000",
        ),
        (["--llm-endpoint", "ftp://localhost:8000"], "expected an http"),
        (
            ["--llm-endpoint", "http://localhost", "--llm-port", "99999"],
            "out of range 1-65535",
        ),
        (
            ["--llm-endpoint", "http://localhost:8000/v1?api-version=1"],
            "query strings and fragments are not supported",
        ),
        (
            ["--llm-endpoint", "http://user:s3cret@localhost:8000"],
            "credentials in the URL are not supported",
        ),
    ],
)
def test_bad_llm_endpoint_is_usage_error_before_output_is_written(
    tmp_path: Path, mock_pipeline, llm_args, expected_message
):
    result = runner.invoke(
        app,
        [
            "collect",
            "openspiel:tic_tac_toe",
            "--base-url",
            "https://example.hf.space",
            "--output-dir",
            str(tmp_path),
            "--model",
            "Qwen/Qwen3-1.7B",
            *llm_args,
        ],
    )

    # Typer renders usage errors in a wrapped box, coloured when GITHUB_ACTIONS
    # or FORCE_COLOR is set; strip the colour codes and flatten before matching.
    plain = _ANSI_ESCAPE.sub("", result.output)
    output = " ".join(plain.replace("\u2502", " ").split())
    assert result.exit_code == 2, result.output
    assert "--llm-endpoint" in output
    assert expected_message in output
    assert "s3cret" not in result.output
    mock_pipeline["serializer_cls"].return_value.write_metadata.assert_not_called()
    mock_pipeline["runner_instance"].run.assert_not_called()


def test_resolved_llm_endpoint_is_printed(tmp_path: Path, mock_pipeline):
    with (
        patch("openenv.core.llm_client.AsyncOpenAI"),
        patch("openenv.cli.commands.collect.console") as console,
    ):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "--model",
                "Qwen/Qwen3-1.7B",
                "--llm-endpoint",
                "http://localhost",
            ],
        )

    assert result.exit_code == 0, result.output
    printed = [str(call.args[0]) for call in console.print.call_args_list]
    assert "[cyan]LLM endpoint:[/cyan] http://localhost" in printed
