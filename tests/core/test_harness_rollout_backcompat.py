# SPDX-License-Identifier: BSD-3-Clause

"""Back-compat guarantees for the openenv.core.harness package split."""

from __future__ import annotations

import importlib

import openenv.core.harness as harness_pkg
from openenv.core.harness import rollout

ROLLOUT_PUBLIC_NAMES = [
    "CLIHarnessAdapter",
    "HarnessAdapter",
    "HarnessRolloutResult",
    "HarnessRunLimits",
    "LoopOwningSession",
    "MCPHarnessAdapter",
    "Message",
    "ModelStep",
    "ModelStepResult",
    "RESERVED_TOOL_NAMES",
    "ResourceSession",
    "ResourceSessionFactory",
    "RolloutEvent",
    "SessionMCPBridge",
    "StepEnvSessionAdapter",
    "ToolResult",
    "ToolTraceEntry",
    "TraceEntry",
    "TrainableSession",
    "VerifyResult",
    "build_harness_rollout_func",
]


def test_rollout_all_matches_expected_names():
    assert sorted(rollout.__all__) == sorted(ROLLOUT_PUBLIC_NAMES)


def test_rollout_names_reexported_from_package_root():
    for name in ROLLOUT_PUBLIC_NAMES:
        assert name in harness_pkg.__all__
        assert getattr(harness_pkg, name) is getattr(rollout, name)


def test_private_resolve_env_reward_reexported():
    # tests/scripts/test_browsergym_harness_eval_examples.py imports this
    # private helper from the package root; keep it importable.
    assert harness_pkg._resolve_env_reward is rollout._resolve_env_reward


def test_collect_modules_import():
    importlib.import_module("openenv.core.harness.collect")
    importlib.import_module("openenv.cli.commands.collect")


def test_adapter_abcs_are_distinct():
    assert harness_pkg.AgenticHarnessAdapter is not harness_pkg.HarnessAdapter
    assert "AgenticHarnessAdapter" in harness_pkg.__all__
    assert "HarnessAdapter" in harness_pkg.__all__
