# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the kernrl environment."""

import os

import pytest
from kernrl.server.kernrl_environment import KernelOptEnvironment


class TestResetSeed:
    """reset(seed=...) must make the problem order reproducible.

    Uses the problems shipped with the env and no GPU, so it runs in CI.
    """

    @staticmethod
    def _problem_ids_after_seed(env, seed, n=5):
        """One seeded reset followed by n - 1 unseeded ones."""
        problem_ids = [env.reset(seed=seed).problem_id]
        problem_ids += [env.reset().problem_id for _ in range(n - 1)]
        return problem_ids

    def test_same_seed_same_sequence_across_instances(self):
        first = self._problem_ids_after_seed(KernelOptEnvironment(), seed=1234)
        second = self._problem_ids_after_seed(KernelOptEnvironment(), seed=1234)
        assert first == second

    def test_reseeding_restarts_the_sequence(self):
        env = KernelOptEnvironment()
        first = self._problem_ids_after_seed(env, seed=1234)
        second = self._problem_ids_after_seed(env, seed=1234)
        assert first == second

    def test_different_seeds_give_different_sequences(self):
        env = KernelOptEnvironment()
        first = self._problem_ids_after_seed(env, seed=1)
        second = self._problem_ids_after_seed(env, seed=2)
        assert first != second

    def test_episode_id_is_kept(self):
        env = KernelOptEnvironment()
        env.reset(episode_id="episode-1")
        assert env.state.episode_id == "episode-1"
        env.reset(episode_id="")
        assert env.state.episode_id == ""
        env.reset()
        assert env.state.episode_id not in (None, "", "episode-1")


def test_solution_subprocesses_get_no_server_secrets(monkeypatch):
    from kernrl.server.profiler import solution_env

    monkeypatch.setenv("HF_TOKEN", "hf_FAKEtoken0000000000000000000000000")
    monkeypatch.setenv("MY_API_KEY", "fake-key")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    env = solution_env()

    assert "HF_TOKEN" not in env
    assert "MY_API_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PATH"] == os.environ["PATH"]


_REFERENCE = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2


def get_inputs():
    return [torch.ones(16, 1024)]
"""

_SOLUTION_TEMPLATE = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        RETURN_LINE
"""


@pytest.mark.parametrize(
    ("solution", "correct"),
    [
        ("return x * 2", True),
        ("return torch.tensor(2.0)", False),
        ("return x[0] * 2", False),
        ("return (x * 2)[:, :1]", False),
    ],
    ids=["same shape", "scalar", "one row", "one column"],
)
def test_correctness_rejects_outputs_of_another_shape(tmp_path, solution, correct):
    """An output of another shape must not pass by broadcasting."""
    pytest.importorskip("torch")
    from kernrl.server.evaluator import LocalGPUEvaluator

    reference_path = tmp_path / "reference.py"
    reference_path.write_text(_REFERENCE)
    solution_path = tmp_path / "solution.py"
    solution_path.write_text(_SOLUTION_TEMPLATE.replace("RETURN_LINE", solution))

    result = LocalGPUEvaluator(device="cpu")._check_correctness(
        solution_path, reference_path, tmp_path
    )

    assert result.correct is correct
    if not correct:
        assert "does not match the reference shape" in result.error
