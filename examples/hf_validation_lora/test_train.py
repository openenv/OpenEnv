# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only contract tests: python -B -m unittest discover -s examples/hf_validation_lora."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "scottwu_train", Path(__file__).with_name("train.py")
)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def public_observation():
    return {
        "system_prompt": "Use every card once; finish with <answer>expression</answer>.",
        "prompt": "Cards: 10♠ K♥ 3♣ 4♠\nValues: 10, 13, 3, 4\nTarget: 24\n",
        "solution": None,
        "solution_steps": [],
        "metadata": {"private": "never pass this field to the model"},
    }


class TrainingContractTests(unittest.TestCase):
    def test_precreated_private_tracker_gets_an_app(self):
        api = Mock()
        with patch("huggingface_hub.metadata_update") as update:
            train.install_trackio_dashboard(api, "owner/fresh-private")
        update.assert_called_once_with(
            "owner/fresh-private",
            {"hf_oauth": True, "hf_oauth_scopes": ["write-repos"]},
            repo_type="space",
            token=api.token,
            overwrite=True,
        )
        files = {
            call.kwargs["path_in_repo"]: call.kwargs["path_or_fileobj"]
            for call in api.upload_file.call_args_list
        }
        self.assertEqual(files["requirements.txt"], b"trackio==0.45.0\n")
        self.assertIn(b"trackio.show()", files["app.py"])
        for call in api.upload_file.call_args_list:
            self.assertEqual(call.kwargs["repo_id"], "owner/fresh-private")
            self.assertEqual(call.kwargs["repo_type"], "space")

    def test_training_precision_requires_native_bf16(self):
        class Cuda:
            def is_bf16_supported(self, including_emulation=True):
                return including_emulation

        torch = SimpleNamespace(cuda=Cuda(), bfloat16="bf16", float16="fp16")
        self.assertEqual(train.training_precision(torch), "fp16")
        with patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertEqual(train.training_precision(torch), "bf16")

    def test_training_interpreter_needs_its_own_python_headers(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                train.sysconfig, "get_paths", return_value={"include": directory}
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Python.h"):
                train.require_python_headers()
            header = Path(directory) / "Python.h"
            header.write_text("/* matching interpreter header */")
            self.assertEqual(train.require_python_headers(), str(header))

    def test_training_requires_a_working_compiler_before_model_loading(self):
        with (
            patch.dict(train.os.environ, {}, clear=True),
            patch.object(train.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "C compiler"):
                train.require_c_compiler()
        with (
            patch.dict(train.os.environ, {"CC": "chosen-compiler"}, clear=True),
            patch.object(
                train.shutil, "which", return_value="/tools/chosen-compiler"
            ) as lookup,
        ):
            self.assertEqual(train.require_c_compiler(), "/tools/chosen-compiler")
            lookup.assert_called_once_with("chosen-compiler")

    def test_every_parameter_counter_is_checked(self):
        before, after = {1: 7, 2: 3}, {1: 7, 2: 4}
        self.assertTrue(train.counters_advanced(before, after))
        self.assertFalse(train.counters_advanced(before, dict(before)))
        self.assertTrue(train.counters_advanced({}, {1: 1}))
        with self.assertRaises(RuntimeError):
            train.optimizer_observation(
                float("inf"), True, train.counters_advanced(before, after), 0
            )

    def test_effective_updates_require_current_reward_contrast(self):
        event = {**train.optimizer_observation(0.5, False, True, 4), "group_std": 0.0}
        self.assertFalse(train.effective_update(event))
        event["group_std"] = 0.1
        self.assertTrue(train.effective_update(event))

    def test_amp_overflow_requires_a_verified_skipped_step(self):
        event = train.optimizer_observation(float("inf"), True, False, 0)
        self.assertIsNone(event["grad_norm"])
        self.assertFalse(event["gradients_finite"])
        self.assertTrue(event["step_skipped"])
        self.assertNotIn("Infinity", json.dumps(event, allow_nan=False))
        for norm, skipped, advanced, changed in [
            (float("nan"), False, False, 0),
            (float("inf"), False, True, 1),
            (float("inf"), True, True, 0),
            (1.0, True, False, 1),
        ]:
            with (
                self.subTest(args=(norm, skipped, advanced, changed)),
                self.assertRaises(RuntimeError),
            ):
                train.optimizer_observation(norm, skipped, advanced, changed)
        event = train.optimizer_observation(0.5, False, True, 4)
        self.assertTrue(event["gradients_finite"])
        self.assertTrue(event["counter_advanced"])
        self.assertEqual(event["changed_tensors"], 4)

    def test_error_frames_omit_messages_and_local_values(self):
        synthetic_secret = "Bearer synthetic-credential-for-error-test"
        try:
            raise RuntimeError(synthetic_secret)
        except RuntimeError as error:
            frames = train.exception_frames(error)
        self.assertTrue(frames)
        self.assertNotIn(synthetic_secret, json.dumps(frames))
        self.assertEqual(frames[-1]["file"], "test_train.py")
        self.assertEqual(set(frames[-1]), {"file", "line", "function"})

    def test_fresh_private_trackio_cannot_reuse_existing_storage(self):
        info = SimpleNamespace(
            private=True,
            sdk="gradio",
            runtime=SimpleNamespace(volumes=[]),
            siblings=[SimpleNamespace(rfilename="README.md")],
        )
        runtime = SimpleNamespace(volumes=[])
        api = SimpleNamespace(
            space_info=lambda _: info, get_space_runtime=lambda _: runtime
        )
        self.assertTrue(train.require_fresh_private_trackio(api, "owner/new")["fresh"])
        for obj, attribute, value in [
            (info, "private", False),
            (info, "sdk", "static"),
            (runtime, "volumes", [SimpleNamespace(source="owner/public")]),
            (info.runtime, "volumes", [SimpleNamespace(source="owner/public")]),
            (info, "siblings", [SimpleNamespace(rfilename="app.py")]),
        ]:
            original = getattr(obj, attribute)
            setattr(obj, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(RuntimeError):
                train.require_fresh_private_trackio(api, "owner/reused")
            setattr(obj, attribute, original)

    def test_parser_uses_last_answer_outside_thinking(self):
        text = "<think><answer>hidden</answer></think><answer>old</answer><ANSWER> (10+13)-3+4 </ANSWER>"
        self.assertEqual(train.extract_expression(text), "(10+13)-3+4")
        self.assertEqual(train.extract_expression("<think><answer>hidden</answer>"), "")
        self.assertEqual(train.extract_expression("24\n10 + 13 - 3 + 4"), "")
        self.assertEqual(train.extract_expression("<answer>unfinished"), "")

    def test_prompt_allowlist_and_reference_guard(self):
        observation = public_observation()
        messages = train.policy_prompt(observation)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertNotIn("private", json.dumps(messages))
        self.assertNotIn("solution", json.dumps(messages))
        self.assertEqual(train.hand_identity(messages), (3, 4, 10, 13))
        observation["solution"] = "a terminal-only reference"
        with self.assertRaises(ValueError):
            train.policy_prompt(observation)
        observation = public_observation()
        observation["prompt"] = "Values: 1, 2, 3, 4, 5, 6\nTarget: 2026\n"
        with self.assertRaises(ValueError):
            train.policy_prompt(observation)

    def test_pilot_requires_within_group_contrast_and_server_validity(self):
        groups = [[{"reward": 0.0, "valid": True} for _ in range(4)] for _ in range(8)]
        self.assertFalse(train.signal_gate(groups)["passed"])

        # Different task means, each group constant, still provide no GRPO signal.
        groups[0] = [{"reward": 1.0, "valid": True} for _ in range(4)]
        self.assertFalse(train.signal_gate(groups)["passed"])
        groups[0][0]["reward"] = 0.0
        groups[1][0]["reward"] = 0.1
        self.assertTrue(train.signal_gate(groups)["passed"])
        for group in groups:
            for row in group:
                row["valid"] = False
        self.assertFalse(train.signal_gate(groups)["passed"])
        # Frozen practical gate: exactly 8/32 valid terminal expressions passes;
        # 7/32 fails, with the same two contrasting groups in both cases.
        for group in groups[:2]:
            for row in group:
                row["valid"] = True
        gate = train.signal_gate(groups)
        self.assertEqual(gate["minimum_valid_fraction"], 0.25)
        self.assertTrue(gate["passed"])
        groups[1][-1]["valid"] = False
        self.assertFalse(train.signal_gate(groups)["passed"])

    def test_expanded_pilot_keeps_thresholds_and_uses_full_denominator(self):
        groups = [
            [{"reward": 0.0, "valid": False} for _ in range(4)] for _ in range(32)
        ]
        for group in groups[:8]:
            for row in group:
                row["valid"] = True
        groups[0][0]["reward"] = 1.0
        groups[1][0]["reward"] = 0.1
        gate = train.signal_gate(groups, expected_groups=32)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["planned_groups"], 32)
        self.assertEqual(gate["minimum_contrasting_groups"], 2)
        self.assertEqual(gate["valid_fraction"], 0.25)
        groups[-1].pop()
        with self.assertRaises(ValueError):
            train.signal_gate(groups, expected_groups=32)

    def test_exec_bridge_uses_no_proxy_and_preserves_terminal_zeros(self):
        calls = []
        response = {
            "rewards": [1, 0],
            "task_ids": ["train-24-0000"] * 2,
            "valid": [True, False],
        }

        class Sandbox:
            def run(self, command, *, stdin, timeout):
                calls.append((command, json.loads(stdin), timeout))
                return SimpleNamespace(exit_code=0, stdout=json.dumps(response))

            def proxy_url_for(self, *args):
                raise AssertionError("exec transport must never touch proxy")

            @property
            def proxy_headers(self):
                raise AssertionError("exec transport must never read proxy headers")

        remote = train.RemoteEnvironment(Sandbox(), "exec")
        outcomes = remote.score("train", [0, 0], ["10+13-3+4", ""])
        self.assertEqual([o["reward"] for o in outcomes], [1.0, 0.0])
        self.assertFalse(outcomes[1]["valid"])
        self.assertEqual(
            calls[0][0], ["/opt/mathenv/bin/python", "/opt/score_batch.py"]
        )
        self.assertEqual(calls[0][1]["expressions"], ["10+13-3+4", ""])
        self.assertEqual(calls[0][2], 50)
        response["rewards"] = [1, float("nan")]
        with self.assertRaises(RuntimeError):
            remote.score("train", [0, 0], ["x", "y"])
        response["rewards"] = [1, 0]
        response["task_ids"][1] = "test-24-0000"
        with self.assertRaises(RuntimeError):
            remote.score("train", [0, 0], ["x", "y"])

    def test_missing_evaluation_is_unavailable_and_pairs_are_matched(self):
        summary = train.paired_summary([])
        self.assertIsNone(summary["adapter"]["mean_raw_reward"])
        self.assertEqual(summary["baseline"]["planned"], 48)
        traces = [
            {
                "phase": "baseline",
                "episode_key": "test-24-0000",
                "seed": 7001,
                "reward": 0.2,
                "valid": True,
                "exact": False,
            },
            {
                "phase": "adapter",
                "episode_key": "test-24-0000",
                "seed": 7001,
                "reward": 1.0,
                "valid": True,
                "exact": True,
            },
            {
                "phase": "adapter",
                "episode_key": "test-24-0000",
                "seed": 7002,
                "reward": 0.0,
                "valid": False,
                "exact": False,
            },
        ]
        summary = train.paired_summary(traces)
        self.assertEqual(summary["paired"]["completed"], 1)
        self.assertAlmostEqual(summary["paired"]["mean_raw_reward_difference"], 0.8)


if __name__ == "__main__":
    unittest.main()
