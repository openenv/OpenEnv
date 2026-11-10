# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only contract tests: python -B -m unittest discover -s examples/hf_validation_lora."""

import ast
import asyncio
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openenv.core import env_client
from websockets.asyncio.server import serve

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


class ResetSensitiveClient:
    def __init__(self):
        self.resets = []
        self.actions = []
        self.closed = False
        self.error = None

    def sync(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        self.closed = True
        self.error = error

    def reset(self, **kwargs):
        self.resets.append(kwargs)
        self.observation = public_observation()
        self.observation.update(
            level="24", task_id=f"{kwargs['split']}-24-{kwargs['index']:04d}"
        )
        self.observation["prompt"] += json.dumps(kwargs, sort_keys=True) + "\n"
        return SimpleNamespace(done=False, observation=self.observation)

    def step(self, action):
        self.actions.append(action)
        # Both the seed and episode id affect the fake environment's raw reward.
        identity = json.dumps(self.resets[-1], sort_keys=True)
        reward = (sum(identity.encode()) % 90 + 1) / 100
        return SimpleNamespace(
            done=True,
            reward=reward if action["expression"] else 0.0,
            observation={
                "result": {"valid": action["expression"] == "10+13-3+4"},
            },
        )


class DirectResetTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(env_client, "ws_connect", env_client.ws_connect))
        self.clients = []

        def new_client(**kwargs):
            client = ResetSensitiveClient()
            self.clients.append(client)
            return client

        client_patch = patch(
            "openenv.core.generic_client.GenericEnvClient", side_effect=new_client
        )
        client_patch.start()
        self.addCleanup(client_patch.stop)
        spec = importlib.util.spec_from_file_location(
            "scottwu_score_batch", Path(__file__).with_name("score_batch.py")
        )
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)
        self.remote = train.RemoteEnvironment(Mock(), "direct")

    def bridge_output(self, payload):
        output = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            redirect_stdout(output),
        ):
            self.bridge.main()
        return json.loads(output.getvalue())

    def test_direct_resets_match_bridge_prompts_and_raw_rewards(self):
        expressions = ["10+13-3+4", "", "invalid"]
        for split, indices in (("train", [0, 7, 0]), ("test", [2, 49, 2])):
            for operation in ("tasks", "score"):
                with self.subTest(split=split, operation=operation):
                    expected = self.bridge_output(
                        dict(
                            operation=operation,
                            split=split,
                            indices=indices,
                            expressions=expressions,
                        )
                    )
                    bridge_client = self.clients[-1]
                    if operation == "tasks":
                        actual = self.remote.tasks(split, indices)
                    else:
                        actual = self.remote.score(split, indices, expressions)
                    direct_client = self.clients[-1]
                    self.assertTrue(bridge_client.closed)
                    self.assertTrue(direct_client.closed)
                    self.assertIsNone(direct_client.error)
                    # Compare with the executed bridge, not a copied reset recipe.
                    self.assertEqual(direct_client.resets, bridge_client.resets)
                    if operation == "tasks":
                        self.assertEqual(
                            [row["prompt"] for row in actual],
                            [row["prompt"] for row in expected["tasks"]],
                        )
                        self.assertEqual([row["index"] for row in actual], indices)
                        self.assertEqual([row["split"] for row in actual], [split] * 3)
                        self.assertNotIn("private", json.dumps(actual))
                    else:
                        self.assertEqual(direct_client.actions, bridge_client.actions)
                        self.assertEqual(
                            direct_client.actions,
                            [{"expression": expression} for expression in expressions],
                        )
                        self.assertEqual(
                            [row["reward"] for row in actual], expected["rewards"]
                        )
                        self.assertEqual(
                            [row["valid"] for row in actual], expected["valid"]
                        )

    def test_direct_client_closes_when_reset_or_step_raises(self):
        for operation, failure in (
            ("tasks", "reset"),
            ("score", "reset"),
            ("score", "step"),
        ):
            with (
                self.subTest(operation=operation, failure=failure),
                patch.object(
                    ResetSensitiveClient,
                    failure,
                    side_effect=RuntimeError("client failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "client failed") as error:
                    if operation == "tasks":
                        self.remote.tasks("test", [3])
                    else:
                        self.remote.score("test", [3], [""])
                self.assertTrue(self.clients[-1].closed)
                self.assertIs(self.clients[-1].error, error.exception)

    def test_direct_applies_the_bridge_reset_contract(self):
        original_reset = ResetSensitiveClient.reset

        def broken_reset(change):
            def reset(client, **kwargs):
                result = original_reset(client, **kwargs)
                result.done = change.get("done", False)
                client.observation.update(
                    {k: v for k, v in change.items() if k != "done"}
                )
                return result

            return reset

        for change in ({"done": True}, {"level": "year"}, {"solution": "10+13-3+4"}):
            for operation in ("tasks", "score"):
                payload = dict(
                    operation=operation, split="test", indices=[3], expressions=[""]
                )
                with (
                    self.subTest(change=change, operation=operation),
                    patch.object(ResetSensitiveClient, "reset", broken_reset(change)),
                ):
                    with self.assertRaisesRegex(RuntimeError, "(?i)reset contract"):
                        self.bridge_output(payload)
                    with self.assertRaisesRegex(RuntimeError, "(?i)reset contract"):
                        if operation == "tasks":
                            self.remote.tasks("test", [3])
                        else:
                            self.remote.score("test", [3], [""])


class DirectRedirectTests(unittest.IsolatedAsyncioTestCase):
    async def test_handshake_redirect_does_not_replay_sandbox_credentials(self):
        self.enterContext(patch.object(env_client, "ws_connect", env_client.ws_connect))
        headers = {
            "Authorization": "Bearer synthetic-hf-token",
            "X-Sandbox-Token": "synthetic-sandbox-token",
        }
        for cross_origin in (True, False):
            with self.subTest(cross_origin=cross_origin):
                initial_headers, redirected_headers = [], []

                async def close(websocket):
                    await websocket.close()

                def collect(connection, request):
                    redirected_headers.append(
                        (
                            request.headers.get("x-sandbox-token"),
                            request.headers.get("authorization"),
                        )
                    )
                    return connection.respond(HTTPStatus.FORBIDDEN, "rejected")

                def redirect(connection, request):
                    if request.path == "/destination":
                        return collect(connection, request)
                    initial_headers.append(
                        (
                            request.headers.get("x-sandbox-token"),
                            request.headers.get("authorization"),
                        )
                    )
                    response = connection.respond(
                        HTTPStatus.TEMPORARY_REDIRECT, "redirect"
                    )
                    response.headers["Location"] = location
                    return response

                async with serve(
                    close, "127.0.0.1", 0, process_request=collect
                ) as destination:
                    port = destination.sockets[0].getsockname()[1]
                    location = (
                        f"ws://127.0.0.1:{port}/destination"
                        if cross_origin
                        else "/destination"
                    )
                    async with serve(
                        close, "127.0.0.1", 0, process_request=redirect
                    ) as upstream:
                        port = upstream.sockets[0].getsockname()[1]
                        url = f"http://127.0.0.1:{port}"
                        sandbox = SimpleNamespace(
                            proxy_url_for=lambda port, path: url + path,
                            proxy_headers=headers,
                        )
                        with self.assertRaises(ConnectionError):
                            await asyncio.to_thread(
                                train.RemoteEnvironment(sandbox, "direct").tasks,
                                "train",
                                [0],
                            )
                self.assertEqual(
                    initial_headers,
                    [("synthetic-sandbox-token", "Bearer synthetic-hf-token")],
                )
                self.assertEqual(redirected_headers, [])


class OptimizerWindowTests(unittest.TestCase):
    def setUp(self):
        # Execute the production nested callback/reward function without main's
        # GPU/Hub setup. Only external tensor and scoring operations are mocked.
        tree = ast.parse(Path(train.__file__).read_text())
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in {"Evidence", "environment_reward"}
        ]

        class CallbackBase:
            def on_step_begin(self, *args, **kwargs):
                pass

        self.report = {"training_groups": [{"std": 0.5}], "optimizer_evidence": []}
        self.parameter = Mock(requires_grad=True)
        model = SimpleNamespace(
            named_parameters=lambda: [("adapter", self.parameter)],
            parameters=lambda: [self.parameter],
        )
        self.torch = Mock()
        self.torch.isfinite.return_value.all.return_value = True
        self.rewards = [0.0, 0.4, 0.4, 0.4]
        namespace = dict(
            vars(train),
            TrainerCallback=CallbackBase,
            report=self.report,
            model=model,
            torch=self.torch,
            deadline=float("inf"),
            check_time=Mock(),
            by_key={"train-24-0000": {}},
            score_texts=lambda *args: [{"reward": r} for r in self.rewards],
        )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), train.__file__, "exec"),
            namespace,
        )
        self.evidence = namespace["Evidence"]()
        self.reward = namespace["environment_reward"]
        self.args = SimpleNamespace(
            world_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            generation_batch_size=4,
            num_generations=4,
            num_iterations=1,
        )
        self.state = SimpleNamespace(global_step=0)
        self.control = SimpleNamespace(should_training_stop=False)
        self.optimizer = SimpleNamespace(state={}, step_was_skipped=False)

    def begin(self):
        self.evidence.on_step_begin(self.args, self.state, self.control)

    def score_group(self):
        # The real reward callback scores four completions and appends one group.
        return self.reward(["answer"] * 4, ["train-24-0000"] * 4)

    def pre(self, norm=0.5):
        self.torch.stack.return_value.sum.return_value.sqrt.return_value.cpu.return_value = norm
        self.evidence.on_pre_optimizer_step(
            self.args, self.state, self.control, optimizer=self.optimizer
        )

    def finish(self, *, skipped=False, advance=True, changed=True):
        self.optimizer.step_was_skipped = skipped
        if advance:
            previous = self.optimizer.state.get(self.parameter, {}).get("step", 0)
            self.optimizer.state[self.parameter] = {"step": previous + 1}
        self.torch.equal.return_value = not changed
        self.evidence.on_optimizer_step(
            self.args, self.state, self.control, optimizer=self.optimizer
        )
        self.state.global_step += 1  # Trainer advances even after an AMP skip.
        self.evidence.on_step_end(self.args, self.state, self.control)
        return self.report["optimizer_evidence"][-1]

    def test_zero_or_multiple_new_groups_fail_before_optimizer(self):
        for count in (0, 2):
            with self.subTest(new_groups=count):
                self.setUp()
                self.begin()
                for _ in range(count):
                    self.score_group()
                with self.assertRaisesRegex(
                    RuntimeError, "one newly scored GRPO group"
                ):
                    self.pre()
                self.assertEqual(self.report["optimizer_evidence"], [])

    def test_one_group_spans_four_completion_advantages_and_std_is_snapshotted(self):
        self.begin()
        self.assertEqual(self.score_group(), [0.0, 0.4, 0.4, 0.4])
        # Representative buffered advantages enter four singleton backwards;
        # tensor arithmetic is mocked, not a real TRL/backward qualification.
        backward = Mock()
        for advantage in (-1.5, 0.5, 0.5, 0.5):
            backward(advantage)
        self.pre()
        # The post-step observation must use the sealed group, not the tail.
        self.report["training_groups"].append({"std": 0.0})
        event = self.finish()
        self.assertAlmostEqual(event["group_std"], 0.17320508075688773)
        self.assertTrue(train.effective_update(event))
        self.assertEqual(event["attempt"], 1)

    def test_reused_generation_cannot_borrow_the_previous_groups_signal(self):
        self.begin()
        self.score_group()
        self.pre()
        self.assertTrue(train.effective_update(self.finish()))
        self.begin()
        # Buffered generation reuse scores no new group during this attempt.
        with self.assertRaisesRegex(RuntimeError, "one newly scored GRPO group"):
            self.pre()
        self.assertEqual(len(self.report["optimizer_evidence"]), 1)

    def test_amp_skip_and_consecutive_attempts_keep_their_own_group(self):
        events = []
        self.begin()
        self.score_group()
        self.pre(float("inf"))
        events.append(self.finish(skipped=True, advance=False, changed=False))
        self.begin()
        self.rewards = [0.4] * 4
        self.score_group()
        self.pre()
        events.append(self.finish())  # Momentum changes alone do not count.
        self.begin()
        self.rewards = [0.0, 0.4, 0.4, 0.4]
        self.score_group()
        self.pre()
        events.append(self.finish())
        self.assertEqual(
            [train.effective_update(e) for e in events], [False, False, True]
        )
        self.assertEqual([e["attempt"] for e in events], [1, 2, 3])
        self.assertIsNone(events[0]["grad_norm"])
        self.assertFalse(events[0]["counter_advanced"])
        self.assertEqual(events[0]["changed_tensors"], 0)
        self.assertEqual(self.report["global_step"], 3)
        json.dumps(self.report, allow_nan=False)

    def test_unsupported_recipe_fails_at_step_begin(self):
        for key, value in (
            ("world_size", 2),
            ("per_device_train_batch_size", 4),
            ("gradient_accumulation_steps", 8),
            ("generation_batch_size", 16),
            ("num_generations", 8),
            ("num_iterations", 2),
        ):
            with self.subTest(argument=key):
                self.setUp()
                setattr(self.args, key, value)
                with self.assertRaisesRegex(RuntimeError, "frozen single-group recipe"):
                    self.begin()


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
