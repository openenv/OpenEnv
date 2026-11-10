"""Offline admission tests using a generated toy packaging marker, not benchmark data."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import container_check
import split


def toy_package(task):
    """Generate a disposable native-layout package solely to exercise the tooling."""
    spec = {
        "task_id": task.name,
        "submission_paths": ["/app/marker.py"],
        "negative_cases": ["unchanged"],
        "capability_family": "toy-packaging",
        "skills": ["packaging-marker"],
        "intended_difficulty": "toy-not-a-benchmark",
    }
    contents = {
        "instruction.md": "Toy packaging marker; not a benchmark task.\n",
        "task.toml": 'version = "1.0"\n',
        "spec.json": json.dumps(spec),
        "environment/Dockerfile": f"FROM {split.BASE_IMAGE}\nWORKDIR /app\nCOPY source/ /app/\n",
        "environment/source/marker.py": 'MARKER = "starter"\n',
        "environment/source/not_transferred.txt": "Only declared Python files may cross.\n",
        "solution/solve.sh": 'printf \'MARKER = "ready"\\n\' > "$APP_DIR/marker.py"\n',
        "solution/alternate.sh": 'printf \'MARKER = "ready"\\n\' > "$APP_DIR/marker.py"\n',
        "negatives/unchanged.sh": ": # Toy no-op control.\n",
        "tests/test.sh": """\
            mkdir -p "$VERIFIER_LOG_DIR"
            if "$PYTHON" -B "$TEST_DIR/test_packaging.py"; then
                printf '1\\n' > "$VERIFIER_LOG_DIR/reward.txt"
            else
                printf '0\\n' > "$VERIFIER_LOG_DIR/reward.txt"
            fi
            """,
        "tests/test_packaging.py": """\
            import os
            import runpy
            import unittest
            from pathlib import Path

            class PackagingMarker(unittest.TestCase):
                def test_file_arrived(self):
                    self.assertTrue((Path(os.environ["APP_DIR"]) / "marker.py").is_file())

                def test_marker_ready(self):
                    marker = runpy.run_path(str(Path(os.environ["APP_DIR"]) / "marker.py"))
                    self.assertEqual(marker["MARKER"], "ready")

                def test_undeclared_file_absent(self):
                    self.assertFalse((Path(os.environ["APP_DIR"]) / "not_transferred.txt").exists())

            if __name__ == "__main__":
                unittest.main(verbosity=2)
            """,
    }
    for name, content in contents.items():
        path = task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
    return spec


class SplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task_id = "toy-packaging-marker"
        self.task_root = self.root / "payload"
        self.task = self.task_root / self.task_id
        self.spec = toy_package(self.task)
        self.digest = split.tree_digest(self.task)
        self.expected = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in split.files(self.task).items()
        }
        self.inventory = {
            "schema_version": 1,
            "revision": "toy-packaging-fixture-v1",
            "task_file_hashes": {self.task_id: self.expected},
        }
        self.inventory_path = self.root / "inventory.json"
        split.write_json(self.inventory_path, self.inventory)
        self.image = "registry.example/toy@sha256:" + "a" * 64
        self.image_lock = self.root / "images.json"
        split.write_json(self.image_lock, {self.task_id: self.image})
        self.row = {
            "task_id": self.task_id,
            "task_sha256": self.digest,
            "cases": [
                {
                    "case": name,
                    "task_sha256": self.digest,
                    "reward": reward,
                    "tests_run": 3,
                    "test_ids": ["toy-arrived", "toy-marker", "toy-no-extra-file"],
                    "skipped_tests": 0,
                    "passed_tests": 3 if reward else 2,
                    "failed_tests": 0 if reward else 1,
                }
                for name, _, reward in split.control_cases(self.task, self.spec)
            ],
        }

    def cloud(self, bundle_hashes=None):
        cloud = types.ModuleType("toy_cloud_adapter")
        plan = {"task_source": {"task_file_hashes": copy.deepcopy(self.expected)}}
        manifest = {
            "files": {
                "task/" + name: {"sha256": digest}
                for name, digest in (
                    self.expected if bundle_hashes is None else bundle_hashes
                ).items()
            }
        }
        manifest["files"]["coding-config.json"] = {"sha256": "0" * 64}

        def prepare(task, config, output, repo_root, source=None):
            split.write_json(output / "plan.json", plan)

        cloud.prepare = Mock(side_effect=prepare)
        cloud._read_plan = Mock(return_value=(plan, {}, b"", manifest))
        return cloud

    def prepare(self, cloud, output):
        with patch.object(split, "pipeline_adapter", return_value=cloud):
            split.prepare(
                self.image_lock,
                self.root,
                output,
                self.task_root,
                self.inventory_path,
                "example/private-results",
            )

    def assert_config(self, config):
        self.assertEqual(
            config["target"],
            {
                "model_id": "Qwen/Qwen3.8-27B",
                "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
                "provider": "deepinfra",
                "supports_seed": False,
                "supports_tools": True,
                "temperature": 0.7,
                "max_steps": 48,
                "max_tokens": 4096,
                "max_total_tokens": 32768,
            },
        )
        self.assertEqual(
            config["generator"],
            dict(
                config["target"], max_steps=1, max_tokens=8192, max_total_tokens=32768
            ),
        )
        self.assertEqual(config["execution"]["max_tool_output_bytes"], 1048576)

    def test_real_local_controls_and_package_use_only_external_toy_files(self):
        evidence_path = self.root / "controls.json"
        evidence = split.check(
            evidence_path, sys.executable, self.task_root, self.inventory_path
        )
        self.assertEqual(len(evidence["tasks"][0]["cases"]), 8)
        output = self.root / "package"
        split.build(output, evidence_path, self.task_root, self.inventory_path)
        packaged = output / split.SPLIT
        self.assertEqual(
            split.files(packaged / "tasks" / self.task_id), split.files(self.task)
        )
        self.assertEqual(
            json.loads((packaged / "task-file-hashes.json").read_text()), self.inventory
        )
        config = json.loads(
            (packaged / "configs" / f"{self.task_id}.template.json").read_text()
        )
        self.assert_config(config)
        self.assertNotIn("agent_image", config["execution"])
        self.assertNotIn("results_repo", config["cloud"])
        manifest = json.loads((packaged / "manifest.json").read_text())
        self.assertEqual(manifest["evaluation_protocol"], "qwen38-gap48-v1")
        self.assertFalse(manifest["cloud_audit_ready"])
        for line in (packaged / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
            )

    def test_cli_accepts_external_paths(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(split.ROOT / "split.py"),
                "check",
                "--task-root",
                str(self.task_root),
                "--inventory",
                str(self.inventory_path),
                "--output",
                str(self.root / "cli.json"),
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads((self.root / "cli.json").read_text())["tasks"][0]["task_id"],
            self.task_id,
        )

    def test_complete_controls_required(self):
        split.validate_controls(self.row, self.task, self.spec)
        changes = {
            "reward": ("reward", 0),
            "zero-tests": ("tests_run", 0),
            "skip": ("skipped_tests", 1),
            "hash": ("task_sha256", "0" * 64),
            "boolean-reward": ("reward", True),
            "count-mismatch": ("tests_run", 4),
            "different-suite": ("test_ids", ["other-a", "other-b", "other-c"]),
            "duplicate-test": ("test_ids", ["toy-arrived"] * 3),
        }
        for change in ("missing", "duplicate", "reorder", *changes):
            row = copy.deepcopy(self.row)
            if change == "missing":
                row["cases"].pop()
            elif change == "duplicate":
                row["cases"][-1] = row["cases"][0]
            elif change == "reorder":
                row["cases"].reverse()
            else:
                key, value = changes[change]
                row["cases"][1][key] = value
            with self.subTest(change=change), self.assertRaises(ValueError):
                split.validate_controls(row, self.task, self.spec)

    def refresh_inventory(self):
        self.expected = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in split.files(self.task).items()
        }
        self.inventory["task_file_hashes"][self.task_id] = self.expected
        split.write_json(self.inventory_path, self.inventory)

    def test_real_check_rejects_negative_escape_duplicates_and_reserved_names(self):
        outside = self.root / "outside.sh"
        witness = self.root / "external-script-ran"
        outside.write_text(f"touch '{witness}'\n")
        escape = "../" * 20 + str(outside.with_suffix("")).lstrip("/")
        for names in (
            [escape],
            ["unchanged", "unchanged"],
            ["baseline"],
            ["reference-0"],
        ):
            self.spec["negative_cases"] = names
            split.write_json(self.task / "spec.json", self.spec)
            self.refresh_inventory()
            output = self.root / "controls.json"
            with (
                self.subTest(names=names),
                self.assertRaisesRegex(ValueError, "basenames"),
            ):
                split.check(output, sys.executable, self.task_root, self.inventory_path)
            self.assertFalse(output.exists())
            self.assertFalse(witness.exists())

    def test_negative_script_must_be_a_contained_regular_file(self):
        script = self.task / "negatives/unchanged.sh"
        script.unlink()
        script.symlink_to(self.task / "solution/solve.sh")
        with self.assertRaisesRegex(ValueError, "outside"):
            split.audit_template(self.task, self.spec)

    def test_real_check_rejects_control_selected_test_suites(self):
        verifier = self.task / "tests/test_packaging.py"
        original = verifier.read_text()
        for change in ("count", "identity"):
            chosen = (
                '["test_file_arrived"]'
                if change == "count"
                else (
                    '["test_file_arrived", "test_marker_alias", "test_undeclared_file_absent"]'
                )
            )
            replacement = textwrap.dedent(f"""\
                PackagingMarker.test_marker_alias = PackagingMarker.test_marker_ready
                marker = runpy.run_path(str(Path(os.environ["APP_DIR"]) / "marker.py"))
                names = ["test_file_arrived", "test_marker_ready", "test_undeclared_file_absent"]
                if marker["MARKER"] == "ready":
                    names = {chosen}
                suite = unittest.TestSuite(PackagingMarker(name) for name in names)
                result = unittest.TextTestRunner(verbosity=2).run(suite)
                raise SystemExit(not result.wasSuccessful())
                """)
            verifier.write_text(
                original.replace("unittest.main(verbosity=2)", replacement)
            )
            self.refresh_inventory()
            output = self.root / "controls.json"
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "different test suites"),
            ):
                split.check(output, sys.executable, self.task_root, self.inventory_path)
            self.assertFalse(output.exists())

    def test_frozen_source_detects_live_and_snapshot_changes(self):
        for mutate_snapshot in (False, True):
            target = self.task / "instruction.md"
            original = target.read_bytes()
            with self.subTest(snapshot=mutate_snapshot), self.assertRaises(ValueError):
                with split.frozen_task(self.task, self.expected) as (snapshot, digest):
                    self.assertEqual(split.tree_digest(snapshot), digest)
                    changed = snapshot / "instruction.md" if mutate_snapshot else target
                    changed.write_text("toy changed")
            target.write_bytes(original)

    def test_post_control_drift_cannot_be_certified(self):
        records = iter(copy.deepcopy(self.row["cases"]))

        def trial(task, spec, script, python):
            record = next(records)
            if record["case"] == "unchanged":
                (self.task / "instruction.md").write_text("toy changed")
            return record

        output = self.root / "controls.json"
        with (
            patch.object(split, "trial", trial),
            self.assertRaisesRegex(ValueError, "changed during controls"),
        ):
            split.check(output, sys.executable, self.task_root, self.inventory_path)
        self.assertFalse(output.exists())

    def test_failed_or_stale_evidence_rejected_before_packaging(self):
        for change in ("cases", "digest", "protocol", "tasks"):
            evidence = {
                "schema_version": 1,
                "split": split.SPLIT,
                "protocol": "fresh-local-file-handoff-v1",
                "tasks": [copy.deepcopy(self.row)],
            }
            if change == "cases":
                evidence["tasks"][0]["cases"].pop()
            elif change == "digest":
                evidence["tasks"][0]["task_sha256"] = "0" * 64
            elif change == "protocol":
                evidence["protocol"] = "wrong"
            else:
                evidence["tasks"] = []
            path = self.root / "bad.json"
            split.write_json(path, evidence)
            output = self.root / "package"
            with self.subTest(change=change), self.assertRaises(ValueError):
                split.build(output, path, self.task_root, self.inventory_path)
            self.assertFalse(output.exists())

    def test_preparation_forwards_exact_hashes_and_explicit_settings(self):
        cloud = self.cloud()
        output = self.root / "plans"
        self.prepare(cloud, output)
        args = cloud.prepare.call_args
        self.assertEqual(
            args.kwargs["source"],
            {
                "kind": "local",
                "revision": self.digest,
                "task_path": ".",
                "task_file_hashes": self.expected,
            },
        )
        config = args.args[1]
        self.assert_config(config)
        self.assertEqual(config["cloud"]["results_repo"], "example/private-results")
        self.assertEqual(config["execution"]["agent_image"], self.image)
        cloud._read_plan.assert_called_once_with(output / self.task_id / "plan.json")
        record = json.loads((output / "preparation-manifest.json").read_text())[
            "tasks"
        ][0]
        self.assertEqual(
            record["native_inventory_revision"], self.inventory["revision"]
        )
        for field, value in (
            ("config_source_sha256", config),
            ("task_file_hashes_sha256", self.expected),
        ):
            self.assertEqual(
                record[field],
                hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )

    def test_inventory_and_bundle_path_or_hash_mismatches_fail_closed(self):
        for boundary in ("inventory", "bundle"):
            for change in ("missing", "extra", "changed"):
                hashes = self.expected.copy()
                if change == "missing":
                    del hashes["instruction.md"]
                elif change == "extra":
                    hashes["unexpected.py"] = "0" * 64
                else:
                    hashes["instruction.md"] = "0" * 64
                inventory = copy.deepcopy(self.inventory)
                if boundary == "inventory":
                    inventory["task_file_hashes"][self.task_id] = hashes
                split.write_json(self.inventory_path, inventory)
                cloud = self.cloud(hashes if boundary == "bundle" else None)
                output = self.root / (boundary + change)
                with (
                    self.subTest(boundary=boundary, change=change),
                    self.assertRaises(ValueError),
                ):
                    self.prepare(cloud, output)
                self.assertEqual(cloud.prepare.call_count, int(boundary == "bundle"))
                self.assertFalse((output / "preparation-manifest.json").exists())

    def test_adapter_ignoring_hashes_or_mutating_frozen_source_cannot_certify(self):
        for change in ("binding", "source", "inventory"):
            cloud = self.cloud()
            if change == "binding":
                cloud._read_plan.return_value[0]["task_source"].pop("task_file_hashes")
            elif change == "inventory":

                def rewrite_binding(task, config, output, repo_root, source=None):
                    source["task_file_hashes"]["instruction.md"] = "0" * 64
                    plan, _, _, manifest = cloud._read_plan.return_value
                    plan["task_source"] = source
                    manifest["files"]["task/instruction.md"]["sha256"] = "0" * 64
                    split.write_json(output / "plan.json", plan)

                cloud.prepare.side_effect = rewrite_binding
            else:

                def prepare(task, config, output, repo_root, source=None):
                    split.write_json(output / "plan.json", {})
                    (task / "instruction.md").write_text("toy changed")

                cloud.prepare.side_effect = prepare
            output = self.root / change
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(cloud, output)
            self.assertFalse((output / "preparation-manifest.json").exists())

    def test_optional_adapter_rejects_old_api_and_wrong_checkout(self):
        pipeline = self.root / "pipeline"
        adapter_path = pipeline / "examples/hub_validation_pipeline/cloud.py"
        adapter_path.parent.mkdir(parents=True)
        adapter_path.write_text("# Mock-only adapter location.\n")
        cloud = self.cloud()
        cloud.__file__ = str(adapter_path)
        cloud.prepare = lambda task, config, output, repo_root: None
        original_path = sys.path[:]
        with patch.object(split.importlib, "import_module", return_value=cloud):
            with self.assertRaisesRegex(ValueError, "binding support"):
                split.pipeline_adapter(pipeline)
            cloud.__file__ = str(self.root / "other/cloud.py")
            with self.assertRaisesRegex(ValueError, "outside"):
                split.pipeline_adapter(pipeline)
        self.assertEqual(sys.path, original_path)
        with self.assertRaisesRegex(ValueError, "no experimental"):
            split.pipeline_adapter(self.root / "missing")

    def test_image_lock_requires_exact_ids_and_registry_pins(self):
        for value in (
            {},
            {self.task_id: split.BASE_IMAGE},
            {self.task_id: "registry.example/toy:latest"},
        ):
            split.write_json(self.image_lock, value)
            cloud = self.cloud()
            with self.subTest(lock=value), self.assertRaises(ValueError):
                self.prepare(cloud, self.root / "plans")
            cloud.prepare.assert_not_called()

    def test_task_and_submission_paths_reject_symlinks_and_traversal(self):
        for value in (
            "/app/../marker.py",
            "/app//marker.py",
            "/elsewhere/marker.py",
            "/app/file.txt",
        ):
            with self.subTest(path=value), self.assertRaises(ValueError):
                split.submission_names({"submission_paths": [value]})
        link = self.task / "link.py"
        link.symlink_to(self.task / "environment/source/marker.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            split.verified_digest(self.task, self.expected)

    def test_container_checker_accepts_external_inputs_without_calling_docker(self):
        output = self.root / "container.json"

        def docker(*args, **kwargs):
            if args[0] == "build":
                self.assertEqual(
                    split.verified_digest(args[-1].parent, self.expected), self.digest
                )
                return "sha256:" + "b" * 64
            if args[:2] == ("image", "inspect"):
                return json.dumps([{"Architecture": "amd64"}])
            return "Python 3.12.0"

        records = iter(copy.deepcopy(self.row["cases"]))
        with (
            patch.object(
                sys,
                "argv",
                [
                    "container_check.py",
                    "--task-root",
                    str(self.task_root),
                    "--inventory",
                    str(self.inventory_path),
                    "--output",
                    str(output),
                ],
            ),
            patch.object(container_check, "docker", side_effect=docker) as calls,
            patch.object(
                container_check, "run", side_effect=lambda *args: next(records)
            ),
        ):
            container_check.main()
        evidence = json.loads(output.read_text())
        split.validate_evidence(
            evidence, "docker-offline-file-handoff-v1", self.task_root, self.inventory
        )
        self.assertEqual(calls.call_args.args[:2], ("image", "rm"))
        evidence["tasks"][0]["private_paths_absent"] = False
        with self.assertRaises(ValueError):
            split.validate_evidence(
                evidence,
                "docker-offline-file-handoff-v1",
                self.task_root,
                self.inventory,
            )

    def docker_engine(self, fail_at=None):
        """Model Docker file handoff while exercising the actual command builder."""
        commands = []
        reference = split.trial(
            self.task, self.spec, self.task / "solution/solve.sh", sys.executable
        )

        def engine(command, **kwargs):
            commands.append(command)
            action = command[1]
            stage = action
            if action == "run":
                stage = "solver" if command[-1] == "/control.sh" else "verifier"
            if stage == fail_at:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if action == "cp":
                solver = Path(command[-1])
                solver.mkdir()
                (solver / "marker.py").write_text('MARKER = "starter"\n')
            if action == "run":
                mounts = {
                    command[i + 1].split(":")[1]: Path(command[i + 1].split(":")[0])
                    for i, value in enumerate(command[:-1])
                    if value == "-v"
                }
                if stage == "solver":
                    (mounts["/app"] / "marker.py").write_text('MARKER = "ready"\n')
                else:
                    (mounts["/logs/verifier"] / "reward.txt").write_text("1\n")
                    return subprocess.CompletedProcess(
                        command, 0, "", reference["verifier_output"]
                    )
            return subprocess.CompletedProcess(command, 0, "", "")

        return commands, engine

    def test_actual_docker_run_forwards_script_contract_and_cleans_owned_containers(
        self,
    ):
        commands, engine = self.docker_engine()
        with patch.object(container_check.subprocess, "run", side_effect=engine):
            result = container_check.run(
                self.task, self.spec, self.image, self.task / "solution/solve.sh"
            )
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["tests_run"], 3)
        self.assertEqual(len(result["test_ids"]), 3)
        launched = [command for command in commands if command[1] in {"create", "run"}]
        self.assertEqual(len(launched), 3)
        names = {command[command.index("--name") + 1] for command in launched}
        removed = {command[-1] for command in commands if command[1:3] == ["rm", "-fv"]}
        self.assertEqual(names, removed)
        for command in (command for command in launched if command[1] == "run"):
            self.assertIn("APP_DIR=/app", command)
            self.assertIn("PYTHON=python", command)
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", command)
            if command[-1] == "/tests/test.sh":
                self.assertIn("TEST_DIR=/tests", command)
                self.assertIn("VERIFIER_LOG_DIR=/logs/verifier", command)

    def test_actual_docker_timeouts_remove_containers_and_anonymous_volumes(self):
        for stage in ("create", "solver", "verifier"):
            commands, engine = self.docker_engine(stage)
            with (
                self.subTest(stage=stage),
                patch.object(container_check.subprocess, "run", side_effect=engine),
                self.assertRaises(subprocess.TimeoutExpired),
            ):
                container_check.run(
                    self.task, self.spec, self.image, self.task / "solution/solve.sh"
                )
            names = {
                command[command.index("--name") + 1]
                for command in commands
                if command[1] in {"create", "run"}
            }
            removed = {
                command[-1] for command in commands if command[1:3] == ["rm", "-fv"]
            }
            self.assertEqual(names, removed)

    def test_visibility_timeout_removes_named_container_and_image(self):
        commands = []

        def docker(*args, **kwargs):
            commands.append(args)
            if args[0] == "build":
                return "sha256:" + "b" * 64
            if args[:2] == ("image", "inspect"):
                return json.dumps([{"Architecture": "amd64"}])
            if args[0] == "run":
                raise subprocess.TimeoutExpired(args, 300)
            return ""

        with (
            patch.object(
                sys,
                "argv",
                [
                    "container_check.py",
                    "--task-root",
                    str(self.task_root),
                    "--inventory",
                    str(self.inventory_path),
                    "--output",
                    str(self.root / "container.json"),
                ],
            ),
            patch.object(container_check, "docker", side_effect=docker),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            container_check.main()
        run = next(command for command in commands if command[0] == "run")
        name = run[run.index("--name") + 1]
        self.assertIn(("rm", "-fv", name), commands)
        self.assertEqual(commands[-1][:2], ("image", "rm"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
