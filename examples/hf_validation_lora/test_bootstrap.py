"""Controller gates and teardown must fail closed before spending GPU time."""

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "bootstrap", Path(__file__).with_name("bootstrap.py")
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class TrainingCommandTests(unittest.TestCase):
    """Execute the generated shell with offline pip, download, and uv shims."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "events.jsonl"
        self.source = self.root / "train.py"
        self.lock = self.root / "training.txt"
        self.repo = "owner/private-artifacts"
        self.revision = "6fbf387237893da0580bd2dac0739a741e188120"
        self.command = bootstrap.training_command(self.repo, self.revision)
        self.assertEqual(self.command[:2], ["/bin/sh", "-lc"])
        # Skip host login profiles, which can replace the fixture-only PATH.
        self.command[1] = "-c"
        # Change only the two output paths, including their uv arguments.
        self.command[2] = self.command[2].replace("/tmp/train.py", str(self.source))
        self.command[2] = self.command[2].replace("/tmp/training.txt", str(self.lock))
        self.env = {
            "PATH": str(self.bin),
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_TOKEN": "offline-token",
            "EVENT_LOG": str(self.log),
            "PIP_EXIT": "0",
            "FAIL_DOWNLOAD": "0",
        }
        self.pip = [
            "pip",
            ["-m", "pip", "install", "--disable-pip-version-check", "uv==0.9.9"],
        ]
        self.downloads = [
            [
                "download",
                f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{src}",
                {"Authorization": "Bearer offline-token"},
                60,
            ]
            for src in ["scripts/train.py", "locks/training-python311.txt"]
        ]
        shims = {
            "python": """import json, os, sys
if sys.argv[1:3] == ['-m', 'pip']:
    with open(os.environ['EVENT_LOG'], 'a') as log:
        log.write(json.dumps(['pip', sys.argv[1:]]) + '\\n')
    sys.exit(int(os.environ['PIP_EXIT']))
assert sys.argv[1:] == ['-'], sys.argv
os.execv(sys.executable, [sys.executable, '-S', '-'])
""",
            "uv": """import json, os, sys
with open(os.environ['EVENT_LOG'], 'a') as log:
    log.write(json.dumps(['uv', sys.argv[1:]]) + '\\n')
""",
        }
        for name, script in shims.items():
            executable = self.bin / name
            executable.write_text(f"#!{sys.executable} -S\n" + script)
            executable.chmod(0o700)
        urllib = self.root / "urllib"
        urllib.mkdir()
        (urllib / "__init__.py").write_text(
            "import sysconfig\n__path__.append(sysconfig.get_path('stdlib') + '/urllib')\n"
        )
        (urllib / "request.py").write_text(
            """import io, json, os
calls = 0
class Request:
    def __init__(self, url, headers):
        self.url, self.headers = url, headers
def urlopen(request, timeout):
    global calls
    calls += 1
    with open(os.environ['EVENT_LOG'], 'a') as log:
        log.write(json.dumps(['download', request.url, request.headers, timeout]) + '\\n')
    if calls == int(os.environ['FAIL_DOWNLOAD']):
        raise OSError('offline download failure')
    return io.BytesIO(('pinned:' + request.url).encode())
"""
        )
        # Refuse to execute the script unless both tools resolve to inert shims.
        probe = subprocess.run(
            self.command[:2] + ["command -v python; command -v uv"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(
            probe.stdout.splitlines(),
            [str(self.bin / "python"), str(self.bin / "uv")],
        )

    def run_command(self):
        result = subprocess.run(
            self.command,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        events = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, events

    def assert_failure_stops_training(self, *, pip_exit=0, download_failure=0):
        self.env.update(PIP_EXIT=str(pip_exit), FAIL_DOWNLOAD=str(download_failure))
        for contents in [None, "stale complete file", "partial"]:
            with self.subTest(existing_files=contents):
                self.log.unlink(missing_ok=True)
                for path in [self.source, self.lock]:
                    if contents is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(contents)
                result, events = self.run_command()
                self.assertEqual(events, [self.pip] + self.downloads[:download_failure])
                self.assertEqual(result.returncode, pip_exit or 1, result.stderr)

    def test_pip_failure_prevents_download_and_training(self):
        self.assert_failure_stops_training(pip_exit=23)

    def test_first_download_failure_prevents_training(self):
        self.assert_failure_stops_training(download_failure=1)

    def test_second_download_failure_prevents_training(self):
        self.assert_failure_stops_training(download_failure=2)

    def test_success_downloads_exact_revision_and_preserves_uv_arguments(self):
        result, events = self.run_command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            events,
            [self.pip]
            + self.downloads
            + [
                [
                    "uv",
                    [
                        "run",
                        "--python",
                        "/usr/local/bin/python",
                        "--with-requirements",
                        str(self.lock),
                        str(self.source),
                    ],
                ]
            ],
        )
        for path, download in zip([self.source, self.lock], self.downloads):
            self.assertEqual(path.read_bytes(), ("pinned:" + download[1]).encode())


class BootstrapLifetimeTests(unittest.TestCase):
    """Model the pinned server watchdog: quiet AND no running background process.

    sandbox-server v0.7.0, src/main.rs:486-495 and src/exec.rs:430-436.
    No SDK request, subprocess, or real sleep is performed.
    """

    def setUp(self):
        self.now = 0
        self.last_activity = 0
        self.background_running = False
        self.expired = False
        self.worker_started = False
        self.terminal_stage = "COMPLETED"
        self.queue_until = 25 * 60
        self.finish_at = 30 * 60  # 25 minutes queued, then five minutes running.
        self.inspection_error = False
        self.canceled = []
        self.cancel_errors = set()
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        report = {
            "results": [
                {"check_id": "runtime." + name, "status": "pass"}
                for name in [
                    "startup",
                    "reward_well_formed",
                    "observation_schema",
                    "state_contract",
                    "episode_isolation",
                ]
            ]
        }
        self.sandbox = SimpleNamespace(
            id="sandbox",
            run=Mock(side_effect=self.run_command),
            files=SimpleNamespace(
                write=Mock(),
                read_text=Mock(return_value=json.dumps(report)),
                download=Mock(),
            ),
        )
        self.create = Mock(return_value=self.sandbox)
        self.api = Mock()
        for info in [self.api.dataset_info, self.api.model_info, self.api.space_info]:
            info.return_value = SimpleNamespace(private=True)
        self.api.create_commit.return_value = SimpleNamespace(oid="a" * 40)
        self.api.run_job.return_value = SimpleNamespace(id="gpu")
        self.api.inspect_job.side_effect = self.inspect
        self.api.cancel_job.side_effect = self.cancel
        self.enterContext(patch.object(bootstrap, "HfApi", return_value=self.api))
        self.enterContext(
            patch.object(bootstrap, "Sandbox", SimpleNamespace(create=self.create))
        )
        self.enterContext(
            patch.object(bootstrap, "get_token", return_value="fake-token")
        )
        self.enterContext(patch.object(bootstrap.time, "monotonic", lambda: self.now))
        self.enterContext(patch.object(bootstrap.time, "sleep", self.advance))
        self.enterContext(patch("sys.stdout", new_callable=io.StringIO))
        self.enterContext(
            patch(
                "sys.argv",
                [
                    "bootstrap.py",
                    "--artifact-repo",
                    "owner/private-artifacts",
                    "--model-repo",
                    "owner/private-model",
                    "--trackio-space",
                    "owner/private-trackio",
                    "--train",
                    "--output",
                    str(self.output),
                ],
            )
        )

    def run_command(self, command, **kwargs):
        self.last_activity = self.now
        if kwargs.get("background"):
            self.background_running = True

    def advance(self, seconds):
        self.now += seconds
        if self.now - self.last_activity > 20 * 60 and not self.background_running:
            self.expired = True

    def inspect(self, job_id):
        if job_id in self.canceled:
            stage = "CANCELED"
        elif job_id == "sandbox":
            stage = "COMPLETED" if self.expired else "RUNNING"
        elif self.inspection_error and self.now >= self.queue_until:
            raise ConnectionError("offline inspection failure")
        elif self.now < self.queue_until:
            stage = "PENDING"
        else:
            self.worker_started = True
            stage = "ERROR" if self.expired else "RUNNING"
            if not self.expired and self.now >= self.finish_at:
                stage = self.terminal_stage
        return SimpleNamespace(status=SimpleNamespace(stage=stage))

    def cancel(self, job_id):
        if job_id in self.cancel_errors:
            raise ConnectionError("offline cancellation failure")
        self.canceled.append(job_id)

    def state(self):
        record = json.loads((self.output / "state.json").read_text())
        self.api.upload_file.assert_called_once()
        self.assertEqual(record["sandbox_final_stage"], "CANCELED")
        return record

    def test_background_server_survives_queue_longer_than_idle_timeout(self):
        bootstrap.main()
        self.assertEqual(self.now, 30 * 60)
        self.assertTrue(self.worker_started)
        self.assertFalse(self.expired)
        self.assertEqual(self.create.call_args.kwargs["idle_timeout"], "20m")
        self.assertFalse(self.create.call_args.kwargs["forward_hf_token"])
        self.assertEqual(self.api.run_job.call_args.kwargs["timeout"], "1h")
        self.assertEqual(self.state()["training_final_stage"], "COMPLETED")
        self.assertEqual(self.canceled, ["sandbox"])

    def test_idle_watchdog_control_without_background_work(self):
        self.sandbox.run("true")
        self.advance(20 * 60 + 5)
        self.assertTrue(self.expired)

    def test_training_error_still_stops_sandbox(self):
        self.terminal_stage = "ERROR"
        with self.assertRaisesRegex(RuntimeError, "training Job ended at ERROR"):
            bootstrap.main()
        self.assertEqual(self.state()["training_final_stage"], "ERROR")
        self.assertEqual(self.canceled, ["sandbox"])

    def test_training_canceled_still_stops_sandbox(self):
        self.terminal_stage = "CANCELED"
        with self.assertRaisesRegex(RuntimeError, "training Job ended at CANCELED"):
            bootstrap.main()
        self.assertEqual(self.state()["training_final_stage"], "CANCELED")
        self.assertEqual(self.canceled, ["sandbox"])

    def test_controller_deadline_cancels_queued_training_and_live_sandbox(self):
        self.queue_until = float("inf")
        with self.assertRaisesRegex(TimeoutError, "controller deadline"):
            bootstrap.main()
        self.assertEqual(self.now, 65 * 60)
        self.assertFalse(self.worker_started)
        self.assertFalse(self.expired)
        self.assertEqual(self.state()["training_final_stage"], "CANCELED")
        self.assertEqual(self.canceled, ["gpu", "sandbox"])

    def test_inspection_error_cancels_training_and_sandbox(self):
        self.inspection_error = True
        with self.assertRaises(ConnectionError):
            bootstrap.main()
        self.assertEqual(self.state()["training_final_stage"], "CANCELED")
        self.assertEqual(self.canceled, ["gpu", "sandbox"])

    def test_unconfirmed_training_cleanup_still_stops_sandbox(self):
        self.finish_at = float("inf")
        self.cancel_errors.add("gpu")
        with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
            bootstrap.main()
        self.assertEqual(
            self.state()["cleanup_errors"], {"training": "ConnectionError"}
        )
        self.assertEqual(self.canceled, ["sandbox"])


class ControllerTests(unittest.TestCase):
    def test_cancellation_is_checked_after_final_sleep(self):
        after_cancel = 0
        canceled = False

        class API:
            def inspect_job(self, job_id):
                nonlocal after_cancel
                if canceled:
                    after_cancel += 1
                return SimpleNamespace(
                    status=SimpleNamespace(
                        stage="CANCELED" if after_cancel == 11 else "RUNNING"
                    )
                )

            def cancel_job(self, job_id):
                nonlocal canceled
                canceled = True

        with patch.object(bootstrap.time, "sleep") as sleep:
            self.assertEqual(bootstrap.stop_owned_job(API(), "owned"), "CANCELED")
        self.assertEqual(sleep.call_count, 10)

    def test_requires_every_measured_check_once(self):
        rows = [
            {"check_id": "runtime." + name, "status": "pass"}
            for name in [
                "startup",
                "reward_well_formed",
                "observation_schema",
                "state_contract",
                "episode_isolation",
            ]
        ]
        bootstrap.require_qualified({"results": rows})
        for invalid in [
            [],
            rows[:1],
            rows + rows[:1],
            rows[:-1] + [rows[0]],
            rows + [{"check_id": "runtime.other", "status": "error"}],
        ]:
            with self.subTest(rows=invalid), self.assertRaises(RuntimeError):
                bootstrap.require_qualified({"results": invalid})
        for status in ["skip", "fail", "error"]:
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                bootstrap.require_qualified(
                    {"results": [{**rows[0], "status": status}] + rows[1:]}
                )

    def test_failed_gpu_inspection_and_cancel_still_stop_sandbox_and_save_state(self):
        calls = []

        class API:
            def inspect_job(self, job_id):
                calls.append(("inspect", job_id))
                if job_id == "gpu":
                    raise ConnectionError("inspection unavailable")
                return SimpleNamespace(status=SimpleNamespace(stage="CANCELED"))

            def cancel_job(self, job_id):
                calls.append(("cancel", job_id))
                if job_id == "gpu":
                    raise ConnectionError("cancellation unavailable")

            def upload_file(self, **kwargs):
                calls.append(
                    ("upload", json.loads(kwargs["path_or_fileobj"].read_text()))
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = {}
            with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
                bootstrap.finish_owned_compute(
                    API(),
                    SimpleNamespace(id="gpu"),
                    SimpleNamespace(id="sandbox"),
                    record,
                    output,
                    "owner/private",
                )
            self.assertIn(("cancel", "gpu"), calls)
            self.assertIn(("inspect", "sandbox"), calls)
            self.assertEqual(record["sandbox_final_stage"], "CANCELED")
            self.assertEqual(record["cleanup_errors"], {"training": "ConnectionError"})
            self.assertEqual(json.loads((output / "state.json").read_text()), record)
            self.assertEqual(calls[-1], ("upload", record))

    def test_initial_inspection_failure_still_cancels_owned_job(self):
        calls = []

        class API:
            def inspect_job(self, job_id):
                if not calls:
                    raise ConnectionError("inspection unavailable")
                return SimpleNamespace(status=SimpleNamespace(stage="CANCELED"))

            def cancel_job(self, job_id):
                calls.append(job_id)

        self.assertEqual(bootstrap.stop_owned_job(API(), "owned"), "CANCELED")
        self.assertEqual(calls, ["owned"])


if __name__ == "__main__":
    unittest.main()
