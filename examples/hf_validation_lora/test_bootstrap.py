"""Controller gates and teardown must fail closed before spending GPU time."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "bootstrap", Path(__file__).with_name("bootstrap.py")
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


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
