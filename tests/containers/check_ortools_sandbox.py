# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run inside the production OR-Tools image as root, with networking disabled."""

import importlib.util
import os
import select
import tempfile
import time
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("sandbox", "/app/env/sandbox.py")
sandbox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sandbox)


class SandboxIsolation(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory(prefix="ortools_container_check_")
        self.addCleanup(self.workdir.cleanup)

    def run_code(self, code, **kwargs):
        return sandbox.run(code, {"value": 42}, self.workdir.name, 10, **kwargs)

    def test_uid_and_import_isolation_after_removing_import_hook(self):
        code = """
import os, sys, importlib, subprocess
assert 60000 <= os.getuid() < 65000
assert os.getgroups() == []
sys.meta_path = [f for f in sys.meta_path if type(f).__name__ != '_Blocker']
for name in ['ortools', 'problems', 'server', 'ortools_env']:
    try:
        importlib.import_module(name)
    except ImportError:
        pass
    else:
        raise AssertionError(name)
for path in ['/app/env/sandbox.py', '/app/env/server/ortools_environment.py']:
    try:
        open(path).read()
    except PermissionError:
        pass
    else:
        raise AssertionError(path)
try:
    subprocess.run(['/opt/sandbox-ortools/bin/python', '-c', 'import ortools'], check=True)
except PermissionError:
    pass
else:
    raise AssertionError('restricted interpreter was executable')
"""
        result = self.run_code(code)
        self.assertEqual(result["returncode"], 0, result["output"])
        result = self.run_code(
            "import ortools; print(ortools.__version__)", allow_ortools=True
        )
        self.assertEqual(result["returncode"], 0, result["output"])

    def test_privileged_file_io_and_canary(self):
        with tempfile.TemporaryDirectory() as private:
            target = Path(private) / "private.json"
            target.write_text('{"private": true}')
            os.environ["OPENENV_TEST_SECRET"] = "synthetic-container-canary"
            self.addCleanup(os.environ.pop, "OPENENV_TEST_SECRET")
            result = self.run_code(
                f"""
import os
assert 'OPENENV_TEST_SECRET' not in os.environ
os.unlink('instance.json')
os.symlink({str(target)!r}, 'instance.json')
os.symlink({str(target)!r}, 'solution.json')
os._exit(0)
""",
                solve=True,
            )
            self.assertEqual(result["returncode"], 0, result["output"])
            self.assertIsNone(result["solution"])
            result = self.run_code("print(instance['value'])")
            self.assertEqual(result["output"], "42\n")
            self.assertEqual(target.read_text(), '{"private": true}')

    def test_malformed_solution_fails_without_server_exception(self):
        result = self.run_code(
            "import os\nopen('solution.json', 'w').write('{broken')\nos._exit(0)",
            solve=True,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertIsNone(result["solution"])

    def test_detached_child_does_not_outlive_call(self):
        result = self.run_code("""
import subprocess, sys
subprocess.Popen([sys.executable, '-c',
    "import time; time.sleep(0.5); open('escaped', 'w').write('escaped')"],
    start_new_session=True)
""")
        self.assertEqual(result["returncode"], 0, result["output"])
        time.sleep(1)
        self.assertFalse((Path(self.workdir.name) / "escaped").exists())

    def test_sessions_get_separate_uids(self):
        other = tempfile.TemporaryDirectory(prefix="ortools_container_check_")
        self.addCleanup(other.cleanup)
        uid, other_uid = sandbox.new_uid(), sandbox.new_uid()
        self.addCleanup(sandbox.release_uid, uid)
        self.addCleanup(sandbox.release_uid, other_uid)
        first = sandbox.run(
            "import os; print(os.getuid()); open('mine', 'w').write('x')",
            {},
            self.workdir.name,
            10,
            uid=uid,
        )
        self.assertEqual(first["output"].strip(), str(uid), first["output"])
        second = sandbox.run(
            f"import os; print(os.getuid()); open({str(Path(self.workdir.name) / 'mine')!r}).read()",
            {},
            other.name,
            10,
            uid=other_uid,
        )
        self.assertIn(str(other_uid), second["output"])
        self.assertIn("PermissionError", second["output"])

    def test_process_count_is_limited(self):
        result = self.run_code(
            "print([l for l in open('/proc/self/limits') if l.startswith('Max processes')][0])"
        )
        self.assertEqual(result["returncode"], 0, result["output"])
        self.assertEqual(result["output"].split()[2], str(sandbox.MAX_PROCESSES))

    def test_cleanup_survives_a_child_that_kills_everything(self):
        # The child signals every process of its uid in a loop, including the call itself.
        uid = sandbox.new_uid()
        self.addCleanup(sandbox.release_uid, uid)
        code = (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import os, signal\\n"
            "while True:\\n    try:\\n        os.kill(-1, signal.SIGKILL)\\n"
            "    except OSError:\\n        pass'], start_new_session=True)\n"
            "import time; time.sleep(5)\n"
        )
        start = time.monotonic()
        sandbox.run(code, {}, self.workdir.name, 10, uid=uid)
        self.assertLess(time.monotonic() - start, 30)
        self.assertEqual(sandbox._processes_of(uid), [])

    def test_cleanup_survives_a_fork_bomb(self):
        # Detached processes that keep forking and send SIGCONT to every process of the uid.
        uid = sandbox.new_uid()
        self.addCleanup(sandbox.release_uid, uid)
        code = (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import os, signal\\n"
            "while True:\\n    try:\\n        os.fork()\\n"
            "        os.kill(-1, signal.SIGCONT)\\n"
            "    except OSError:\\n        pass'], start_new_session=True)\n"
            "import time; time.sleep(2)\n"
        )
        start = time.monotonic()
        sandbox.run(code, {}, self.workdir.name, 10, uid=uid)
        self.assertLess(time.monotonic() - start, 30)
        self.assertEqual(sandbox._processes_of(uid), [])

    def test_concurrent_sessions_clean_up_their_own_processes(self):
        import threading

        uids = [sandbox.new_uid() for _ in range(3)]
        dirs = [
            tempfile.TemporaryDirectory(prefix="ortools_container_check_") for _ in uids
        ]
        for uid, d in zip(uids, dirs):
            self.addCleanup(sandbox.release_uid, uid)
            self.addCleanup(d.cleanup)
        code = (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
            "import time; time.sleep(1)\n"
        )
        threads = [
            threading.Thread(
                target=sandbox.run, args=(code, {}, d.name, 10), kwargs={"uid": uid}
            )
            for uid, d in zip(uids, dirs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        for uid in uids:
            self.assertEqual(sandbox._processes_of(uid), [])

    def test_killer_does_not_hold_the_servers_descriptors(self):
        # Another thread's Popen waits for EOF on its error pipe: the killer must not keep it open.
        uid = sandbox.new_uid()
        self.addCleanup(sandbox.release_uid, uid)
        read_fd, write_fd = os.pipe()
        killer = sandbox._start_killer(uid)
        os.close(write_fd)
        try:
            ready, _, _ = select.select([read_fd], [], [], 5)
            self.assertEqual(ready, [read_fd], "the killer kept a copy of the pipe")
            self.assertEqual(os.read(read_fd, 1), b"")
        finally:
            os.close(read_fd)
            sandbox._kill_all(uid, killer)

    def test_setup_failure_does_not_leave_the_killer(self):
        uid = sandbox.new_uid()
        self.addCleanup(sandbox.release_uid, uid)

        def no_temporary_files(*args, **kwargs):
            raise OSError("no space left")

        original = sandbox.tempfile.TemporaryFile
        sandbox.tempfile.TemporaryFile = no_temporary_files
        try:
            with self.assertRaises(OSError):
                sandbox.run("print(1)", {}, self.workdir.name, 10, uid=uid)
        finally:
            sandbox.tempfile.TemporaryFile = original
        deadline = time.monotonic() + 5
        while (
            sandbox._processes_of(uid - sandbox.KILLER_OFFSET)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        self.assertEqual(sandbox._processes_of(uid - sandbox.KILLER_OFFSET), [])

    def test_uid_released_mid_call_returns_to_the_pool_when_the_call_ends(self):
        # A client that disconnects mid-call releases its uid while the code still runs:
        # another session must not get the uid before that call has cleaned up.
        import threading

        uid = sandbox.new_uid()
        thread = threading.Thread(
            target=sandbox.run,
            args=("import time; time.sleep(3)", {}, self.workdir.name, 10),
            kwargs={"uid": uid},
        )
        thread.start()
        deadline = time.monotonic() + 10
        while not sandbox._processes_of(uid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(sandbox._processes_of(uid), "the code never started")

        sandbox.release_uid(uid)
        self.assertNotIn(uid, sandbox._free_uids)

        thread.join(30)
        self.assertIn(uid, sandbox._free_uids)
        self.assertEqual(sandbox._processes_of(uid), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
