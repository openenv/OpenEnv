# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run inside the production OR-Tools image as root, with networking disabled."""

import importlib.util
import os
import pwd
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
        uid = pwd.getpwnam("sandbox").pw_uid
        code = f"""
import os, sys, importlib, subprocess
assert os.getuid() == {uid} != 0
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

    @unittest.expectedFailure
    def test_known_blocker_detached_child_outlives_call(self):
        # Bounded reproduction: the escaped child writes once and exits itself.
        result = self.run_code("""
import subprocess, sys
subprocess.Popen([sys.executable, '-c',
    "import time; time.sleep(0.5); open('escaped', 'w').write('escaped')"],
    start_new_session=True)
""")
        self.assertEqual(result["returncode"], 0, result["output"])
        time.sleep(1)
        self.assertFalse((Path(self.workdir.name) / "escaped").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
