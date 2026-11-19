# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run inside the production data_analysis_env image as root, with a fake HF_TOKEN in the env.

This process plays the server: it holds the token and the gold answers. The checks run model code
in real sessions, then repeat the dangerous parts *without* the audit hook, as the session's uid,
to show the OS boundary holds on its own. EXPECT_NETNS=1 means the container was given
CAP_SYS_ADMIN (and CAP_SYS_PTRACE, only so this check can enter a worker's namespace), so the
workers must have no network at all; EXPECT_NETNS=0 checks that the server reports it can't
isolate the network.

    docker run --rm -e HF_TOKEN=hf_FAKE... -e EXPECT_NETNS=1 --cap-add SYS_ADMIN --cap-add SYS_PTRACE \
        -v $PWD/tests/containers/check_data_analysis_sandbox.py:/checks.py:ro \
        --entrypoint /app/.venv/bin/python data-analysis-env /checks.py
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from data_analysis_env.server import sandbox, tasks
from data_analysis_env.server.data_analysis_environment import DataAnalysisEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

FAKE = os.environ.get("HF_TOKEN", "")
EXPECT_NETNS = os.environ.get("EXPECT_NETNS") == "1"
GOLD = "GOLD-SENTINEL-98765"

# libc socket()+connect(): neither the socket module nor the audit hook is involved.
CTYPES_CONNECT = """
import ctypes, socket, struct, sys
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.socket(2, 1, 0)
addr = struct.pack("=H", 2) + struct.pack("!H", int(sys.argv[2])) + socket.inet_aton(sys.argv[1]) + bytes(8)
print("CONNECTED" if libc.connect(fd, addr, 16) == 0 else f"ERRNO {ctypes.get_errno()}")
"""

# libc socket(AF_INET): the socket module and the audit hook are not involved.
CTYPES_SOCKET = """
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
print("SOCKET" if libc.socket(2, 1, 0) >= 0 else f"ERRNO {ctypes.get_errno()}")
"""


def as_uid(uid, code, *args, netns_of=None):
    """Python as `uid`, no audit hook, empty environment; optionally in a worker's netns."""
    prefix = ["nsenter", f"--target={netns_of}", "--net", "--"] if netns_of else []
    return subprocess.run(
        prefix
        + [
            "setpriv",
            f"--reuid={uid}",
            f"--regid={uid}",
            "--clear-groups",
            "--no-new-privs",
            "--",
            sys.executable,
            "-c",
            code,
            *args,
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        cwd="/",
    )


def worker_status(pid):
    return dict(
        line.split(":", 1)
        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
        if ":" in line
    )


class DataAnalysisSandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.geteuid() == 0 and FAKE.startswith("hf_FAKE"), (
            "run as root with a fake HF_TOKEN"
        )
        cls.root = tempfile.mkdtemp(prefix="data_analysis_check_")
        os.chmod(cls.root, 0o755)
        os.environ["DATA_ANALYSIS_TASKS_ROOT"] = cls.root
        os.environ["DATA_ANALYSIS_EXEC_TIMEOUT_S"] = "10"
        rows = []
        for task_id in ("t1", "t2"):
            d = Path(cls.root, "eval", task_id)
            d.mkdir(parents=True)
            (d / "data.csv").write_text('x,y,name\n1,2,"a,b"\n3,4,"c,d"\n')
            rows.append(
                {
                    "task_id": task_id,
                    "question": "Sum of x and y?",
                    "answer": GOLD if task_id == "t2" else "10",
                    "reward_mode": "numeric",
                    "atol": 1e-3,
                    "rtol": 1e-3,
                    "difficulty_tier": "easy",
                    "hf_bucket": "someone/bucket",
                    "bucket_prefix": f"x/{task_id}",
                    "files": ["data.csv"],
                }
            )
        for path in [Path(cls.root), *Path(cls.root).rglob("*")]:
            os.chmod(path, 0o755 if path.is_dir() else 0o644)
        tasks.load_split = lambda split: rows

    def setUp(self):
        self.env = DataAnalysisEnvironment()
        self.addCleanup(self.env.close)
        self.files = self.env.reset(task_id="t1").metadata["files"]

    def run_code(self, code, env=None):
        obs = (env or self.env).step(
            CallToolAction(tool_name="run_python", arguments={"code": code})
        )
        return obs.result.data

    def session(self, env=None):
        return (env or self.env)._session

    def test_real_pandas_and_reward(self):
        out = self.run_code(
            f"import pandas as pd\ndf = pd.read_csv({self.files[0]!r})\n"
            "print(int(df[['x', 'y']].sum().sum()), df['name'].str.split(',', expand=True)[0].tolist())"
        )
        self.assertEqual(out, "10 ['a', 'c']\n")
        obs = self.env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "10"})
        )
        self.assertEqual((obs.done, obs.reward), (True, 1.0))

    def test_worker_runs_as_its_own_unprivileged_user(self):
        out = self.run_code("import os\nos.getresuid(), os.getresgid(), os.getgroups()")
        uid = self.session().uid
        self.assertGreaterEqual(uid, sandbox.DEFAULT_UID_BASE)
        self.assertEqual(
            out.strip(), f"(({uid}, {uid}, {uid}), ({uid}, {uid}, {uid}), [])"
        )
        status = worker_status(self.session().pid)
        self.assertEqual(int(status["CapEff"].strip(), 16), 0)
        self.assertEqual(int(status["CapPrm"].strip(), 16), 0)
        limits = Path(f"/proc/{self.session().pid}/limits").read_text()
        self.assertRegex(limits, r"Max processes\s+64\s+64")
        self.assertRegex(limits, r"Max address space\s+4294967296")

    def test_two_sessions_are_two_users(self):
        other = DataAnalysisEnvironment()
        self.addCleanup(other.close)
        other.reset(task_id="t1")
        self.run_code("1")
        self.run_code("open('note', 'w').write('mine')", env=other)
        a, b = self.session(), self.session(other)
        self.assertNotEqual(a.uid, b.uid)
        # Without the hook: a's uid can't read b's home, kill b's worker, or write task files.
        r = as_uid(a.uid, "import sys; open(sys.argv[1]).read()", f"{b.home}/note")
        self.assertIn("PermissionError", r.stderr)
        r = as_uid(a.uid, "import os, sys; os.kill(int(sys.argv[1]), 9)", str(b.pid))
        self.assertIn("PermissionError", r.stderr)
        r = as_uid(a.uid, "import sys; open(sys.argv[1], 'a')", self.files[0])
        self.assertIn("PermissionError", r.stderr)
        # Nor list which tasks are on the server.
        for path in (self.root, f"{self.root}/eval"):
            r = as_uid(a.uid, "import os, sys; os.listdir(sys.argv[1])", path)
            self.assertIn("PermissionError", r.stderr)

    def test_no_token_and_no_server_secrets(self):
        out = self.run_code("import os\ndict(os.environ)")
        self.assertNotIn(FAKE, out)
        self.assertNotIn("HF_TOKEN", out)
        for path in (
            f"/proc/{os.getpid()}/environ",
            "/proc/1/environ",
            "/root/.bashrc",
        ):
            out = self.run_code(
                f"import pandas as pd\npd.read_csv({path!r}, header=None, sep='\\x01')"
            )
            self.assertIn("SandboxViolation", out)
            self.assertNotIn(FAKE, out)
        # The OS boundary alone, as the session uid: the server's environ, memory and fds.
        uid = self.session().uid
        probe = (
            "import sys\nfor p in sys.argv[1:]:\n"
            "    try:\n        open(p, 'rb').read(1); print('OPENED', p)\n"
            "    except (PermissionError, OSError):\n        print('DENIED', p)\n"
        )
        me = os.getpid()
        r = as_uid(
            uid,
            probe,
            f"/proc/{me}/environ",
            f"/proc/{me}/mem",
            f"/proc/{me}/fd/0",
            "/proc/1/environ",
        )
        self.assertEqual(r.stdout.count("DENIED"), 4, r.stdout + r.stderr)
        r = as_uid(uid, "import os; print(os.listdir('/root'))")
        self.assertIn("PermissionError", r.stderr)

    def test_gold_answer_is_not_reachable(self):
        self.env.reset(task_id="t2")
        self.run_code("1")
        uid, server = self.session().uid, os.getpid()
        # Not in the prompt, the state, or anything the worker can read.
        self.assertNotIn(GOLD, self.env.state.model_dump_json())
        r = as_uid(
            uid,
            "import sys\ntry:\n    print(open(sys.argv[1], 'rb').read()[:10])\nexcept OSError as e:\n    print('DENIED', type(e).__name__)",
            f"/proc/{server}/mem",
        )
        self.assertIn("DENIED", r.stdout)
        r = as_uid(
            uid,
            "import ctypes, sys\nlibc = ctypes.CDLL(None, use_errno=True)\n"
            "print('ATTACHED' if libc.ptrace(16, int(sys.argv[1]), None, None) == 0 else 'ERRNO')",
            str(server),
        )
        self.assertEqual(r.stdout.strip(), "ERRNO")

    def test_writes_only_in_home_and_strays_are_swept(self):
        out = self.run_code("open('x.txt', 'w').write('ok')\nopen('x.txt').read()")
        self.assertEqual(out, "'ok'\n")
        self.assertIn("SandboxViolation", self.run_code("open('/tmp/escape', 'w')"))
        uid, home = self.session().uid, self.session().home
        self.assertEqual(os.stat(home).st_uid, uid)
        self.assertEqual(os.stat(home).st_mode & 0o777, 0o700)
        # Code past the hook can write to /tmp and leave a process behind; closing the session
        # kills the process and removes the file before the uid is reused.
        stray = subprocess.Popen(
            [
                "setpriv",
                f"--reuid={uid}",
                f"--regid={uid}",
                "--clear-groups",
                "sleep",
                "600",
            ]
        )
        as_uid(uid, "open('/tmp/data_analysis_stray', 'w').write('x')")
        self.assertTrue(os.path.exists("/tmp/data_analysis_stray"))
        self.env.close()
        self.assertEqual(stray.wait(timeout=10), -signal.SIGKILL)
        self.assertFalse(os.path.exists("/tmp/data_analysis_stray"))
        self.assertFalse(os.path.exists(home))

    def test_network(self):
        self.assertIn(
            "SandboxViolation", self.run_code("import socket\nsocket.socket()")
        )
        self.assertIn(
            "SandboxViolation",
            self.run_code(
                "import pandas as pd\npd.read_csv('https://huggingface.co/robots.txt')"
            ),
        )
        status = sandbox.sandbox_status()
        self.assertEqual(status["network_isolated"], EXPECT_NETNS, status)
        self.assertTrue(status["socket_filter"], status)
        worker, uid = self.session().pid, self.session().uid
        # The worker runs under the seccomp filter, and a process with that filter (and without
        # the audit hook) can't create a socket.
        self.assertEqual(worker_status(worker)["Seccomp"].strip(), "2")
        # sandbox.py alone (stdlib only), not the package: its imports need a HOME.
        filtered = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('sandbox_only', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "module.block_sockets()\n"
        )
        r = as_uid(uid, filtered + CTYPES_SOCKET, sandbox.__file__)
        self.assertEqual(r.stdout.strip(), "ERRNO 1", r.stdout + r.stderr)  # EPERM
        r = as_uid(uid, CTYPES_SOCKET)  # the same process without the filter
        self.assertEqual(r.stdout.strip(), "SOCKET", r.stdout + r.stderr)
        if EXPECT_NETNS:
            self.assertNotEqual(
                os.readlink(f"/proc/{worker}/ns/net"), os.readlink("/proc/self/ns/net")
            )
            for ip, port in (("127.0.0.1", "8000"), ("1.1.1.1", "80")):
                r = as_uid(uid, CTYPES_CONNECT, ip, port, netns_of=worker)
                self.assertTrue(r.stdout.startswith("ERRNO"), r.stdout + r.stderr)
        else:
            self.assertIn("unshare", status["network_reason"])

    def test_limits(self):
        out = self.run_code(
            "import numpy as np\na = np.ones(6 * 1024**3 // 8)\na.sum()"
        )
        self.assertIn("Error", out)
        self.assertEqual(self.run_code("1 + 1"), "2\n")
        start = time.time()
        out = self.run_code("import time\ntime.sleep(60)")
        self.assertIn("TimeoutError", out)
        self.assertLess(time.time() - start, 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
