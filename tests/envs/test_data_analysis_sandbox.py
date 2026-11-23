# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""data_analysis_env's sandbox: real CPython in a worker per session, and what model code can't do.

Most tests run anywhere (the worker then runs as the current user, and the audit hook and the
process limits are what is checked). The tests marked `root_only` check the OS boundary (a uid
per session) and run on Linux as root, as in the Docker image. tests/containers/
check_data_analysis_sandbox.py runs the same boundary checks inside the production image.
"""

import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("pandas", reason="pandas is not installed")
# The worker preloads these. The env's own CI job installs them, the generic test job doesn't.
pytest.importorskip("openpyxl", reason="openpyxl is not installed")
pytest.importorskip("sklearn", reason="scikit-learn is not installed")

from data_analysis_env.server import sandbox  # noqa: E402
from data_analysis_env.server.sandbox import SandboxSession  # noqa: E402
from data_analysis_env.server.session import PythonSession  # noqa: E402

TIMEOUT_S = 6
FAKE_TOKEN = "hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00"

root_only = pytest.mark.skipif(
    not sandbox.drops_privileges(), reason="needs root on Linux"
)


@pytest.fixture(scope="module")
def data_dir():
    # Not pytest's tmp_path: as root it is 0700, and session users must be able to read the data.
    root = Path(tempfile.mkdtemp(prefix="data_analysis_test_"))
    task = root / "tasks" / "t1"
    task.mkdir(parents=True)
    (task / "sales.csv").write_text("region,amount\nnorth,10\nsouth,30\nnorth,20\n")
    (task / "sales.tsv").write_text("region\tamount\nnorth\t10\nsouth\t30\n")
    (task / "sales.json").write_text(
        '[{"region": "north", "amount": 10}, {"region": "south", "amount": 30}]'
    )
    (task / "sea.csv").write_text(
        "Game,Air Temperature,Sea Surface Temperature,Year\n"
        '"Zelda,Nintendo",20.5,18.0,1998\n'
        '"Halo,Microsoft",22.0,19.5,2001\n'
        '"Mario,Nintendo",21.0,18.5,1998\n'
    )
    (task / "owners.csv").write_text("region,owner\nnorth,Ana\nsouth,Luis\n")
    other = root / "tasks" / "t2"
    other.mkdir()
    (other / "other.csv").write_text("x\n1\n")
    (root / "secret.txt").write_text(FAKE_TOKEN + "\n")
    for path in [root, *root.rglob("*")]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def task_dir(data_dir):
    return str(data_dir / "tasks" / "t1")


@pytest.fixture()
def session(task_dir, monkeypatch):
    monkeypatch.setenv(
        "HF_TOKEN", FAKE_TOKEN
    )  # in the server: must not reach the worker
    s = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    yield s
    s.close()


def run(session, code):
    stdout, stderr, exit_code = session.run(code)
    return exit_code, stdout, stderr


def load(session, task_dir, name="sea.csv", var="df"):
    path = os.path.join(task_dir, name)
    code, _, err = run(session, f"import pandas as pd\n{var} = pd.read_csv({path!r})")
    assert code == 0, err


# --- Real pandas: what an AST interpreter rejects ---------------------------------------------


def test_reads_csv_with_pandas(session, task_dir):
    path = os.path.join(task_dir, "sales.csv")
    code, out, err = run(
        session,
        f"import pandas as pd\ndf = pd.read_csv({path!r})\n"
        "print(df.groupby('region').amount.sum().to_dict())",
    )
    assert code == 0, err
    assert "{'north': 30, 'south': 30}" in out


@pytest.mark.parametrize(
    "ext,reader",
    [("tsv", "pd.read_csv({p!r}, sep='\\t')"), ("json", "pd.read_json({p!r})")],
)
def test_reads_other_formats(session, task_dir, ext, reader):
    p = os.path.join(task_dir, f"sales.{ext}")
    code, out, err = run(
        session,
        "import pandas as pd\nprint(int(" + reader.format(p=p) + ".amount.sum()))",
    )
    assert code == 0, err
    assert out.strip() == "40"


def test_list_indexing_and_str_split(session, task_dir):
    load(session, task_dir)
    code, out, err = run(
        session,
        "sub = df[['Air Temperature', 'Sea Surface Temperature']]\nprint(sub.shape)\n"
        "print(df['Game'].str.split(',', expand=True)[0].tolist())",
    )
    assert code == 0, err
    assert out.splitlines() == ["(3, 2)", "['Zelda', 'Halo', 'Mario']"]


def test_groupby_agg_and_merge(session, task_dir):
    load(session, task_dir)
    load(session, task_dir, "sales.csv", "sales")
    load(session, task_dir, "owners.csv", "owners")
    code, out, err = run(
        session,
        "g = df.groupby('Year').agg(air=('Air Temperature', 'mean'), n=('Game', 'count'))\n"
        "m = sales.merge(owners, on='region', how='left')\n"
        "print(g.loc[1998, 'air'], int(g.loc[1998, 'n']), m.groupby('owner').amount.sum().to_dict())",
    )
    assert code == 0, err
    assert out.strip() == "20.75 2 {'Ana': 30, 'Luis': 30}"


def test_scikit_learn_works(session, task_dir):
    load(session, task_dir, "sales.csv", "sales")
    code, out, err = run(
        session,
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import cross_val_score\n"
        "import numpy as np\n"
        "X = np.arange(20).reshape(-1, 1)\ny = (X[:, 0] > 9).astype(int)\n"
        "float(cross_val_score(LogisticRegression(solver='newton-cg'), X, y, cv=2).mean())",
    )
    assert code == 0, err
    assert 0.0 <= float(out) <= 1.0


def test_state_persists_between_calls(session):
    assert run(session, "def double(v):\n    return 2 * v\nx = 3")[0] == 0
    assert run(session, "double(x)")[1] == "6\n"


def test_pyarrow_is_not_loaded_in_the_worker(session):
    code, out, err = run(session, "import sys\n'pyarrow' in sys.modules")
    assert (code, out) == (0, "False\n"), err


# --- Notebook-style output and errors ----------------------------------------------------------


def test_only_the_last_expression_is_printed(session):
    assert run(session, "1 + 1\nx = 5\nx * 2")[:2] == (0, "10\n")
    assert run(session, "x = 5")[:2] == (0, "")
    assert run(session, "print('a')")[:2] == (0, "a\n")
    assert run(session, "'text'")[:2] == (0, "'text'\n")


def test_output_is_capped(session):
    code, out, _ = run(session, "print('x' * 100_000)")
    assert code == 0
    assert len(out) < 21_000 and out.endswith("(output truncated)")


def test_error_names_the_type_and_the_model_line(session, task_dir):
    load(session, task_dir)
    code, out, err = run(
        session, "print('before')\ncols = ['Air Temperature', 'Sea Temp']\ndf[cols]"
    )
    assert code == 1 and out == "before\n"
    assert err.splitlines()[-1].startswith("KeyError:") and "Sea Temp" in err
    assert "line 3: df[cols]" in err
    assert "site-packages" not in err and "data_analysis_env" not in err


def test_syntax_error_and_system_exit_keep_the_session(session):
    assert run(session, "keep = 1")[0] == 0
    code, _, err = run(session, "x = (1,\ny = 2")
    assert code == 1 and err.startswith("SyntaxError:")
    code, _, err = run(session, "import sys\nsys.exit(2)")
    assert code == 1 and "SystemExit: 2" in err
    code, _, err = run(session, "raise KeyboardInterrupt")
    assert code == 1 and "KeyboardInterrupt" in err
    code, _, err = run(session, "raise GeneratorExit")
    assert code == 1 and "GeneratorExit" in err
    assert run(session, "keep")[:2] == (0, "1\n")


def test_warnings_reach_the_model_but_are_not_errors(session):
    code, out, err = run(session, "import numpy as np\nprint(np.mean(np.array([])))")
    assert code == 0, err
    assert out.strip() == "nan" and "RuntimeWarning" in err


def test_python_session_in_process():
    s = PythonSession()
    assert s.run("a = [1, 2]\nsum(a)") == ("3\n", "", 0)
    out, err, code = s.run("a[5]")
    assert code == 1 and err.splitlines()[-1] == "IndexError: list index out of range"
    assert json.dumps(s.run("'\\udc80'"))  # lone surrogates survive the JSON reply


# --- What model code can't do -------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "import ctypes",
        "import _ctypes",
        "import cffi",
        "import _posixsubprocess",
        "import pyarrow",
        "import pandas as pd\npd.read_csv('{other}', engine='pyarrow')",
        "import numpy as np\nnp.ctypeslib.ctypes.CDLL(None)",
        "import subprocess\nsubprocess.run(['echo', 'pwned'])",
        "import subprocess\nsubprocess._fork_exec(['echo', 'pwned'])",
        "import os\nos.system('echo pwned')",
        "import os\nos.popen('echo pwned').read()",
        "import os\nos.fork()",
        "import os\nos.execv('/bin/echo', ['echo', 'pwned'])",
        "import os\nos.kill(os.getppid(), 9)",
        "import socket\nsocket.socket()",
        "import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=1)",
        "import socket\nsocket.getaddrinfo('huggingface.co', 443)",
        "import urllib.request\nurllib.request.urlopen('https://huggingface.co/robots.txt')",
        "import pandas as pd\npd.read_csv('https://huggingface.co/robots.txt')",
        "open('/etc/hosts').read()",
        "open('/etc/passwd', 'rb').read()",
        "open('{secret}').read()",
        "open('{other}').read()",
        "import pandas as pd\npd.read_csv('{secret}')",
        "import os\nos.listdir('/')",
        "import os\nos.listdir('{tasks}')",
        "import os\nos.chdir('/')",
        "import os\nos.putenv('X', '1')",
        "import os\nos.symlink('/etc/passwd', 'link')",
        "import gc\ngc.get_referrers(1)",
        "import gc\ngc.get_objects()",
        "import sys\nsys.settrace(lambda *a: None)",
        "(lambda: 0).__code__.replace(co_consts=(1,))",
        "import shutil\nshutil.copy('/etc/hosts', 'x')",
        "import sqlite3\nsqlite3.connect('/etc/passwd')",
        "input()",
    ],
)
def test_forbidden(session, data_dir, snippet):
    snippet = snippet.format(
        secret=data_dir / "secret.txt",
        other=data_dir / "tasks" / "t2" / "other.csv",
        tasks=data_dir / "tasks",
    )
    code, out, err = run(session, snippet)
    assert code != 0, (out, err)
    assert "pwned" not in out and FAKE_TOKEN not in out + err


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "/proc/self/environ", "/proc/1/environ"]
)
def test_pandas_cannot_read_system_files(session, path):
    code, out, _ = run(
        session,
        f"import pandas as pd\nprint(pd.read_csv({path!r}, header=None, sep='\\x01'))",
    )
    assert code != 0, out


@pytest.mark.parametrize(
    "snippet",
    [
        # Patching what the hook could look up at call time.
        (
            "import os\nh = os.path.realpath(os.environ['HOME'])\n"
            + "os.path.realpath = lambda p, **k: h + '/x'\nopen('/etc/hosts').read()"
        ),
        "import os\nos.lstat = lambda p: None\nopen('/etc/hosts').read()",
        (
            "import sys\nm = sys.modules['data_analysis_env.server.sandbox']\n"
            + "m._BLOCKED_EVENT_PREFIXES = ()\nm.BLOCKED_IMPORTS = ()\nimport socket\nsocket.socket()"
        ),
        "import builtins\nbuiltins.isinstance = lambda *a: False\nopen('/etc/hosts').read()",
        # A path-like object that shows the hook one path and the opener another.
        (
            "class P:\n    n = 0\n    def __fspath__(self):\n        P.n += 1\n"
            + "        return 'x' if P.n > 1 else '/etc/hosts'\nopen(P()).read()"
        ),
        (
            "import sqlite3\nclass S(str):\n    def __eq__(s, o):\n        return True\n"
            + "sqlite3.connect(S('/etc/x.db'))"
        ),
        # A relative path that means something else under another directory's fd.
        "import os\nfd = os.open('.', os.O_RDONLY)\nos.open('../../etc/hosts', os.O_RDONLY, dir_fd=fd)",
    ],
)
def test_hook_does_not_trust_patchable_code(session, snippet):
    code, out, err = run(session, snippet)
    assert code != 0, (out, err)
    assert "localhost" not in out


def test_pathlib_works_in_home(session):
    code, out, err = run(
        session,
        "import pathlib\np = pathlib.Path('notes.txt')\np.write_text('ok')\np.read_text()",
    )
    assert (code, out) == (0, "'ok'\n"), err


def test_violation_message_is_short_and_clear(session):
    code, _, err = run(session, "import ctypes")
    assert code == 1
    assert (
        err.splitlines()[-1]
        == "SandboxViolation: sandbox: import of ctypes is not allowed"
    )


def test_environment_is_an_allowlist(session):
    code, out, err = run(session, "import os\ndict(os.environ)")
    assert code == 0, err
    env = eval(out)  # noqa: S307 - our own worker's repr of a dict of strings
    assert "HF_TOKEN" not in env and FAKE_TOKEN not in out
    assert set(env) <= set(sandbox.WORKER_ENV_ALLOWLIST) | set(sandbox.THREAD_ENV) | {
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "no_proxy",
    }


def test_writes_only_under_the_session_home(session, task_dir):
    code, out, err = run(
        session,
        "import os, tempfile, pandas as pd\n"
        "pd.DataFrame({'a': [1]}).to_csv('out.csv', index=False)\n"
        "os.mkdir('sub')\nos.rename('out.csv', 'sub/out.csv')\n"
        "fd, p = tempfile.mkstemp()\nos.close(fd)\nos.remove(p)\n"
        "print(os.getcwd() == os.path.realpath(os.environ['HOME']), pd.read_csv('sub/out.csv').a.sum())",
    )
    assert (code, out) == (0, "True 1\n"), err
    target = os.path.join(task_dir, "written.csv")
    for snippet in (
        f"import pandas as pd\npd.DataFrame({{'a': [1]}}).to_csv({target!r})",
        f"import os\nos.open({target!r}, os.O_WRONLY | os.O_CREAT)",
        "open('/tmp/data_analysis_env_probe', 'w').write('x')",
        f"import os\nos.remove({os.path.join(task_dir, 'sales.csv')!r})",
    ):
        assert run(session, snippet)[0] != 0, snippet
    assert not os.path.exists(target)
    assert os.path.exists(os.path.join(task_dir, "sales.csv"))


def test_replace_is_checked_like_rename(session, task_dir):
    """os.replace raises the os.rename audit event, so it is limited to the session home."""
    code, out, err = run(
        session,
        "import os\nopen('a', 'w').write('1')\nos.replace('a', 'b')\nopen('b').read()",
    )
    assert (code, out) == (0, "'1'\n"), err
    target = os.path.join(task_dir, "replaced.csv")
    code, _, err = run(session, f"import os\nos.replace('b', {target!r})")
    assert code != 0 and "not allowed" in err
    assert not os.path.exists(target)


def test_no_writes_through_a_directory_fd_outside_home(session, task_dir):
    target = os.path.join(task_dir, "via_dir_fd.csv")
    code, _, err = run(
        session,
        f"import os\nfd = os.open({task_dir!r}, os.O_RDONLY)\n"
        + "os.close(os.open('via_dir_fd.csv', os.O_WRONLY | os.O_CREAT, dir_fd=fd))",
    )
    assert code != 0 and "opening a directory" in err, err
    assert not os.path.exists(target)
    # Directories in the home can still be opened and used as dir_fd.
    code, out, err = run(
        session,
        "import os\nos.mkdir('d')\nfd = os.open('d', os.O_RDONLY)\n"
        + "os.close(os.open('f', os.O_WRONLY | os.O_CREAT, dir_fd=fd))\nos.listdir('d')",
    )
    assert (code, out) == (0, "['f']\n"), err


def test_home_is_private_and_removed_on_close(task_dir):
    s = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    assert s.run("open('x', 'w').write('1')")[2] == 0
    home = s.home
    assert os.stat(home).st_mode & 0o777 == 0o700
    s.close()
    assert not os.path.exists(home)


def test_file_size_is_capped(session):
    code, _, err = run(
        session,
        "with open('big', 'wb') as f:\n"
        f"    f.write(b'x' * ({sandbox.DEFAULT_FILE_LIMIT_MB} * 1024 * 1024 + 1))",
    )
    assert code == 1 and "OSError" in err


def test_home_size_and_file_count_are_capped(session, monkeypatch):
    monkeypatch.setattr(sandbox, "HOME_LIMIT_MB", 1)
    monkeypatch.setattr(sandbox, "HOME_FILES_LIMIT", 50)
    assert run(session, "x = 1\nopen('small', 'w').write('x' * 1000)")[0] == 0
    home = session.home
    # Over the size cap while the call runs: stopped before its timeout.
    start = time.time()
    code, _, err = run(
        session,
        "open('big', 'wb').write(b'x' * 2 * 1024 * 1024)\nimport time\ntime.sleep(60)",
    )
    assert code == 1 and "working directory went over" in err, err
    assert time.time() - start < TIMEOUT_S
    assert not os.path.exists(home)
    # The session restarted with an empty home.
    assert run(session, "import os\nos.listdir('.')")[:2] == (0, "[]\n")
    assert run(session, "x")[0] == 1
    # Over the file count cap when the call ends.
    code, _, err = run(session, "for i in range(100):\n    open(f'f{i}', 'w').close()")
    assert code == 1 and "working directory went over" in err, err


# --- The worker process ---------------------------------------------------------------------------


def test_timeout_kills_and_restarts_the_session(session):
    assert run(session, "x = 41")[0] == 0
    pid = session.pid
    start = time.time()
    code, _, err = run(session, "import time\ntime.sleep(60)")
    assert code == 1 and "TimeoutError" in err
    assert time.time() - start < TIMEOUT_S + 3
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    code, _, err = run(session, "print(x)")  # the session was restarted
    assert code == 1 and "NameError" in err
    assert run(session, "1 + 1")[:2] == (0, "2\n")


def test_a_stalled_reply_times_out_and_the_session_restarts(session):
    """Model code that writes part of a reply frame and stalls must not block the server."""
    stall = (
        "import os, struct, sys, time\n"
        "f = sys._getframe()\n"
        "while 'conn' not in f.f_locals:\n"
        "    f = f.f_back\n"
        "os.write(f.f_locals['conn'].fileno(), struct.pack('!i', 1000) + b'x')\n"
        "time.sleep(60)\n"
    )
    result = []
    start = time.time()
    caller = threading.Thread(target=lambda: result.append(run(session, stall)))
    caller.start()
    caller.join(TIMEOUT_S + 3)
    assert result, "the call did not return"
    code, _, err = result[0]
    assert code == 1 and "TimeoutError" in err
    assert time.time() - start < TIMEOUT_S + 3
    assert run(session, "1 + 1")[:2] == (0, "2\n")


def test_the_timeout_never_signals_the_worker_by_pid(session, monkeypatch):
    """A worker's pid can be reused once the spawner reaps it, so the timer thread must not kill by pid."""
    assert run(session, "x = 1")[0] == 0
    caller = threading.current_thread()
    from_other_threads = []
    real_kill = os.kill

    def kill(pid, sig):
        if threading.current_thread() is not caller:
            from_other_threads.append(pid)
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)
    code, _, err = run(session, "import time\ntime.sleep(60)")
    assert code == 1 and "TimeoutError" in err
    assert from_other_threads == []


def test_a_slow_home_check_after_the_reply_keeps_the_session(session, monkeypatch):
    """The timer stops once the reply is read, so a slow home check can't kill a finished call's worker."""
    check = sandbox._home_over_limit
    monkeypatch.setattr(
        sandbox,
        "_home_over_limit",
        lambda home: time.sleep(TIMEOUT_S + 1) or check(home),
    )
    assert run(session, "x = 5\nprint(1)")[:2] == (0, "1\n")
    monkeypatch.setattr(sandbox, "_home_over_limit", check)
    assert run(session, "print(x)")[:2] == (0, "5\n")


def test_worker_exit_is_a_crash_and_the_session_restarts(session):
    code, _, err = run(session, "import os\nos._exit(0)")
    assert code == 1 and "crashed" in err
    assert run(session, "1 + 1")[:2] == (0, "2\n")


def test_close_stops_the_worker(task_dir):
    s = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    assert s.run("print(1)")[2] == 0
    pid = s.pid
    os.kill(pid, 0)
    s.close()
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_sessions_are_separate_processes(task_dir):
    a = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    b = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    try:
        assert a.run("secret = 1")[2] == 0
        assert b.run("secret")[2] == 1
        assert a.pid != b.pid and a.home != b.home
        # b can't read a's home through the hook either
        assert b.run(f"open({os.path.join(a.home, 'x')!r}, 'w')")[2] == 1
    finally:
        a.close()
        b.close()


def test_replies_are_json_not_pickle(session):
    """A worker taken over by model code can only send JSON the server validates."""
    evil = (
        "import sys, pickle\n"
        "f = sys._getframe()\n"
        "while f is not None and 'conn' not in f.f_locals:\n"
        "    f = f.f_back\n"
        "f.f_locals['conn'].send_bytes(pickle.dumps(object()))\n"
    )
    code, _, err = run(session, evil)
    assert code == 1 and "crashed" in err
    assert run(session, "1")[:2] == (0, "1\n")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux prctl")
def test_workers_and_spawner_are_not_dumpable(session):
    assert run(session, "1")[0] == 0
    for pid in (session.pid, sandbox._get_spawner().pid):
        # Non-dumpable: /proc/<pid>/mem belongs to root whatever the process uid.
        assert os.stat(f"/proc/{pid}/mem").st_uid == 0


_SOCKETS_AFTER_FILTER = """
import socket, sys
from data_analysis_env.server.sandbox import block_sockets
block_sockets()
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    try:
        socket.socket(family, socket.SOCK_STREAM).close()
        print("OPENED", family)
    except PermissionError:
        print("DENIED")
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux seccomp")
def test_workers_cannot_create_sockets_without_the_hook(session):
    """The seccomp filter alone (this process has no audit hook) refuses every socket."""
    r = subprocess.run(
        [sys.executable, "-c", _SOCKETS_AFTER_FILTER],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.stdout.split() == ["DENIED"] * 3, r.stdout + r.stderr
    status = sandbox.sandbox_status()
    assert status["socket_filter"] is True, status
    assert run(session, "1")[0] == 0
    proc_status = Path(f"/proc/{session.pid}/status").read_text()
    assert "\nSeccomp:\t2\n" in proc_status


# --- The OS boundary (Linux, root) ---------------------------------------------------------------


def _as_uid(uid: int, code: str, *args: str) -> subprocess.CompletedProcess:
    """Run Python as `uid` with no audit hook: what code that bypassed the hook could do."""
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        user=uid,
        group=uid,
        extra_groups=[],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        cwd="/",
    )


@root_only
def test_each_session_runs_as_its_own_unprivileged_user(task_dir):
    a = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    b = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    try:
        ua = a.run("import os\nos.getresuid(), os.getgroups()")[0]
        ub = b.run("import os\nos.getresuid()")[0]
        assert a.uid != b.uid and 0 not in (a.uid, b.uid)
        assert ua.strip() == f"(({a.uid}, {a.uid}, {a.uid}), [])"
        assert ub.strip() == f"({b.uid}, {b.uid}, {b.uid})"
        # Without the hook, a's uid can't read b's home, signal b's worker or the server.
        assert b.run("open('note', 'w').write('b')")[2] == 0
        r = _as_uid(a.uid, "import sys; open(sys.argv[1]).read()", f"{b.home}/note")
        assert "PermissionError" in r.stderr
        r = _as_uid(a.uid, "import os, sys; os.kill(int(sys.argv[1]), 9)", str(b.pid))
        assert "PermissionError" in r.stderr
        r = _as_uid(a.uid, f"open('/proc/{os.getpid()}/environ').read()")
        assert "PermissionError" in r.stderr
        r = _as_uid(
            a.uid, "import sys; open(sys.argv[1], 'a')", f"{task_dir}/sales.csv"
        )
        assert "PermissionError" in r.stderr
    finally:
        a.close()
        b.close()


@root_only
def test_closing_a_session_kills_its_strays_and_temp_files(task_dir):
    s = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    assert s.run("1")[2] == 0
    uid = s.uid
    stray = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        user=uid,
        group=uid,
        extra_groups=[],
        env={},
    )
    # Every world-writable directory of the image, where it exists here.
    dirs = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/dev/mqueue")
    strays = [f"{d}/data_analysis_env_stray" for d in dirs if os.path.isdir(d)]
    r = _as_uid(uid, "import sys\nfor p in sys.argv[1:]: open(p, 'w')", *strays)
    assert r.returncode == 0, r.stderr
    assert all(os.path.exists(p) for p in strays)
    s.close()
    assert stray.wait(timeout=10) == -signal.SIGKILL
    assert not any(os.path.exists(p) for p in strays)


@root_only
def test_a_uid_sweep_leaves_the_session_home_to_the_server(task_dir):
    """The spawner sweeps a uid under a time limit, so it must not walk the home (any size)."""
    s = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    assert s.run("open('note', 'w').write('x')")[2] == 0
    home = s.home
    try:
        sandbox._get_spawner().sweep(s.uid)
        assert os.path.exists(os.path.join(home, "note"))
    finally:
        s.close()
    assert not os.path.exists(home)


_SYSV_IPC = """
import ctypes
libc = ctypes.CDLL(None)
print(libc.shmget(0, 4096, 0o600), libc.msgget(0, 0o600), libc.semget(0, 1, 0o600))
"""


def _ipc_ids(kind: str) -> set:
    rows = Path(f"/proc/sysvipc/{kind}").read_text().splitlines()[1:]
    return {row.split()[1] for row in rows}


@root_only
def test_closing_a_session_removes_its_sysv_ipc_objects(task_dir):
    s = SandboxSession(read_dirs=[task_dir], timeout_s=TIMEOUT_S)
    assert s.run("1")[2] == 0
    r = _as_uid(s.uid, _SYSV_IPC)
    created = dict(zip(("shm", "msg", "sem"), r.stdout.split()))
    assert len(created) == 3 and "-1" not in created.values(), r.stdout + r.stderr

    def alive():
        return [kind for kind, i in created.items() if i in _ipc_ids(kind)]

    assert alive() == ["shm", "msg", "sem"]
    libc = ctypes.CDLL(None)
    servers = libc.shmget(0, 4096, 0o600)  # another user's: must stay
    s.close()
    assert alive() == []
    assert str(servers) in _ipc_ids("shm")
    libc.shmctl(servers, sandbox.IPC_RMID, None)
