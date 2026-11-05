"""E2B Code Interpreter sandbox wrapper.

Wraps the E2B Code Interpreter SDK to provide a normalized execution interface
that decouples the rest of the environment from the E2B API surface.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, List, Optional

_E2B_IMPORT_ERROR: ImportError | None = None

try:
    from e2b_code_interpreter import Sandbox
except ImportError as _e2b_import_error:  # pragma: no cover
    _E2B_IMPORT_ERROR = _e2b_import_error
    Sandbox = None  # type: ignore[assignment]

# E2B's default code-interpreter template runs its Jupyter server, and so every
# notebook kernel, as root with the notebook in /home/user. Verification has
# always run from that kernel, so ``run_command`` keeps the same user and
# directory and verify commands see the same permissions they did before.
_COMMAND_USER = "root"
_COMMAND_CWD = "/home/user"


@dataclass
class CellResult:
    """Normalized result from a code or shell execution."""

    stdout: str
    stderr: str
    error: Optional[str]  # formatted traceback string, or None
    error_name: Optional[str]  # exception class name, or None
    text_results: List[str]  # text/plain representations of display outputs
    images: List[str]  # base64-encoded PNG strings
    execution_count: int
    success: bool


def _failed_command(exc: Exception, exit_code: int) -> CellResult:
    """Describe a command that exited non-zero as a failed `CellResult`.

    A non-zero exit raises the SDK's ``CommandExitException``, which carries the
    exit code and the output the command produced before it failed.
    """
    return CellResult(
        stdout=exc.stdout or "",
        stderr=exc.stderr or "",
        error=f"exit code {exit_code}",
        error_name=type(exc).__name__,
        text_results=[],
        images=[],
        execution_count=0,
        success=False,
    )


class E2BSandbox:
    """
    Manages a single E2B Code Interpreter sandbox session.

    One sandbox = one notebook kernel. Variables persist between ``run_code``
    calls within the same sandbox, matching Jupyter notebook semantics.

    Lifecycle:
        sbx = E2BSandbox(api_key=...)
        result = sbx.run_code("x = 42")
        result = sbx.run_code("print(x)")   # prints 42 — state persists
        sbx.kill()                           # terminates sandbox on episode end
    """

    # Setup code run once per sandbox to ensure matplotlib images are captured.
    # E2B's Jupyter kernel uses IPython's inline backend by default, which
    # captures plots as PNG via plt.show(). If user code calls
    # matplotlib.use('Agg') this breaks capture. We patch plt.show() to
    # always go through IPython.display so images appear in results.
    _SETUP_CODE = """\
import matplotlib.pyplot as plt

def _patched_show(*args, **kwargs):
    import matplotlib.pyplot as _plt
    from IPython.display import display as _disp, Image as _Img
    import io as _io
    figs = [_plt.figure(n) for n in _plt.get_fignums()]
    if not figs:
        return
    for fig in figs:
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        _disp(_Img(data=buf.read()))
    _plt.close('all')

plt.show = _patched_show
del _patched_show
"""

    def __init__(self, api_key: str):
        if Sandbox is None:
            raise ImportError(
                "e2b-code-interpreter is not installed. Install the "
                "jupyter_env package dependencies to use E2BSandbox. "
                f"Original import error: {_E2B_IMPORT_ERROR}"
            )
        # E2B SDK v1+: use Sandbox.create() factory, pass api_key via ApiParams
        self._sbx = Sandbox.create(api_key=api_key)
        self.sandbox_id: str = self._sbx.sandbox_id
        # Ensure matplotlib images are always captured
        self._sbx.run_code(self._SETUP_CODE)

    def run_code(self, code: str) -> CellResult:
        """Execute Python code in the persistent kernel, return normalized result."""
        execution = self._sbx.run_code(code)
        return self._normalize(execution)

    def run_shell(self, command: str, timeout_s: float = 120) -> CellResult:
        """
        Execute a shell command inside the sandbox.

        Implemented via subprocess inside the Python kernel so we can reuse
        the same E2B ``run_code`` path rather than a separate API call.
        """
        shell_code = (
            "import subprocess, sys\n"
            f"_result = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout={float(timeout_s)!r})\n"
            "print(_result.stdout, end='')\n"
            "if _result.stderr: print(_result.stderr, end='', file=sys.stderr)\n"
            "if _result.returncode != 0:\n"
            "    raise SystemExit(_result.returncode)\n"
        )
        return self.run_code(shell_code)

    def run_command(self, command: str, timeout_s: float = 120) -> CellResult:
        """
        Execute a shell command as its own sandbox process, outside the kernel.

        ``run_shell`` runs inside the notebook kernel, so it inherits whatever
        the notebook has done to it: a rebound ``subprocess.run``, a changed
        working directory, edited ``os.environ``. That is right for the agent's
        own shell tool and wrong for verification. This goes through E2B's
        process API instead and reports the command's real exit status.

        It removes the coupling to the notebook kernel. It is not an isolation
        boundary: the kernel runs as root, so agent code can still change what
        any process in the sandbox sees, including this command's login shell.

        Args:
            command (`str`):
                Shell command to run.
            timeout_s (`float`, *optional*, defaults to `120`):
                Seconds to wait for the command before giving up on it.

        Returns:
            [`CellResult`]: `success` is `True` only for a zero exit status.

        Raises:
            `Exception`: whatever the SDK raises when the command cannot be
                started, or when the wait for it ends without an exit status.
                Only a command that ran and exited is reported as a result:
                a sandbox that is unreachable, expired or unauthenticated says
                nothing about the work being verified.
        """
        # Started in the background so there is a handle to kill if the wait
        # fails: E2B keeps a command running when its connection drops.
        handle = self._sbx.commands.run(
            command,
            background=True,
            user=_COMMAND_USER,
            cwd=_COMMAND_CWD,
            timeout=timeout_s,
        )
        try:
            result = handle.wait()
        except Exception as exc:
            # The SDK raises CommandExitException for a command that ran and
            # exited non-zero. It is told apart by its exit code rather than by
            # class so that this module keeps importing, and the tests keep
            # driving it, without the SDK installed.
            exit_code = getattr(exc, "exit_code", None)
            if exit_code is not None:
                return _failed_command(exc, exit_code)
            # No exit status: the command timed out, or the connection to it
            # broke. Ask E2B to stop it, but a kill signals the command's own
            # process, so a child of it can stay alive and still write to the
            # sandbox. Nothing here can call the command finished, so the
            # failure is raised rather than counted as a verify result.
            with contextlib.suppress(Exception):
                handle.kill()
            raise
        return CellResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            error=None,
            error_name=None,
            text_results=[],
            images=[],
            execution_count=0,
            success=True,
        )

    def write_file(self, filename: str, content: bytes) -> None:
        """Upload a file into the sandbox filesystem."""
        self._sbx.files.write(filename, content)

    def kill(self) -> None:
        """Terminate the sandbox. Safe to call multiple times."""
        try:
            self._sbx.kill()
        except Exception:
            try:
                self._sbx.close()
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize(self, execution: Any) -> CellResult:
        stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
        stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""

        error: Optional[str] = None
        error_name: Optional[str] = None
        if execution.error:
            error_name = execution.error.name
            error = f"{execution.error.name}: {execution.error.value}\n{execution.error.traceback}"

        text_results: List[str] = []
        images: List[str] = []
        for r in execution.results or []:
            if r.text:
                text_results.append(r.text)
            if r.png:
                images.append(r.png)

        return CellResult(
            stdout=stdout,
            stderr=stderr,
            error=error,
            error_name=error_name,
            text_results=text_results,
            images=images,
            execution_count=execution.execution_count or 0,
            success=execution.error is None,
        )
