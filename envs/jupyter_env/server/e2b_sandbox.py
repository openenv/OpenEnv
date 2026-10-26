"""E2B Code Interpreter sandbox wrapper.

Wraps the E2B Code Interpreter SDK to provide a normalized execution interface
that decouples the rest of the environment from the E2B API surface.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

_E2B_IMPORT_ERROR: ImportError | None = None

try:
    from e2b_code_interpreter import Sandbox
except ImportError as _e2b_import_error:  # pragma: no cover
    _E2B_IMPORT_ERROR = _e2b_import_error
    Sandbox = None  # type: ignore[assignment]


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
