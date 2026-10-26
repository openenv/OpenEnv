"""What can this server actually run, right now?

A client should not have to guess. `capabilities()` answers three questions in one call — which
harnesses exist and how well each is trusted, which sandboxes have working credentials, and which
datasets are served — so the failure "you asked for a sandbox whose API key is missing" happens at
discovery time instead of 90 seconds into a rollout.

Credential checks reuse Harbor's own `preflight()` per backend rather than reimplementing key
lookups. Two things about that call have to be handled and are easy to miss:

  * it raises **`SystemExit`**, a `BaseException`, so a bare `except Exception` will not catch it and
    the server process dies instead of reporting an unavailable sandbox
  * `EnvironmentFactory.run_preflight` is never called by `Job.create()` — only Harbor's CLI calls
    it — so a library caller gets no credential validation at all unless it asks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import seams

# Harbor agents that run HOST-SIDE, in this server's process, rather than being installed into the
# sandbox. The distinction is not cosmetic: their LLM traffic never leaves the host, so they reach
# the capture proxy on localhost and need no public URL at all.
HOST_SIDE_AGENTS = frozenset({"terminus-2", "computer-1", "oracle", "nop", "dspy-rlm"})

# Backends worth advertising. Harbor registers 23; these are the ones with a credential story we
# check and have exercised. Others still work via `--sandbox <name>`, just unadvertised.
KNOWN_SANDBOXES = ("docker", "e2b", "modal", "daytona", "hf-sandbox")


@dataclass
class SandboxStatus:
    name: str
    available: bool
    detail: str = ""


@dataclass
class HarnessStatus:
    name: str
    dialect: str
    kind: str  # "installed" (runs in the sandbox) | "base" (runs host-side)
    # "validated" | "unstable:<why>" | "unsupported:<why>" | "blocked:<why>" | "untested".
    # Only "validated" is offered for comparison work; see seams.py for what each one means.
    status: str
    needs_subclass: bool  # True when Harbor's wrapper cannot be configured as shipped
    notes: str = ""


@dataclass
class Capabilities:
    harnesses: list[HarnessStatus] = field(default_factory=list)
    sandboxes: list[SandboxStatus] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    llm: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "harnesses": [vars(h) for h in self.harnesses],
            "sandboxes": [vars(s) for s in self.sandboxes],
            "datasets": self.datasets,
            "llm": self.llm,
        }

    @property
    def available_sandboxes(self) -> list[str]:
        return [s.name for s in self.sandboxes if s.available]

    @property
    def validated_harnesses(self) -> list[str]:
        return [h.name for h in self.harnesses if h.status == "validated"]

    def render(self, *, verbose: bool = False) -> str:
        """Human-readable summary, printed at server start.

        Deliberately shows what is NOT available and why, not just what is. A sandbox missing its
        API key is the single most common reason a rollout dies 90 seconds in, and Harbor's own
        preflight message names the exact variable — so it is worth surfacing at startup rather than
        making someone read a traceback later.

        Args:
            verbose (`bool`, *optional*, defaults to `False`):
                List every harness instead of only the validated ones.

        Returns:
            `str`: A multi-line report.
        """
        out: list[str] = []

        if self.llm:
            model = self.llm.get("model", "?")
            ok = self.llm.get("ok")
            level = self.llm.get("capture_level") or ""
            if ok:
                mark = "TRAIN"
            elif self.llm.get("reachable"):
                # Loud, and in the same column as a failure, because this is the line that decides
                # whether anything this server produces can be trained on. It is not an error, and it
                # must not read as a success either.
                mark = "EVAL ONLY"
            elif ok is False:
                mark = "FAILED"
            else:
                mark = "unchecked"
            out.append(f"llm       {model}  [{mark}]")
            if self.llm.get("url"):
                auth = " (authenticated)" if self.llm.get("authenticated") else ""
                out.append(f"          {self.llm['url']}{auth}")
            if mark == "TRAIN" and self.llm.get("logprobs_mode"):
                # Worth a line even when everything is fine: it is the one property of a trainable
                # endpoint that is invisible in the data and wrong by default.
                out.append(
                    f"          logprobs: {self.llm['logprobs_mode']}"
                    + (
                        "  (temperature-scaled, as training needs)"
                        if self.llm["logprobs_mode"] == "processed"
                        else ""
                    )
                )
            if mark == "EVAL ONLY":
                if self.llm.get("logprobs_mode") == "raw":
                    # Demoted for a different reason than the tier name suggests: this endpoint has
                    # token ids, and saying "no token ids" about it would be simply untrue.
                    detail = "token ids present but logprobs are RAW (pre-temperature)"
                elif level == "logprobs":
                    detail = "logprobs, no token ids"
                else:
                    detail = "no token ids, no logprobs"
                out.append(
                    f"          {detail} — rollouts carry reward and trace but are NOT trainable"
                )
            for fix in self.llm.get("param_fixes") or []:
                # A rewritten request is a changed experiment: dropping `temperature` alters the
                # sampling distribution, so it cannot be a silent accommodation.
                out.append(f"          upstream compat: {fix}")
            # Anything the probe flagged. A `[TRAIN]` endpoint can still carry a warning worth
            # reading — raw rather than processed logprobs being the one that looks perfect and
            # trains on the wrong importance ratio — so these print at every level, not just on
            # failure. INFO is dropped; it is detail, not a decision.
            for finding in self.llm.get("findings") or []:
                if not str(finding).startswith("[INFO]"):
                    out.append(f"          {_wrap_finding(str(finding))}")

        usable = [s for s in self.sandboxes if s.available]
        out.append(f"\nsandboxes {len(usable)} of {len(self.sandboxes)} usable")
        for s in self.sandboxes:
            if s.available:
                out.append(f"  [ok]   {s.name}")
            else:
                out.append(f"  [--]   {s.name:<10} {s.detail[:88]}")

        if self.datasets:
            total = sum(d.get("num_tasks", 0) for d in self.datasets)
            out.append(f"\ndatasets  {len(self.datasets)} split(s), {total} tasks")
            for d in self.datasets:
                if d.get("error"):
                    out.append(f"  [--]   {d['name']:<48} {d['error'][:60]}")
                else:
                    out.append(
                        f"  [ok]   {d['name']:<48} {d.get('num_tasks', 0):>6} tasks"
                    )

        validated = [h for h in self.harnesses if h.status == "validated"]
        out.append(
            f"\nharnesses {len(validated)} validated of {len(self.harnesses)} known"
        )
        shown = self.harnesses if verbose else validated
        by_dialect: dict[str, list[str]] = {}
        for h in shown:
            label = h.name + ("*" if h.kind == "base" else "")
            by_dialect.setdefault(h.dialect, []).append(label)
        for dialect, names in sorted(by_dialect.items()):
            out.append(f"  {dialect:<18} {', '.join(sorted(names))}")
        out.append("  (* runs host-side in this process, so it needs no public URL)")

        if not usable:
            out.append(
                "\nWARNING: no sandbox has working credentials; every rollout will fail."
            )
        return "\n".join(out)


def _wrap_finding(text: str, width: int = 92, indent: str = " " * 10) -> str:
    """Wrap one finding to the terminal, keeping the continuation under the same column.

    The long ones matter most and are the ones a single line truncates into uselessness.
    """
    import textwrap

    lines = textwrap.wrap(text, width=width) or [text]
    return ("\n" + indent).join(lines)


def _missing_sdk(environment_class: Any) -> str:
    """Report the backend's own verdict on whether its SDK is importable.

    Loading the class is not enough. Every Harbor backend guards its provider SDK with a
    module-level `try: import ... except ImportError: _HAS_X = False`, and raises `MissingExtraError`
    from `__init__` rather than at import. So the module imports cleanly, the class loads cleanly,
    the backend reports available, and the failure arrives only once a rollout tries to build a
    sandbox, by which time it reads as a broken rollout rather than a missing dependency.

    Reading the flag asks the backend the same question its constructor will ask, before offering it.

    Returns:
        `str`: A message naming the missing extra, or `""` when the SDK is present.
    """
    import sys

    module = sys.modules.get(getattr(environment_class, "__module__", ""), None)
    if module is None:
        return ""
    absent = sorted(
        flag
        for flag, value in vars(module).items()
        if flag.startswith("_HAS_") and value is False
    )
    if not absent:
        return ""
    extras = ", ".join(flag.removeprefix("_HAS_").lower() for flag in absent)
    # Deliberately not `harbor[cloud]`: that extra cannot be installed at all, because it pulls
    # `langsmith[sandbox]` and `tensorlake` together and they demand incompatible `websockets`
    # ranges. Pointing someone at it would send them to a resolver error.
    return (
        f"SDK not installed ({extras}). Install it with: pip install 'openenv[harbor]', "
        "which pulls every sandbox backend."
    )


def check_sandbox(name: str) -> SandboxStatus:
    """Ask Harbor whether this backend's credentials are present.

    `SystemExit` is caught explicitly: Harbor's preflight raises it as its failure signal, and it
    does not inherit from `Exception`.
    """
    try:
        from harbor.environments.factory import (
            _load_environment_class,
            EnvironmentFactory,
        )
        from harbor.models.environment_type import EnvironmentType
    except ImportError as exc:
        return SandboxStatus(name, False, f"harbor not installed: {exc}")

    try:
        env_type = EnvironmentType(name)
    except ValueError:
        return SandboxStatus(name, False, f"unknown environment type {name!r}")

    # Credentials are only half of it. Harbor's preflight checks env vars but never imports the
    # backend, so a provider with perfect credentials and no SDK installed reports "available" and
    # then fails at rollout time with MissingExtraError. Load the class first.
    try:
        environment_class = _load_environment_class(env_type)
    except Exception as exc:  # noqa: BLE001 - Harbor raises its own MissingExtraError here
        hint = str(exc).replace("\n", " ")[:160]
        return SandboxStatus(
            name, False, hint or f"backend {name!r} could not be loaded"
        )

    missing = _missing_sdk(environment_class)
    if missing:
        return SandboxStatus(name, False, missing)

    try:
        EnvironmentFactory.run_preflight(env_type)
    except SystemExit as exc:
        # Harbor's own message names the missing variable; pass it through rather than paraphrase.
        return SandboxStatus(name, False, str(exc) or "credentials missing")
    except ImportError as exc:
        return SandboxStatus(
            name, False, f"extra not installed: pip install 'harbor[{name}]' ({exc})"
        )
    except Exception as exc:  # noqa: BLE001 - an unexpected failure is still "not available"
        return SandboxStatus(name, False, f"{type(exc).__name__}: {str(exc)[:160]}")
    return SandboxStatus(name, True)


def list_harnesses() -> list[HarnessStatus]:
    """Every harness with a seam, and how much to trust it."""
    out: list[HarnessStatus] = []
    for name in sorted(set(seams.SEAMS)):
        seam = seams.get(name)
        out.append(
            HarnessStatus(
                name=name,
                dialect=seam.dialect,
                kind="base" if name in HOST_SIDE_AGENTS else "installed",
                status=seam.status,
                needs_subclass=seam.import_path is not None,
                notes=seam.notes,
            )
        )
    return out


def capabilities(
    *,
    datasets: list[str] | None = None,
    sandboxes: tuple[str, ...] = KNOWN_SANDBOXES,
    llm: dict[str, Any] | None = None,
) -> Capabilities:
    """Full picture. Safe to call on a fresh instance; does no I/O beyond credential checks."""
    result = Capabilities(
        harnesses=list_harnesses(),
        sandboxes=[check_sandbox(s) for s in sandboxes],
        llm=dict(llm or {}),
    )
    if datasets:
        from .tasks import HarborTaskProvider

        result.datasets = HarborTaskProvider(datasets).list_splits()
    return result
