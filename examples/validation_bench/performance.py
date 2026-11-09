#!/usr/bin/env python3
"""Experiment 3: actual cached Docker builds, hardened launches and validation.

Run --plan-only first. Docker execution belongs to the coordinating agent. The
startup arm is a controlled cost comparison, never an admission decision.
"""

import argparse
import contextvars
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from openenv.validation import runner
from openenv.validation.providers import docker
from openenv.validation.runtime.contracts import LaunchSpec
from openenv.validation.types import Level


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
LAB_ROOT = Path("/tmp/openenv-validation-bench-20261109")
LABEL = "org.openenv.validation.run"
CURRENT = contextvars.ContextVar("performance_trial", default=None)
ARMS = ("startup", "full")
BUDGET = {"cpu": 1.0, "memory_mb": 256, "disk_mb": 64, "episode_timeout_s": 10.0}
PHASES = (
    "build",
    "start",
    "inspect",
    "collection",
    "oracle",
    "network",
    "containment",
    "resources",
    "probe_exec",
    "grading",
    "bundling",
    "cleanup",
    "cleanup_retry",
    "cleanup_verify",
    "other",
)
INSTALLED = False


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def file_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in immutable context: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def checksums(root):
    rows = file_hashes(root)
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{value}  {name}\n" for name, value in rows.items() if name != "SHA256SUMS"
        )
    )


class Trial:
    def __init__(self, row, directory):
        self.row = row
        self.directory = directory
        self.started = time.monotonic()
        self.thread = threading.get_ident()
        self.events = []
        self.commands = []
        self.errors = []
        self.subjects = []
        self.images = []
        self.local = threading.local()
        self.lock = threading.Lock()

    def error(self, phase, exc):
        with self.lock:
            self.errors.append(
                {"phase": phase, "type": type(exc).__name__, "message": str(exc)[:4096]}
            )

    @contextmanager
    def phase(self, name, subject=None):
        stack = self.local.stack = self.local.__dict__.get("stack", [])
        event = {
            "phase": name,
            "subject": subject,
            "thread": threading.get_ident(),
            "start_s": time.monotonic() - self.started,
        }
        wall, cpu = time.monotonic(), time.thread_time()
        frame = [0.0, 0.0]
        stack.append(frame)
        try:
            yield
        except BaseException as exc:
            event["exception"] = type(exc).__name__
            self.error(name, exc)
            raise
        finally:
            elapsed, consumed = time.monotonic() - wall, time.thread_time() - cpu
            stack.pop()
            event.update(
                wall_s=elapsed,
                thread_cpu_s=consumed,
                exclusive_wall_s=max(0.0, elapsed - frame[0]),
                exclusive_thread_cpu_s=max(0.0, consumed - frame[1]),
            )
            if stack:
                stack[-1][0] += elapsed
                stack[-1][1] += consumed
            with self.lock:
                self.events.append(event)


def timed(original, phase, trial=None, subject=None):
    @functools.wraps(original)
    def call(*args, **kwargs):
        active = trial if trial is not None else CURRENT.get()
        if active is None:
            return original(*args, **kwargs)
        token = CURRENT.set(active)
        try:
            with active.phase(phase, subject):
                return original(*args, **kwargs)
        finally:
            CURRENT.reset(token)

    return call


def command_summary(argv):
    # Retain executable arguments; large trusted probe programs use a content hash.
    return [
        value
        if len(value) < 512
        else {
            "bytes": len(value.encode()),
            "sha256": hashlib.sha256(value.encode()).hexdigest(),
        }
        for value in map(str, argv)
    ]


def install_instrumentation():
    """Install once, before workers; per-subject wrappers capture their trial."""
    global INSTALLED
    if INSTALLED:
        return
    for name, phase in (
        ("collect_runtime_evidence", "collection"),
        ("_oracle_probes", "oracle"),
        ("execute_graders", "grading"),
        ("write_runtime_bundle", "bundling"),
    ):
        setattr(runner, name, timed(getattr(runner, name), phase))
    original_init = docker.DockerRunningSubject.__init__

    def subject_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trial = CURRENT.get()
        if trial is None:
            return
        trial.subjects.append(self)
        for name, phase in (
            ("inspect", "inspect"),
            ("measure_network", "network"),
            ("measure_containment", "containment"),
            ("measure_resources", "resources"),
            ("exec", "probe_exec"),
            ("stop", "cleanup"),
        ):
            original = getattr(self, name)
            if name == "stop":
                original = logged_stop(self, original, trial)
            setattr(self, name, timed(original, phase, trial, self.name))

    docker.DockerRunningSubject.__init__ = subject_init
    original_command = docker._command

    @functools.wraps(original_command)
    def command(argv, *args, **kwargs):
        trial = CURRENT.get()
        if trial is None:
            return original_command(argv, *args, **kwargs)
        argv = list(argv)
        if trial.row.get("no_cache") and argv[:2] == ["docker", "build"]:
            argv.insert(2, "--no-cache")
        record = {
            "argv": command_summary(argv),
            "start_s": time.monotonic() - trial.started,
        }
        started = time.monotonic()
        try:
            result = original_command(argv, *args, **kwargs)
            record.update(
                exit_code=result[0],
                stdout_bytes=len(result[1].encode()),
                stderr_bytes=len(result[2].encode()),
            )
            if argv[:2] == ["docker", "build"]:
                record["stderr_tail"] = result[2][-4096:]
            return result
        except BaseException as exc:
            record["exception"] = type(exc).__name__
            raise
        finally:
            record["wall_s"] = time.monotonic() - started
            with trial.lock:
                trial.commands.append(record)

    docker._command = command
    INSTALLED = True


def logged_stop(subject, original, trial):
    attempted = False

    @functools.wraps(original)
    def stop():
        nonlocal attempted
        if not attempted and trial.row["kind"] == "failure":
            attempted = True
            try:
                log = subject.logs(max_bytes=8192)
                (trial.directory / (subject.name + ".log")).write_text(log)
            except Exception as exc:
                trial.error("subject_logs", exc)
        return original()

    return stop


def provider_for(trial):
    provider = docker.DockerValidationProvider(cleanup_subjects=[])
    build, start = provider.build, provider.start

    def build_image(*args, **kwargs):
        with trial.phase("build"):
            image = build(*args, **kwargs)
            trial.images.append(image)
            return image

    def start_subject(spec):
        spec = spec.model_copy(
            update={
                "run_id": trial.row["run_id"],
                "env_vars": {
                    "VALIDATION_FAULT": "hung_step"
                    if trial.row["fault"] == "sigint"
                    else trial.row["fault"]
                },
            }
        )
        with trial.phase("start"):
            return start(spec)

    provider.build, provider.start = build_image, start_subject
    return provider


def owned_ids(run_id):
    code, stdout, stderr = docker._command(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={LABEL}={run_id}",
        ],
        10,
    )
    if code:
        raise RuntimeError("Independent Docker listing failed: " + stderr[-1024:])
    return stdout.split()


def teardown(trial):
    """Retain initial leaks/errors even when narrowly scoped recovery succeeds."""
    with trial.phase("cleanup_verify"):
        try:
            initial = owned_ids(trial.row["run_id"])
        except Exception as exc:
            trial.error("cleanup_verify_initial", exc)
            initial = None
    recovery_started = time.monotonic()
    if initial:
        trial.error("cleanup_leak_after_stop", RuntimeError(str(initial)))
    if initial is None or initial:
        with trial.phase("cleanup_retry"):
            for subject in reversed(trial.subjects):
                try:
                    subject.stop()
                except BaseException as exc:
                    trial.error("cleanup_retry", exc)
            try:
                survivors = owned_ids(trial.row["run_id"])
                if survivors:
                    code, _, stderr = docker._command(
                        ["docker", "rm", "--force", "--volumes", *survivors], 15
                    )
                    if code:
                        raise RuntimeError(stderr[-1024:])
            except BaseException as exc:
                trial.error("cleanup_recovery", exc)
    with trial.phase("cleanup_verify"):
        try:
            final = owned_ids(trial.row["run_id"])
            if final:
                raise RuntimeError(
                    "Containers survived experiment cleanup: " + str(final)
                )
        except BaseException as exc:
            trial.error("cleanup_assert_empty", exc)
            final = None
    return {
        "initial_after_stop": initial,
        "final": final,
        "assert_empty": final == [],
        "recovery_and_verification_wall_s": time.monotonic() - recovery_started,
        "completed_at_s": time.monotonic() - trial.started,
    }


def run_trial(row, context, output):
    directory = output / "runs" / row["id"]
    directory.mkdir(parents=True, exist_ok=False)
    trial = Trial(row, directory)
    token = CURRENT.set(trial)
    cpu_started = time.thread_time()
    result = {**row, "verdict": None, "checks": {}, "runtime_status_counts": {}}
    running = None
    try:
        provider = provider_for(trial)
        if row["arm"] == "full":
            report = runner.run_validation(
                context,
                max_level=Level.RUNTIME,
                provider=provider,
                artifacts_dir=directory / "bundle",
                report_target="exp3:offline-served-probe",
            )
            serialized = report.model_dump(mode="json")
            save(directory / "report.json", serialized)
            result.update(
                verdict=serialized["verdict"],
                checks={
                    check["check_id"]: check["status"]
                    for check in serialized["results"]
                },
            )
            result["runtime_status_counts"] = dict(
                Counter(
                    check["status"]
                    for check in serialized["results"]
                    if check["check_id"].startswith("runtime.")
                )
            )
            result["all_status_counts"] = dict(
                Counter(check["status"] for check in serialized["results"])
            )
            result["not_selected_runtime_checks"] = [
                check
                for check in (
                    "runtime.rubric_introspectable",
                    "runtime.reward_attribution",
                    "runtime.task_declaration_accuracy",
                )
                if check not in result["checks"]
            ]
        else:
            manifest = (
                runner.default_parser_registry()
                .parser_for(runner.detect_signature(context))
                .parse(context)
            )
            image = provider.build(context, manifest.execution)
            running = provider.start(
                LaunchSpec(
                    image_ref=image,
                    resources=manifest.resources,
                    network=manifest.network,
                    run_id=row["run_id"],
                )
            )
            save(directory / "inspection.json", running.inspect())
            result["startup_ready"] = True
            result["admission_decision"] = False
    except BaseException as exc:
        trial.error("run", exc)
    finally:
        if running is not None:
            try:
                running.stop()
            except BaseException as exc:
                trial.error("startup_arm_stop", exc)
        # Full runner normally stops itself, but retain handles for exceptional exits.
        if result["verdict"] is None and row["arm"] == "full":
            for subject in reversed(trial.subjects):
                try:
                    subject.stop()
                except BaseException as exc:
                    trial.error("outer_stop", exc)
        result["cleanup"] = teardown(trial)
        result.update(
            wall_s=time.monotonic() - trial.started,
            validator_thread_cpu_s=time.thread_time() - cpu_started,
        )
        CURRENT.reset(token)
    result.update(
        phases=trial.events,
        commands=trial.commands,
        errors=trial.errors,
        image_refs=trial.images,
        subject_names=[subject.name for subject in trial.subjects],
    )
    result["phase_exclusive_wall_s"] = {
        name: sum(
            event["exclusive_wall_s"]
            for event in trial.events
            if event["phase"] == name and event["thread"] == trial.thread
        )
        for name in PHASES
        if name != "other"
    }
    result["phase_exclusive_wall_s"]["other"] = max(
        0.0, result["wall_s"] - sum(result["phase_exclusive_wall_s"].values())
    )
    bundle = directory / "bundle"
    result["artifact_bytes"] = sum(
        path.stat().st_size for path in bundle.rglob("*") if path.is_file()
    )
    result["retained_payload_bytes"] = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    if (bundle / "collector-trace.json").exists():
        result["preserved_operations"] = [
            exchange["operation"]
            for exchange in json.loads((bundle / "collector-trace.json").read_text())
        ]
        result["collector_completion"] = json.loads(
            (bundle / "collector-evidence.json").read_text()
        )
    if row["kind"] in {"warmup", "warm", "warm_padding"} and row["arm"] == "full":
        counts = result["all_status_counts"] if "all_status_counts" in result else {}
        result["expected_healthy_match"] = (
            result["runtime_status_counts"].get("pass") == 9
            and counts.get("skip") == 4
            and result["verdict"] == "warn"
            and not counts.get("fail")
            and not counts.get("error")
        )
        if not result["expected_healthy_match"]:
            result["errors"].append(
                {
                    "phase": "expected_verdict",
                    "message": "Baseline expects nine runtime PASS, four emitted SKIP, overall WARN",
                }
            )
    save(directory / "metrics.json", result)
    checksums(directory)
    return result


def quantiles(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1] if ordered else None,
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
    }


def summarize(rows, batches, failures=()):
    cells = []
    for concurrency in sorted(
        {row["concurrency"] for row in rows if row["kind"] == "warm"}
    ):
        for arm in ARMS:
            selected = [
                row
                for row in rows
                if row["kind"] == "warm"
                and row["concurrency"] == concurrency
                and row["arm"] == arm
            ]
            signatures = Counter(
                json.dumps(
                    {"verdict": row["verdict"], "checks": row["checks"]}, sort_keys=True
                )
                for row in selected
            )
            cells.append(
                {
                    "concurrency": concurrency,
                    "arm": arm,
                    "n": len(selected),
                    "wall_s": quantiles([row["wall_s"] for row in selected]),
                    "validator_thread_cpu_s": quantiles(
                        [row["validator_thread_cpu_s"] for row in selected]
                    ),
                    "artifact_bytes": quantiles(
                        [row["artifact_bytes"] for row in selected]
                    ),
                    "phases_exclusive_wall_s": {
                        phase: quantiles(
                            [row["phase_exclusive_wall_s"][phase] for row in selected]
                        )
                        for phase in PHASES
                    },
                    "error_runs": sum(bool(row["errors"]) for row in selected),
                    "verdict_signatures": dict(signatures),
                    "verdict_changes": max(0, len(signatures) - 1),
                }
            )
    return {
        "quantile_method": "nearest-rank; all completed samples including errors; no outlier removal",
        "cpu_scope": "time.thread_time; calling validator thread only; excludes engine, subject, child-process and other validator-thread CPU",
        "timing_scope": "raw spans inclusive; exclusive totals use calling validator thread and subtract same-thread children; total includes cleanup verification, excludes final metrics/checksums",
        "startup_scope": "controlled hardened startup cost comparison; no admission verdict",
        "cells": cells,
        "batches": batches,
        "errors": [
            {"id": row["id"], "errors": row["errors"]}
            for row in [*rows, *failures]
            if row["errors"]
        ],
        "all_cleanup_empty": bool(rows)
        and all(row["cleanup"]["assert_empty"] for row in rows)
        and all(row["parent_cleanup"]["assert_empty"] for row in failures),
    }


def plan(args, context_hashes):
    rng = random.Random(args.seed)
    study = json.loads(args.study.read_text())
    prefix = (
        "exp3-" + hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12]
    )
    batches, number = [], 0

    def batch(kind, arm, concurrency, count, fault="good", no_cache=False):
        nonlocal number
        rows = []
        for index in range(concurrency if kind == "warm" else count):
            number += 1
            run_kind = "warm_padding" if kind == "warm" and index >= count else kind
            name = f"{number:03d}-{run_kind}-{arm}-c{concurrency}"
            rows.append(
                {
                    "id": name,
                    "kind": run_kind,
                    "arm": arm,
                    "concurrency": concurrency,
                    "fault": fault,
                    "no_cache": no_cache,
                    "run_id": f"{prefix}-{number:03d}",
                }
            )
        return {"kind": kind, "arm": arm, "concurrency": concurrency, "runs": rows}

    warmups = list(ARMS)
    rng.shuffle(warmups)
    batches.extend(batch("warmup", arm, 1, 1) for arm in warmups)
    jobs = []
    for concurrency in args.concurrency:
        for arm in ARMS:
            for offset in range(0, args.repeats, concurrency):
                jobs.append((arm, concurrency, min(concurrency, args.repeats - offset)))
    rng.shuffle(jobs)
    batches.extend(batch("warm", *job) for job in jobs)
    for _ in range(args.cold_builds):
        batches.append(batch("cache_bypassing_build", "build", 1, 1, no_cache=True))
    failures = [
        fault
        for fault in ("startup_failure", "hung_step", "sigint")
        for _ in range(args.failure_repeats)
    ]
    rng.shuffle(failures)
    batches.extend(batch("failure", "full", 1, 1, fault=fault) for fault in failures)
    return {
        "schema_version": 1,
        "argv": sys.argv,
        "seed": args.seed,
        "study": study,
        "study_sha256": digest(args.study),
        "harness_sha256": digest(HERE),
        "context": str(args.context),
        "context_source_digest": runner.source_digest(args.context),
        "context_file_hashes": context_hashes,
        "resources": BUDGET,
        "repeats_per_arm_per_concurrency": args.repeats,
        "concurrency": args.concurrency,
        "cold_builds": args.cold_builds,
        "failure_repeats": args.failure_repeats,
        "healthy_expectations": {
            "study": "9 runtime PASS, 7 SKIP, overall WARN",
            "verified_baseline": "9 runtime PASS, 4 emitted SKIP, static PASS, overall WARN",
            "discrepancy": "Three capability-inapplicable runtime checks are not selected; preserve absence rather than manufacturing SKIP",
        },
        "batches": batches,
    }


def verify_inputs(args):
    if platform.python_version() != "3.12.13":
        raise RuntimeError("Run with tests/validation_runtime/.venv Python 3.12.13")
    uv = subprocess.run(
        ["uv", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if uv.split()[:2] != ["uv", "0.9.3"]:
        raise RuntimeError("Expected uv 0.9.3, got " + uv)
    manifest = (
        runner.default_parser_registry()
        .parser_for(runner.detect_signature(args.context))
        .parse(args.context)
    )
    if {name: getattr(manifest.resources, name) for name in BUDGET} != BUDGET:
        raise RuntimeError(
            "Context resources differ from frozen CPU/memory/disk/episode budget"
        )
    hashes = file_hashes(args.context)
    if args.context_manifest:
        recorded = json.loads(args.context_manifest.read_text())["context_file_hashes"]
        if hashes != recorded:
            raise RuntimeError("Staged context does not match its retained manifest")
    (wheel,) = (args.lab / "wheel").glob("openenv-*.whl")
    if hashes.get("wheelhouse/" + wheel.name) != digest(wheel):
        raise RuntimeError("Context OpenEnv wheel differs from exact baseline wheel")
    package = Path(runner.__file__).resolve().parents[1]
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.startswith("openenv/") and not member.endswith("/"):
                installed = package.parent / member
                if not installed.is_file() or installed.read_bytes() != archive.read(
                    member
                ):
                    raise RuntimeError(
                        "Installed OpenEnv differs from baseline wheel: " + member
                    )
    return hashes, {
        "python": platform.python_version(),
        "uv": uv,
        "openenv": importlib.metadata.version("openenv"),
        "installed_package": str(package),
        "wheel_sha256": digest(wheel),
        "wheel": str(wheel),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "baseline_report_sha256": digest(args.lab / "cli/good/report.json"),
        "baseline_manifest_sha256": digest(args.lab / "run-manifest.json"),
        "acquisition_scope": "Base-image/dependency acquisition occurred in baseline lab, outside these build timings",
        "acquisition_log_hashes": {
            name: digest(args.lab / "logs" / name)
            for name in ("base-image-pull.log", "dependency-acquisition.log")
        },
    }


def preflight(args):
    pins = json.loads((ROOT / "tests/validation_runtime/toolchain.json").read_text())
    code, stdout, stderr = docker._command(
        ["docker", "info", "--format", "{{json .}}"], 30
    )
    if code:
        raise RuntimeError("Docker preflight failed: " + stderr[-1024:])
    details = json.loads(stdout)
    if (
        details["Architecture"] not in {"arm64", "aarch64"}
        or details["OSType"] != "linux"
    ):
        raise RuntimeError("This frozen experiment targets a Linux arm64 Docker engine")
    images = {}
    for image in (pins["base_image"], docker.HELPER_IMAGE):
        code, stdout, stderr = docker._command(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image], 30
        )
        if code:
            raise RuntimeError(
                "Required pre-acquired image missing; stage outside timing: " + image
            )
        images[image] = stdout.strip()
    return {
        "docker": {
            key: details.get(key)
            for key in (
                "ServerVersion",
                "Architecture",
                "OperatingSystem",
                "OSType",
                "KernelVersion",
                "Driver",
                "CgroupDriver",
                "CgroupVersion",
            )
        },
        "preacquired_images": images,
        "local_free_bytes": shutil.disk_usage(args.output).free,
        "acquisition": "No pulls/downloads performed by preflight; lab acquisition excluded",
    }


def cold_build(row, args):
    directory = args.output / "runs" / row["id"]
    directory.mkdir(parents=True)
    trial = Trial(row, directory)
    token = CURRENT.set(trial)
    try:
        manifest = (
            runner.default_parser_registry()
            .parser_for(runner.detect_signature(args.context))
            .parse(args.context)
        )
        provider_for(trial).build(args.context, manifest.execution)
    except BaseException as exc:
        trial.error("cold_build", exc)
    finally:
        CURRENT.reset(token)
    result = {
        **row,
        "wall_s": time.monotonic() - trial.started,
        "images": trial.images,
        "phases": trial.events,
        "commands": trial.commands,
        "errors": trial.errors,
        "scope": "--no-cache offline install/build; warm base image and retained dependencies, not a fully cold host",
    }
    save(directory / "metrics.json", result)
    checksums(directory)
    return result


def failure_subprocess(row, args):
    """Signal only this fresh Python root process, after its logged second step."""
    directory = args.output / "failures" / row["id"]
    directory.mkdir(parents=True)
    request = directory / "request.json"
    save(request, row)
    command = [
        sys.executable,
        str(HERE),
        "--context",
        str(args.context),
        "--output",
        str(args.output),
        "--study",
        str(args.study),
        "--lab",
        str(args.lab),
        "--worker-case",
        str(request),
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    milestone = None
    error = None
    started = time.monotonic()
    with (
        (directory / "stdout.log").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=environment,
        )
        try:
            if row["fault"] in {"sigint", "hung_step"}:
                deadline = time.monotonic() + 120
                while process.poll() is None and time.monotonic() < deadline:
                    for container in owned_ids(row["run_id"]):
                        code, out, err = docker._command(
                            ["docker", "logs", "--tail", "30", container], 5, 8192
                        )
                        if code == 0 and "OPENENV_VALIDATION_STEP_BLOCKED" in out + err:
                            milestone = time.monotonic()
                            save(
                                directory / "milestone.json",
                                {
                                    "container": container,
                                    "milestone": "second_step_received",
                                    "at_monotonic": milestone,
                                },
                            )
                            break
                    if milestone is not None:
                        break
                    time.sleep(0.05)
                if milestone is None:
                    raise RuntimeError("No second-step milestone before exit/deadline")
                if row["fault"] == "sigint":
                    process.send_signal(signal.SIGINT)
            process.wait(timeout=15 if milestone is not None else 120)
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            if process.poll() is None:
                # Production _command must reap its separate-session Docker client.
                process.send_signal(signal.SIGINT)
                process.wait(timeout=660)
    elapsed = time.monotonic() - milestone if milestone is not None else None
    artifact = args.output / "runs" / row["id"] / "metrics.json"
    result = (
        json.loads(artifact.read_text())
        if artifact.exists()
        else {
            **row,
            "errors": [{"phase": "worker", "message": "worker produced no metrics"}],
            "verdict": None,
            "checks": {},
        }
    )
    # Independent parent check also covers a worker killed before its finally block.
    recovery = Trial(row, directory)
    token = CURRENT.set(recovery)
    try:
        result["parent_cleanup"] = teardown(recovery)
    finally:
        CURRENT.reset(token)
    cleanup_elapsed = time.monotonic() - milestone if milestone is not None else None
    result["parent_cleanup_commands"] = recovery.commands
    result["parent_cleanup_phases"] = recovery.events
    result["errors"].extend(recovery.errors)
    if error:
        result["errors"].append({"phase": "failure_supervisor", **error})
    result.update(
        worker_exit_code=process.returncode,
        subprocess_command=command,
        subprocess_wall_s=time.monotonic() - started,
        milestone_observed=milestone is not None,
        seconds_milestone_to_process_exit=elapsed,
        seconds_milestone_to_cleanup_verified=cleanup_elapsed,
        cleanup_deadline_s=15 if milestone is not None else 120,
    )
    if row["fault"] in {"sigint", "hung_step"}:
        preserved = result.get("preserved_operations") == [
            "reset",
            "state",
            "step",
            "state",
        ]
        result["preserved_prefix_matches"] = preserved
        trace_path = args.output / "runs" / row["id"] / "bundle/collector-trace.json"
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
        preserved = preserved and (
            trace[0]["response_json"]["data"]["observation"]["counter"] == 0
            and trace[2]["response_json"]["data"]["observation"]["counter"] == 1
            and trace[3]["response_json"]["data"]["step_count"] == 1
        )
        result["preserved_prefix_matches"] = preserved
        completion = result.get("collector_completion") or {}
        expected_reason = (
            "KeyboardInterrupt" if row["fault"] == "sigint" else "TimeoutError"
        )
        reason = completion.get("failure_reason") or ""
        result["expected_failure_match"] = (
            preserved
            and completion.get("complete") is False
            and completion.get("failure_phase") == "step"
            and expected_reason.lower() in reason.lower()
            and result["checks"].get("runtime.startup")
            == ("error" if row["fault"] == "sigint" else "fail")
            and result["verdict"] == "fail"
            and elapsed is not None
            and elapsed < 15
            and cleanup_elapsed is not None
            and cleanup_elapsed < 15
            and result["parent_cleanup"]["assert_empty"]
        )
        if not result["expected_failure_match"]:
            result["errors"].append(
                {
                    "phase": "failure_evidence",
                    "message": "prefix/milestone-to-exit deadline mismatch",
                }
            )
    else:
        result["expected_failure_match"] = (
            result["checks"].get("runtime.startup") == "fail"
            and result["verdict"] == "fail"
            and result["parent_cleanup"]["assert_empty"]
        )
    save(directory / "result.json", result)
    checksums(directory)
    return result


def self_check():
    assert quantiles(list(range(1, 21))) == {"n": 20, "p50": 10, "p95": 19}
    assert quantiles([3]) == {"n": 1, "p50": 3, "p95": 3}
    assert quantiles([]) == {"n": 0, "p50": None, "p95": None}
    trial = Trial({"id": "check"}, Path("."))
    with trial.phase("resources"):
        with trial.phase("containment"):
            sum(range(100))
    assert (
        sum(event["exclusive_wall_s"] for event in trial.events)
        <= trial.events[-1]["wall_s"] + 1e-8
    )

    def worker(index):
        token = CURRENT.set(index)
        try:
            time.sleep(0.001)
            return CURRENT.get()
        finally:
            CURRENT.reset(token)

    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(worker, range(32))) == list(range(32))
    assert CURRENT.get() is None
    print("Self-check passed (no Docker): quantiles, nested timing, context isolation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=Path, default=LAB_ROOT / "performance-context"
    )
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=LAB_ROOT / "performance-context-manifest.json",
    )
    parser.add_argument(
        "--lab", type=Path, default=LAB_ROOT / "docker-baseline/l2-d2aada5dfcff"
    )
    parser.add_argument("--study", type=Path, default=LAB_ROOT / "study.json")
    parser.add_argument("--output", type=Path, default=LAB_ROOT / "exp3")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", choices=(1, 4, 8), default=[1, 4, 8]
    )
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--cold-builds", type=int, choices=(0, 1, 3), default=0)
    parser.add_argument("--failure-repeats", type=int, choices=(0, 3), default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.repeats < 1 or len(set(args.concurrency)) != len(args.concurrency):
        parser.error("positive repeats and distinct concurrency levels required")
    args.context, args.output = args.context.resolve(), args.output.resolve()
    if args.worker_case:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        install_instrumentation()
        row = json.loads(args.worker_case.read_text())
        run_trial(row, args.context, args.output)
        return
    hashes, toolchain = verify_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = plan(args, hashes)
    frozen["toolchain"] = toolchain
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        # argv differs when the exact frozen dry-run is executed.
        if {k: v for k, v in prior.items() if k != "argv"} != {
            k: v for k, v in frozen.items() if k != "argv"
        }:
            raise RuntimeError("Existing plan differs; choose a fresh output directory")
    else:
        save(plan_path, frozen)
    if args.plan_only:
        checksums(args.output)
        print(f"Frozen plan (no Docker executed): {plan_path}")
        return
    if (args.output / "execution.json").exists():
        raise RuntimeError(
            "Execution already started here; choose a fresh output directory"
        )
    save(
        args.output / "execution.json",
        {
            "argv": sys.argv,
            "plan_sha256": digest(plan_path),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    save(args.output / "preflight.json", preflight(args))
    install_instrumentation()
    rows, batches, cold, failures = [], [], [], []
    abort = None
    try:
        for index, batch in enumerate(frozen["batches"]):
            print(
                f"Batch {index + 1}/{len(frozen['batches'])}: {batch['kind']} {batch['arm']} c{batch['concurrency']} n{len(batch['runs'])}",
                flush=True,
            )
            started = time.monotonic()
            if batch["kind"] == "cache_bypassing_build":
                cold.extend(cold_build(row, args) for row in batch["runs"])
                continue
            if batch["kind"] == "failure":
                failures.extend(failure_subprocess(row, args) for row in batch["runs"])
                continue
            with ThreadPoolExecutor(max_workers=batch["concurrency"]) as pool:
                results = list(
                    pool.map(
                        lambda row: run_trial(row, args.context, args.output),
                        batch["runs"],
                    )
                )
            elapsed = time.monotonic() - started
            rows.extend(results)
            batches.append(
                {
                    "index": index,
                    "kind": batch["kind"],
                    "arm": batch["arm"],
                    "concurrency": batch["concurrency"],
                    "n": len(results),
                    "measured_n": sum(result["kind"] == "warm" for result in results),
                    "wall_s": elapsed,
                    "throughput_runs_s": len(results) / elapsed,
                    "error_runs": sum(bool(result["errors"]) for result in results),
                    "verdicts": dict(
                        Counter(str(result["verdict"]) for result in results)
                    ),
                }
            )
            save(
                args.output / "report.json",
                {
                    "summary": summarize(rows, batches, failures),
                    "runs": rows,
                    "cold_builds": cold,
                    "failures": failures,
                },
            )
    except BaseException as exc:
        abort = {"type": type(exc).__name__, "message": str(exc)[:4096]}
        raise
    finally:
        # Executor shutdown waits for workers after interruption; retain their
        # durable metrics even when list(pool.map(...)) did not return.
        rows = []
        for metric_path in sorted((args.output / "runs").glob("*/metrics.json")):
            retained = json.loads(metric_path.read_text())
            if retained["kind"] in {"warmup", "warm", "warm_padding"}:
                rows.append(retained)
        # Context changed underneath a run is an error, never a new benchmark input.
        unchanged = file_hashes(args.context) == frozen["context_file_hashes"]
        images = sorted({image for row in rows for image in row["image_refs"]})
        save(
            args.output / "report.json",
            {
                "summary": summarize(rows, batches, failures),
                "runs": rows,
                "cold_builds": cold,
                "failures": failures,
                "context_unchanged": unchanged,
                "primary_image_ids": images,
                "same_primary_image": len(images) == 1,
                "execution_abort": abort,
                "missing_planned_run_ids": sorted(
                    {row["id"] for batch in frozen["batches"] for row in batch["runs"]}
                    - {row["id"] for row in [*rows, *cold, *failures]}
                ),
                "image_cleanup": "No images deleted; retained immutable IDs for root-owned space decisions",
            },
        )
        checksums(args.output)
    print(f"Report: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
