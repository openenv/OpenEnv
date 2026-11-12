"""Paired real-source defect experiment using production collection and graders.

No compute submission, package installation, or source download occurs here.
Inputs are pinned public snapshots prepared outside the checkout. The expected
matrix and randomized schedule are written before any measured server starts.
"""

import argparse
import collections
import dataclasses
import difflib
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.basic import (
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.report import CheckResult
from openenv.validation.runtime.collector import collect_runtime_evidence
from openenv.validation.runtime.contracts import IsolationEvidence, RuntimePlan
from openenv.validation.runtime.scheduler import execute_graders


CASES = {
    "malformed_observation": "runtime.observation_schema",
    "invalid_reward": "runtime.reward_well_formed",
    "reset_state_leak": "runtime.episode_isolation",
    "hung_step": "runtime.startup",
}
GRADERS = [
    RewardWellFormedGrader(),
    ObservationSchemaGrader(),
    StateContractGrader(),
    EpisodeIsolationGrader(),
]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def hashes(root):
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def server_environment(source, config, case, witness):
    # Explicit environment: no credentials, Hub tokens or proxy variables forwarded.
    return {
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO / "src"), str(source.parent), str(HERE / "exp1")]
            + [str(source / path) for path in config.get("import_paths", [])]
        ),
        "ENABLE_WEB_INTERFACE": "false",
        "MAX_CONCURRENT_ENVS": "4",
        "BENCH_APP": config["app"],
        "BENCH_CASE": case,
        "BENCH_WITNESS": str(witness),
        "BENCH_TRIGGER": json.dumps(config["action"]),
        "BENCH_OBSERVATION_KEY": config["observation_key"],
        **config.get("environment", {}),
    }


def patch_leak(config, root, destination):
    shutil.copytree(root, destination, ignore=shutil.ignore_patterns("__pycache__"))
    path = destination / config["leak_file"]
    original = path.read_text()
    before, after = config["leak_replace"]
    if original.count(before) != 1:
        raise ValueError("server leak patch does not match the pinned source once")
    modified = original.replace(before, after)
    path.write_text(modified)
    return "".join(
        difflib.unified_diff(
            original.splitlines(True),
            modified.splitlines(True),
            fromfile="original/" + config["leak_file"],
            tofile="mutant/" + config["leak_file"],
        )
    )


def run_trial(config, case, directory, timeout):
    directory.mkdir(parents=True)
    source = Path(config["root"])
    if case == "reset_state_leak":
        source = directory / "source" / config["package"]
        patch = patch_leak(config, Path(config["root"]), source)
        (directory / "server.patch").write_text(patch)
    witness = directory / "witness.jsonl"
    env = server_environment(source, config, case, witness)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--app-dir",
        str(HERE / "exp1"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    save(
        directory / "launch.json",
        {
            "command": command,
            "environment": env,
            "source_hashes": hashes(source),
            "mutation_scope": "server reset assignment"
            if case == "reset_state_leak"
            else "wire wrapper"
            if case != "healthy"
            else "none",
        },
    )
    process = None
    started = time.monotonic()
    with (directory / "server.log").open("w") as log:
        try:
            process = subprocess.Popen(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("real source server exited during startup")
                try:
                    with urllib.request.urlopen(
                        base + "/health", timeout=0.5
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("real source server readiness deadline exceeded")
            plan = RuntimePlan.model_validate(config["plan"])
            reference = collect_runtime_evidence(base, plan, episode_timeout_s=timeout)
            sequential = fresh = None
            if not reference.failure_reason:
                sequential = collect_runtime_evidence(
                    base, plan, episode_timeout_s=timeout, episodes=2
                )
                fresh = collect_runtime_evidence(base, plan, episode_timeout_s=timeout)
            evidence = {
                "reference": dataclasses.asdict(reference),
                "sequential": dataclasses.asdict(sequential) if sequential else None,
                "fresh": dataclasses.asdict(fresh) if fresh else None,
            }
            save(directory / "raw-evidence.json", evidence)
            startup = CheckResult(
                check_id="runtime.startup",
                status="fail" if reference.failure_reason else "pass",
                evidence=[
                    reference.failure_reason
                    or "schema and protocol collection completed"
                ],
                duration_s=0,
            )
            subject = Subject(
                root=source,
                manifest=NormalizedManifest.model_validate(config["manifest"]),
                image_ref=None,
                running=None,
                outputs_dir=directory,
                runtime_evidence=reference,
                isolation_evidence=IsolationEvidence(sequential=sequential, fresh=fresh)
                if sequential and fresh
                else None,
            )
            checks = [startup, *execute_graders(GRADERS, subject, prior=[startup])]
            rows = [c.model_dump(mode="json") for c in checks]
            events = [json.loads(line) for line in witness.read_text().splitlines()]
            activation = confirm_activation(case, config, events, evidence)
            status = {c["check_id"]: c["status"] for c in rows}
            result = {
                "environment": config["name"],
                "case": case,
                "checks": rows,
                "status": status,
                "activation": activation,
                "elapsed_s": time.monotonic() - started,
                "source_unchanged": hashes(Path(config["root"]))
                == config["source_hashes"],
            }
        except Exception as error:
            result = {
                "environment": config["name"],
                "case": case,
                "harness_error": repr(error),
                "status": {},
                "activation": {"confirmed": False},
                "elapsed_s": time.monotonic() - started,
            }
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)
    result["server_returncode"] = process.returncode if process else None
    result["artifact_directory"] = str(directory)
    save(directory / "result.json", result)
    return result


def confirm_activation(case, config, events, evidence):
    """Independent stimulus witnesses; never read grader results here."""
    triggered = [e for e in events if e["event"] == "fault_activated"]
    upstream = [e for e in events if e["event"] == "upstream_response"]
    if case in {"malformed_observation", "invalid_reward", "hung_step"}:
        confirmed = bool(triggered) and all(
            e["action"] == config["action"] for e in triggered
        )
        wire = [
            json.loads(e["response_json"])["data"]
            for e in evidence["reference"]["exchanges"]
            if e["operation"] == "step"
        ]
        if case == "malformed_observation":
            confirmed = confirmed and any(
                isinstance(e["observation"].get(config["observation_key"]), dict)
                for e in wire
            )
        elif case == "invalid_reward":
            confirmed = confirmed and any(
                e["reward"] == "exp1-invalid-reward" for e in wire
            )
        if case == "hung_step":
            confirmed = (
                confirmed
                and all(e["upstream_completed"] for e in triggered)
                and evidence["reference"]["failure_phase"] == "step"
            )
        return {
            "confirmed": confirmed,
            "trigger_count": len(triggered),
            "kind": "independent ASGI upstream/downstream witness",
            "event_file": "witness.jsonl",
        }
    if case == "reset_state_leak":
        later = [
            e
            for e in upstream
            if e["operation"] == "step" and e["session"]["resets"] > 1
        ]
        if config.get("leak_state_key"):
            resets = [
                e
                for e in upstream
                if e["operation"] == "state"
                and e["session"]["resets"] > 1
                and e["response"]["data"].get("step_count") == 0
            ]
            confirmed = any(
                e["response"]["data"].get(config["leak_state_key"], 0) > 0
                for e in resets
            )
        else:
            confirmed = config["leak_witness"] in json.dumps(later)
        return {
            "confirmed": confirmed,
            "trigger_count": len(later) if confirmed else 0,
            "kind": "real server patched reset; independently inspect second-episode response",
            "observed_later_episode_steps": len(later),
            "event_file": "witness.jsonl",
        }
    return {
        "confirmed": not triggered and any(e["operation"] == "step" for e in upstream),
        "kind": "unmodified-source control",
        "trigger_count": 0,
    }


def inventory():
    rows = []
    for distribution in importlib.metadata.distributions():
        metadata = distribution.read_text("METADATA") or ""
        record = distribution.read_text("RECORD") or ""
        rows.append(
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
                "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
            }
        )
    return sorted(rows, key=lambda x: x["name"].lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3)
    args = parser.parse_args()
    if args.replicates < 3:
        parser.error("at least three paired replicates are required")
    configs = json.loads(args.config.read_text())
    for config in configs:
        NormalizedManifest.model_validate(config["manifest"])
        RuntimePlan.model_validate(config["plan"])
    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        parser.error(
            "output directory already exists; use a new path to preserve evidence"
        )
    for path in [Path(__file__), HERE / "exp1" / "server.py"]:
        shutil.copyfile(path, args.output / ("executed-" + path.name))
    rng = random.Random(20261109)
    schedule = []
    for config in configs:
        config["source_hashes"] = hashes(Path(config["root"]))
        for replicate in range(args.replicates):
            for fault in CASES:
                pair = ["healthy", fault]
                rng.shuffle(pair)
                seed = rng.randrange(2**32)
                for case in pair:
                    trial = {
                        **config,
                        "plan": {
                            **config["plan"],
                            "reset": {**config["plan"]["reset"], "seed": seed},
                        },
                    }
                    schedule.append(
                        {
                            "replicate": replicate,
                            "fault": fault,
                            "case": case,
                            "config": trial,
                        }
                    )
    # Freeze exact expected cases, sampled seeds and within-pair order before launch.
    frozen = {
        "schema": "exp1-v1",
        "random_seed": 20261109,
        "replicates": args.replicates,
        "collection": {"episode_timeout_s": args.timeout},
        "invocation": [sys.executable, *sys.argv],
        "expected": {
            "healthy": {
                "scope": "startup, observation schema, finite declared reward, state contract",
                "expected_status": "see control_expectations_by_environment",
                "isolation": "pass when fresh traces deterministic; skip is an unproven boundary, never pass",
            },
            **{
                case: {
                    "target_check": check,
                    "expected_status": "fail",
                    "requires_independent_activation": True,
                }
                for case, check in CASES.items()
            },
        },
        "schedule": schedule,
        "control_expectations_by_environment": {
            c["name"]: c.get(
                "control_expectations",
                {
                    "runtime.startup": "pass",
                    "runtime.observation_schema": "pass",
                    "runtime.reward_well_formed": "pass",
                    "runtime.state_contract": "pass",
                },
            )
            for c in configs
        },
        "limitations": [
            "source bootstrap; no published-image qualification",
            "original CLI refusals retained separately",
            "no grader/core modifications",
            "wire faults explicitly distinct from server reset mutation",
            "timing fields remain unchanged and may make isolation SKIP",
        ],
        "python": sys.version,
        "dependency_inventory": inventory(),
        "production_validation_hashes": hashes(REPO / "src" / "openenv" / "validation"),
        "harness_hashes": {
            str(p): digest(p) for p in [Path(__file__), HERE / "exp1" / "server.py"]
        },
    }
    save(args.output / "frozen-expectations.json", frozen)
    expectation_hash = digest(args.output / "frozen-expectations.json")
    results = []
    for index, trial in enumerate(schedule):
        directory = (
            args.output
            / f"{index:03}-{trial['config']['name']}-r{trial['replicate']}-{trial['fault']}-{trial['case']}"
        )
        result = run_trial(trial["config"], trial["case"], directory, args.timeout)
        result.update(
            replicate=trial["replicate"],
            paired_fault=trial["fault"],
            expectation_sha256=expectation_hash,
        )
        target = CASES[trial["fault"]]
        result["target_check"] = target
        result["target_status"] = result["status"].get(target, "error")
        result["false_pass"] = (
            trial["case"] != "healthy"
            and result["activation"]["confirmed"]
            and result["target_status"] == "pass"
        )
        expected_control = frozen["control_expectations_by_environment"][
            result["environment"]
        ]
        result["false_fail"] = trial["case"] == "healthy" and any(
            expected == "pass" and result["status"].get(name) == "fail"
            for name, expected in expected_control.items()
        )
        result["control_matches_expected"] = (
            all(
                result["status"].get(name) == expected
                for name, expected in expected_control.items()
            )
            if trial["case"] == "healthy"
            else None
        )
        results.append(result)
        save(args.output / "results.json", results)
        print(
            json.dumps(
                {
                    "trial": index,
                    "environment": result["environment"],
                    "case": result["case"],
                    "activation": result["activation"]["confirmed"],
                    "target_status": result["target_status"],
                    "harness_error": result.get("harness_error"),
                }
            ),
            flush=True,
        )
    report = {
        "experiment": 1,
        "scope": "real-source defect detection with production collector/graders",
        "expectation_sha256": expectation_hash,
        "trials": results,
        "summary": {
            "total_trials": len(results),
            "fault_trials": sum(r["case"] != "healthy" for r in results),
            "activated_fault_trials": sum(
                r["case"] != "healthy" and r["activation"]["confirmed"] for r in results
            ),
            "false_pass": sum(r["false_pass"] for r in results),
            "false_fail_scoped_control": sum(r["false_fail"] for r in results),
            "target_status_faults": dict(
                collections.Counter(
                    r["target_status"] for r in results if r["case"] != "healthy"
                )
            ),
            "all_check_statuses": dict(
                collections.Counter(s for r in results for s in r["status"].values())
            ),
        },
    }
    save(args.output / "report.json", report)
    save(args.output / "artifact-hashes.json", hashes(args.output))
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
