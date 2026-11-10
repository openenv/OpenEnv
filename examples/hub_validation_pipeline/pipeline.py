"""Discover pinned tasks, validate them, then evaluate only an admitted local corpus."""

import argparse
import asyncio
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import tomli_w
from examples.hub_validation_pipeline import sources
from openenv.validation.parsers.harbor_toml import (
    _unique_object,
    canary_scripts,
    read_harbor_contract,
)
from openenv.validation.policy import load_policy
from openenv.validation.report import ValidationReportV2
from openenv.validation.runner import _DIGEST_EXCLUDED_DIRS, source_digest
from openenv.validation.runtime.json_values import dumps, loads
from openenv.validation.types import Lane

PILOT = "harbor-semantic-v1"
PILOT_CHECKS = {
    "static.manifest",
    "runtime.startup",
    "runtime.reward_well_formed",
    "runtime.episode_determinism",
    "semantic.oracle_max",
    "semantic.floor_gap",
}
REPO = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(value) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    path.write_text("".join(dumps(row) + "\n" for row in rows))


def copy_subject(source, destination, expected_digest):
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*_DIGEST_EXCLUDED_DIRS),
    )
    if source_digest(destination) != expected_digest:
        raise ValueError("source changed while copying the task")


def validation_environment():
    """Give the validation worker Docker access without inference or Hub secrets."""
    names = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
        "SSH_AUTH_SOCK",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    env = {key: value for key, value in os.environ.items() if key in names}
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO / "envs")))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def validate_task(subject, report_path, profile, timeout_s):
    if profile not in {PILOT, "full"}:
        raise ValueError("unsupported admission profile")
    command = [
        sys.executable,
        "-m",
        "openenv.cli",
        "validate",
        str(subject),
        "--level",
        "semantic",
        "--policy",
        "v2",
        "--json",
        "--output",
        str(report_path),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=validation_environment(),
        cwd=REPO,
    )
    try:
        _, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # SIGTERM reaches validate's cleanup handler before a final hard stop.
        process.terminate()
        try:
            process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise TimeoutError("validation deadline exceeded; task excluded") from None
    if not report_path.is_file():
        raise RuntimeError(
            f"validator exited {process.returncode} without a report: {stderr[-500:]}"
        )
    return process.returncode, loads(report_path.read_text())


def verified_bundle(directory):
    """Verify every retained artifact before using its admission evidence."""
    entries = {}
    for row in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = row.split("  ", 1)
        relative = Path(name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or name in entries
        ):
            raise ValueError("invalid artifact checksum inventory")
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("artifact path escapes evidence directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
        entries[name] = digest
    if not {"report.json", "harbor-evidence.json"}.issubset(entries):
        raise ValueError(
            "native validation evidence is missing from checksum inventory"
        )
    return entries


def reward_fraction(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, Decimal))
        or not Decimal(value).is_finite()
    ):
        raise ValueError("native reward must be finite and numeric")
    return Fraction(value)


def trial_rewards(trial, artifact_dir, checksums, key, low, high):
    """Bind every raw output, including an optional nonzero oracle exit marker."""
    raw = trial["raw_rewards"]
    allowed = {"reward.json", "reward.txt", "oracle-exit-code.txt"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("unexpected native raw output inventory")
    prefix = f"harbor-trials/{trial['role']}"
    paths = {f"{prefix}/{name}" for name in raw}
    actual = {
        path.relative_to(artifact_dir).as_posix()
        for path in (artifact_dir / prefix).iterdir()
        if path.name in allowed
    }
    if paths != actual:
        raise ValueError("native raw output inventory is incomplete")
    outputs = {}
    for name, reference in raw.items():
        if reference.get("path") != f"{prefix}/{name}" or checksums.get(
            reference["path"]
        ) != reference.get("sha256"):
            raise ValueError(
                "raw native output is not in the verified artifact inventory"
            )
        outputs[name] = (artifact_dir / reference["path"]).read_text()
    if "reward.json" in outputs:
        values = loads(outputs["reward.json"], object_pairs_hook=_unique_object)
    elif "reward.txt" in outputs:
        try:
            values = {"reward": Decimal(outputs["reward.txt"].strip())}
        except ArithmeticError as error:
            raise ValueError("invalid native reward text") from error
    else:
        raise ValueError("native trial has no raw rewards")
    if not isinstance(values, dict) or not values or key not in values:
        raise ValueError("native reward map lacks the declared key")
    rewards = {name: reward_fraction(value) for name, value in values.items()}
    if any(not low <= value <= high for value in rewards.values()):
        raise ValueError("native reward is outside the declared range")
    observed = {
        name: reward_fraction(value) for name, value in trial["rewards"].items()
    }
    if rewards != observed:
        raise ValueError("raw native rewards differ from the recorded reward map")
    if trial["role"] != "nop":
        code = trial.get("oracle_exit_code")
        if (
            trial.get("oracle_exit_observed_at_agent_end") is not True
            or isinstance(code, bool)
            or not isinstance(code, int)
            or code != int(outputs.get("oracle-exit-code.txt", "0"))
        ):
            raise ValueError("native oracle exit evidence is inconsistent")
    return rewards


def admission(report, exit_code, subject, artifact_dir, profile, original_digest):
    """Return a task/image-bound admission envelope, or reject the task."""
    if profile not in {PILOT, "full"}:
        raise ValueError("unsupported admission profile")
    ValidationReportV2.model_validate(report)
    if exit_code != 0 or report["verdict"] not in {"pass", "warn"}:
        raise ValueError("validation failed")
    if (
        report["signature"] != "task.toml"
        or report["policy_version"] != "v2"
        or report["lane"] != "local"
    ):
        raise ValueError("unexpected validation signature or policy")
    effective_digest = source_digest(subject)
    if report["source_digest"] != effective_digest:
        raise ValueError("task changed after validation")
    parsed_contract = read_harbor_contract(subject)
    contract = loads(
        (subject / "validation/openenv.json").read_text(),
        object_pairs_hook=_unique_object,
    )
    canaries = (
        canary_scripts(subject, parsed_contract.canaries)
        if parsed_contract.canaries is not None
        else []
    )
    results = {row["check_id"]: row for row in report["results"]}
    if len(results) != len(report["results"]):
        raise ValueError("duplicate validation check IDs")
    if any(row["status"] in {"fail", "error"} for row in results.values()):
        raise ValueError("a validation check failed or errored")
    known_checks = set(load_policy("v2").entries_for_lane(Lane.LOCAL))
    if set(results) - known_checks:
        raise ValueError("unknown validation check ID")
    required = (
        PILOT_CHECKS | ({"semantic.canary_floor"} if canaries else set())
        if profile == PILOT
        else set(load_policy("v2").entries_for_lane(Lane.LOCAL))
    )
    if not required.issubset(results) or any(
        results[key]["status"] != "pass" for key in required
    ):
        raise ValueError("required validation coverage is incomplete")
    if not {1, 2, 3}.issubset(report["levels_run"]):
        raise ValueError("semantic validation was not executed")
    if profile != PILOT and (
        report["verdict"] != "pass" or 3 not in report["levels_run"]
    ):
        raise ValueError("full validation is incomplete")
    checksums = verified_bundle(artifact_dir)
    if dumps(loads((artifact_dir / "report.json").read_text())) != dumps(report):
        raise ValueError("bundled report differs from the validator result")
    native = loads((artifact_dir / "harbor-evidence.json").read_text())
    if (
        native.get("schema_version") != "2"
        or native["profile"] != PILOT
        or native["task_effective_digest"] != effective_digest
        or native["original_source_digest"] != original_digest
        or contract.get("original_source_digest") != original_digest
        or native["reward_key"] != contract["reward_key"]
        or report["manifest"]["execution"]["profile"] != PILOT
        or report["manifest"]["capabilities"]["canaries"] != parsed_contract.canaries
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", native.get("image_ref") or "")
        or not re.fullmatch(
            r"openenv-validation:[a-f0-9]{32}", native.get("retained_image_tag") or ""
        )
        or native["cleanup"]["verified"] is not True
        or "failure" in native
    ):
        raise ValueError(
            "native oracle/floor, identity or cleanup evidence is incomplete"
        )
    trials = native["trials"]
    roles = ["nop", "oracle", "oracle-replay"] + [
        f"canary-{index}" for index in range(1, len(canaries) + 1)
    ]
    if (
        [row["role"] for row in trials] != roles
        or len({row["container_id"] for row in trials}) != len(trials)
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", row["container_id"] or "")
            or row["image_ref"] != native["image_ref"]
            or row["retained_image_tag"] != native["retained_image_tag"]
            or row["unmodified_task_digest"] != report["source_digest"]
            or row["configuration"] != trials[0]["configuration"]
            or row["cleanup"]["verified"] is not True
            or "failure" in row
            or "reward_error" in row
            for row in trials
        )
    ):
        raise ValueError(
            "native trials do not establish fresh identical-image execution"
        )
    reward_contract = contract["reward"]
    low, high = map(reward_fraction, reward_contract["range"])
    margin = reward_fraction(reward_contract["floor_margin"])
    tolerance = reward_fraction(reward_contract["oracle_tolerance"])
    rewards = []
    for index, trial in enumerate(trials):
        overlay = (
            {}
            if index == 0
            else {
                "environment.docker_image": native["retained_image_tag"],
            }
        )
        if index >= 3:
            canary = canaries[index - 3]
            overlay["solution/solve.sh"] = {
                "canary": canary,
                "sha256": hashlib.sha256((subject / canary).read_bytes()).hexdigest(),
            }
        if trial["task_overlay"] != overlay:
            raise ValueError(
                "native task overlay differs from the declared source bytes"
            )
        rewards.append(
            trial_rewards(
                trial, artifact_dir, checksums, native["reward_key"], low, high
            )
        )
    if (
        rewards[1] != rewards[2]
        or trials[1]["oracle_exit_code"] != trials[2]["oracle_exit_code"]
        or trials[1]["oracle_exit_code"] != 0
    ):
        raise ValueError("raw native oracle replay or exit code is inconsistent")
    floor, oracle = (row[native["reward_key"]] for row in rewards[:2])
    if (
        floor != reward_fraction(native.get("floor_reward"))
        or oracle != reward_fraction(native.get("oracle_reward"))
        or oracle < high - tolerance
        or oracle - floor < margin
        or high - floor < margin
        or any(row[native["reward_key"]] > floor for row in rewards[3:])
    ):
        raise ValueError("raw native rewards do not satisfy the oracle/floor contract")
    return {
        "profile": profile,
        "rfc008_complete": profile == "full",
        "original_source_digest": original_digest,
        "task_effective_digest": effective_digest,
        "image_ref": native["image_ref"],
        "reward_key": native["reward_key"],
        "floor_reward": native["floor_reward"],
        "oracle_reward": native["oracle_reward"],
        "resources": report["manifest"]["resources"],
        "container_configuration": native["trials"][0]["configuration"],
        "report_sha256": checksums["report.json"],
        "evidence_sha256": checksums["harbor-evidence.json"],
        "out_of_profile_skips": sorted(
            key for key, row in results.items() if row["status"] == "skip"
        ),
    }


def materialize(record, destination):
    """Pin the execution image while retaining the validated task and overlay hashes."""
    subject = Path(record["subject"])
    if source_digest(subject) != record["task_effective_digest"]:
        raise ValueError("validated subject changed before materialization")
    copy_subject(subject, destination, record["task_effective_digest"])
    config_path = destination / "task.toml"
    config = tomllib.loads(config_path.read_text())
    config.setdefault("environment", {})["docker_image"] = record["image_ref"]
    for key, value in (
        ("cpus", record["resources"]["cpu"]),
        ("memory_mb", record["resources"]["memory_mb"]),
        ("storage_mb", record["resources"]["disk_mb"]),
    ):
        config["environment"][key] = float(value) if key == "cpus" else value
    config_path.write_text(tomli_w.dumps(config))
    return {
        **record,
        "execution_path": str(destination),
        "execution_digest": source_digest(destination),
        "execution_overlay": {
            "environment": config["environment"],
        },
    }


def docker(*arguments):
    return subprocess.run(
        ["docker", *arguments], check=True, text=True, capture_output=True, timeout=30
    ).stdout.strip()


def model_guard(records, projects):
    """Attach an actual-container image check before the agent can call its model."""
    by_path = {str(Path(row["execution_path"]).resolve()): row for row in records}

    def created(trial):
        from harbor.trial.hooks import TrialEvent

        path = str(trial.config.task.path.resolve())
        if path not in by_path:
            raise ValueError("model runner selected an unapproved task")
        record = by_path[path]
        if source_digest(Path(path)) != record["execution_digest"]:
            raise ValueError("approved task changed before execution")
        project = trial.agent_environment.session_id
        if not re.fullmatch(r"[a-z0-9_-]+", project):
            raise ValueError("unexpected Harbor project identity")
        projects.add(project)

        async def inspect_image(event):
            ids = docker(
                "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"
            ).splitlines()
            if len(ids) != 1:
                raise ValueError("expected one approved task container")
            actual = loads(docker("inspect", ids[0]))[0]
            if actual["Image"] != record["image_ref"]:
                raise ValueError("model task image differs from validated image")
            expected = record["container_configuration"]
            mounts = sorted(
                [
                    {
                        "type": item["Type"],
                        "destination": item["Destination"],
                        "rw": item["RW"],
                    }
                    for item in actual["Mounts"]
                ],
                key=lambda item: item["destination"],
            )
            network = actual["HostConfig"]["NetworkMode"]
            if network == project + "_default":
                network = "compose-default"
            if (
                actual["Config"]["User"] != expected["user"]
                or actual["Config"]["WorkingDir"] != expected["workdir"]
                or {key: actual["HostConfig"].get(key) for key in expected["host"]}
                != expected["host"]
                or mounts != expected["mounts"]
                or network != expected["network_mode"]
            ):
                raise ValueError("model sandbox configuration differs from validation")

        trial.add_hook(TrialEvent.AGENT_START, inspect_image)

    return created


def cleanup_projects(projects):
    """Remove and recheck only resources belonging to this run's model trials."""
    for project in projects:
        selector = f"label=com.docker.compose.project={project}"
        containers = docker("ps", "-aq", "--filter", selector).splitlines()
        owned_volumes = set()
        if containers:
            for container in loads(docker("inspect", *containers)):
                for mount in container["Mounts"]:
                    if mount["Type"] == "volume":
                        name = mount["Name"]
                        if not re.fullmatch(r"[a-f0-9]{64}", name):
                            raise ValueError(
                                "model trial mounted a volume of unproven ownership"
                            )
                        owned_volumes.add(name)
        for kind, listing, removal in (
            ("container", ("ps", "-aq"), ("rm", "-f", "-v")),
            ("network", ("network", "ls", "-q"), ("network", "rm")),
            ("volume", ("volume", "ls", "-q"), ("volume", "rm")),
        ):
            ids = docker(*listing, "--filter", selector).splitlines()
            if ids:
                docker(*removal, *ids)
            if docker(*listing, "--filter", selector):
                raise RuntimeError(f"model {kind} cleanup could not be verified")
        if owned_volumes.intersection(docker("volume", "ls", "-q").splitlines()):
            raise RuntimeError("model anonymous-volume cleanup could not be verified")


def evaluate(
    records, output, *, llm_url, model, harness, expose, timeout_s, step_limit
):
    """Run existing OpenEnv Harbor capture on the frozen approved corpus."""
    from openenv.harbor.runner import run_batch
    from openenv.harbor.tasks import resolve_task_dirs

    if not records:
        return []
    approved = output / "approved"
    expected = {str(Path(row["execution_path"]).resolve()): row for row in records}
    paths = resolve_task_dirs(str(approved))
    if {str(path.resolve()) for path in paths} != set(expected):
        raise ValueError("OpenEnv loader and approved corpus disagree")
    results = []
    # Different tasks can choose different headline reward keys.
    for index, path in enumerate(paths):
        record = expected[str(path.resolve())]
        projects = set()
        cleanup_ok = True
        try:
            batch = asyncio.run(
                run_batch(
                    llm_url=llm_url,
                    model=model,
                    dataset=str(approved),
                    task_indices=[index],
                    harness=harness,
                    sandbox="docker",
                    expose=expose,
                    trials_dir=output / "model-trials",
                    reward_key=record["reward_key"],
                    agent_timeout_sec=timeout_s,
                    agent_step_limit=step_limit,
                    on_trial_created=model_guard(records, projects),
                    keep_sandbox=True,
                    environment_kwargs={
                        "auto_mounting": False,
                        "keep_containers": True,
                    },
                )
            )
            if len(batch) != 1 or Path(batch[0].task_id).resolve() != path.resolve():
                raise ValueError("model result accounting is incomplete")
            results.append(
                {
                    "task_id": record["task_id"],
                    "admission": record,
                    "result": batch[0].model_dump(),
                }
            )
        except Exception as error:
            results.append(
                {"task_id": record["task_id"], "error": str(error), "ok": False}
            )
        finally:
            try:
                cleanup_projects(projects)
            except Exception as error:
                cleanup_ok = False
                results[-1]["cleanup_error"] = str(error)
            write_rows(output / "rollouts.jsonl", results)
        if not cleanup_ok:
            break
    return results


def local_tasks(root, limit):
    root = root.resolve(strict=True)
    manifests = [
        root / path / "task.toml"
        for path in sources.task_roots(
            manifest.relative_to(root).as_posix()
            for manifest in root.rglob("task.toml")
        )
    ]
    return [
        {
            "task_id": f"local:{root.name}@{source_digest(manifest.parent)}:{manifest.parent.relative_to(root)}",
            "path": str(manifest.parent),
            "source": "local",
            "source_id": str(root),
            "revision": source_digest(manifest.parent),
            "task_path": manifest.parent.relative_to(root).as_posix(),
        }
        for manifest in manifests[:limit]
    ]


def run(args):
    if vars(args).get("submit_hf_jobs", False):
        raise ValueError("Use prepared_batch with a reviewed shared reservation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    coding_config = getattr(args, "coding_config", None)
    coding_config = loads(coding_config.read_text()) if coding_config else None
    contract = loads(args.contract.read_text()) if coding_config is None else None
    if contract is not None and contract.get("profile") != PILOT:
        raise ValueError(f"contract must declare {PILOT}")
    candidates, tasks, excluded, accepted = [], [], [], []
    plans = []
    explicit = bool(args.hf_dataset or args.harbor_dataset or args.local_dataset)
    if not explicit:
        for source, discover in (
            ("hf", sources.discover_hf),
            ("harbor", sources.discover_harbor),
        ):
            if args.source in {source, "both"}:
                try:
                    candidates.extend(discover(args.dataset_limit))
                except Exception as error:
                    excluded.append(
                        {"source": source, "stage": "discovery", "reason": str(error)}
                    )
    for spec in args.hf_dataset:
        repo, separator, revision = spec.partition("@")
        if not separator:
            revision = sources.HfApi(token=False).dataset_info(repo).sha
        candidates.append({"source": "hf", "id": repo, "revision": revision})
    for spec in args.harbor_dataset:
        name, separator, revision = spec.partition("@")
        if not separator:
            raise ValueError("explicit Harbor datasets require org/name@sha256:digest")
        candidates.append({"source": "harbor", "id": name, "revision": revision})
    write_rows(output / "candidates.jsonl", candidates)
    for candidate in candidates:
        try:
            tasks.extend(
                sources.acquire(candidate, output / "downloads", args.tasks_per_dataset)
            )
        except Exception as error:
            excluded.append(
                {"candidate": candidate, "stage": "acquisition", "reason": str(error)}
            )
    for root in args.local_dataset:
        tasks.extend(local_tasks(root, args.tasks_per_dataset))
    write_rows(output / "tasks.jsonl", tasks)
    for task in tasks:
        key = hashlib.sha256(task["task_id"].encode()).hexdigest()[:24]
        subject = output / "subjects" / key
        try:
            original_digest = source_digest(Path(task["path"]))
            copy_subject(task["path"], subject, original_digest)
            from openenv.harbor.tasks import resolve_task_dirs

            if subject not in resolve_task_dirs(str(subject.parent), refresh=True):
                raise ValueError("OpenEnv did not load the acquired task")
            if coding_config is not None:
                from examples.hub_validation_pipeline.cloud import prepare

                plan_dir = output / "plans" / key
                plan = prepare(
                    subject,
                    coding_config,
                    plan_dir,
                    source={
                        name: task[name]
                        for name in (
                            "source",
                            "source_id",
                            "revision",
                            "task_path",
                            "task_id",
                        )
                    },
                )
                row = {
                    "task_id": task["task_id"],
                    "run_id": plan["run_id"],
                    "plan": str(plan_dir / "plan.json"),
                    "state": "prepared",
                    "accepted": False,
                }
                plans.append(row)
                write_rows(output / "planned.jsonl", plans)
                continue
            write_json(
                subject / "validation" / "openenv.json",
                {
                    **contract,
                    "original_source_digest": original_digest,
                },
            )
            report_path = output / "reports" / f"{key}.json"
            exit_code, report = validate_task(
                subject, report_path, args.profile, args.validation_timeout
            )
            envelope = admission(
                report,
                exit_code,
                subject,
                report_path.with_suffix(".artifacts"),
                args.profile,
                original_digest,
            )
            accepted.append(
                materialize(
                    {**task, **envelope, "subject": str(subject)},
                    output / "approved" / "tasks" / key,
                )
            )
        except Exception as error:
            stage = (
                "cloud_submission"
                if coding_config is not None and getattr(args, "submit_hf_jobs", False)
                else "cloud_preparation"
                if coding_config is not None
                else "validation"
            )
            excluded.append({**task, "stage": stage, "reason": str(error)})
        write_rows(output / "accepted.jsonl", accepted)
        write_rows(output / "excluded.jsonl", excluded)
    write_rows(output / "excluded.jsonl", excluded)
    write_rows(output / "accepted.jsonl", accepted)
    if coding_config is not None:
        write_rows(output / "planned.jsonl", plans)
        summary = {
            "profile": "harbor-coding-audit-v1",
            "phase": "submitted"
            if getattr(args, "submit_hf_jobs", False)
            else "prepared",
            "candidates": len(candidates),
            "tasks": len(tasks),
            "plans": len(plans),
            "excluded": len(excluded),
            "accepted": 0,
            "rfc008_complete": False,
            "note": "Plans/submitted Jobs are not qualification results; inspect each immutable run report.",
            "finished_at": time.time(),
        }
        write_json(output / "summary.json", summary)
        print(dumps(summary))
        return 0 if plans else 1
    results = (
        evaluate(
            accepted,
            output,
            llm_url=args.llm_url,
            model=args.model,
            harness=args.harness,
            expose=args.expose,
            timeout_s=args.model_timeout,
            step_limit=args.model_step_limit,
        )
        if args.llm_url and accepted
        else []
    )
    summary = {
        "profile": args.profile,
        "rfc008_complete": args.profile == "full" and bool(accepted),
        "candidates": len(candidates),
        "tasks": len(tasks),
        "accepted": len(accepted),
        "excluded": len(excluded),
        "model_trials": len(results),
        "model_requested": bool(args.llm_url),
        "finished_at": time.time(),
    }
    write_json(output / "summary.json", summary)
    print(dumps(summary))
    model_ok = len(results) == len(accepted) and all(
        row.get("result", {}).get("ok") is True and "cleanup_error" not in row
        for row in results
    )
    return 0 if accepted and (not args.llm_url or model_ok) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New run directory; must not exist"
    )
    parser.add_argument(
        "--contract",
        type=Path,
        help="Trusted reward/resource contract for local native Harbor validation",
    )
    parser.add_argument(
        "--coding-config",
        type=Path,
        help="Prepare the supplemental coding audit on HF Jobs instead of local Docker validation",
    )
    parser.add_argument("--profile", choices=["full", PILOT], default="full")
    parser.add_argument("--source", choices=["hf", "harbor", "both"], default="both")
    parser.add_argument("--dataset-limit", type=int, default=2)
    parser.add_argument("--tasks-per-dataset", type=int, default=2)
    parser.add_argument("--hf-dataset", action="append", default=[])
    parser.add_argument("--harbor-dataset", action="append", default=[])
    parser.add_argument("--local-dataset", action="append", type=Path, default=[])
    parser.add_argument("--validation-timeout", type=float, default=900)
    parser.add_argument("--llm-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--harness", default="opencode")
    parser.add_argument(
        "--expose", choices=["direct", "gradio", "cloudflare"], default="gradio"
    )
    parser.add_argument("--model-timeout", type=float, default=120)
    parser.add_argument("--model-step-limit", type=int, default=32)
    args = parser.parse_args()
    if args.coding_config is None and args.contract is None:
        parser.error("--contract or --coding-config is required")
    if args.coding_config is not None and (args.contract is not None or args.llm_url):
        parser.error(
            "--coding-config selects its own contract/model and cannot combine --contract or --llm-url"
        )
    if args.dataset_limit < 0 or args.tasks_per_dataset < 0:
        parser.error("discovery/task limits must be nonnegative")
    if (
        not all(
            math.isfinite(value) and value > 0
            for value in (args.validation_timeout, args.model_timeout)
        )
        or args.model_step_limit < 1
    ):
        parser.error("timeouts and model step limit must be positive")
    if args.llm_url and not args.model:
        parser.error("--model is required with --llm-url")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
