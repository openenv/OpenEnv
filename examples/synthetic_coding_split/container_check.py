"""Repeat controls on the pinned Python image, using public-only Docker builds."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from split import (
    BASE_IMAGE,
    control_cases,
    frozen_task,
    native_inventory,
    SPLIT,
    submission_names,
    test_results,
    tree_digest,
    validate_controls,
    write_json,
)


def docker(*args, timeout=300, combined=False, missing_ok=False):
    if args[0] in {"build", "create", "run"}:
        args = (args[0], "--platform", "linux/amd64", *args[1:])
    result = subprocess.run(
        ["docker", *map(str, args)], text=True, capture_output=True, timeout=timeout
    )
    if result.returncode:
        message = result.stderr or result.stdout
        if not (missing_ok and "No such container:" in message):
            raise ValueError(message)
    return (result.stdout + (result.stderr if combined else "")).strip()


@contextmanager
def owned_container():
    """Remove this run's container and anonymous volumes, including on CLI timeout."""
    name = "openenv-synthetic-" + uuid.uuid4().hex
    try:
        yield name
    finally:
        docker("rm", "-fv", name, missing_ok=True)


def run(task, spec, image, script):
    with tempfile.TemporaryDirectory(prefix="synthetic-container-") as name:
        temporary = Path(name)
        solver, verifier, reward = [
            temporary / x for x in ("solver", "verifier", "reward")
        ]
        # Name the container before creation so lost/failed CLI responses are removable.
        with owned_container() as container:
            docker("create", "--name", container, "--network", "none", image)
            docker("cp", f"{container}:/app", solver)
        verifier.mkdir()
        reward.mkdir()
        common = [
            "--network",
            "none",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "-e",
            "APP_DIR=/app",
            "-e",
            "PYTHON=python",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONHASHSEED=0",
            "-w",
            "/app",
        ]
        if script is not None:
            with owned_container() as container:
                docker(
                    "run",
                    "--name",
                    container,
                    *common,
                    "-v",
                    f"{solver}:/app",
                    "-v",
                    f"{script.resolve()}:/control.sh:ro",
                    image,
                    "/bin/bash",
                    "/control.sh",
                )
        transferred = {}
        for relative in submission_names(spec):
            source = solver / relative
            if source.is_symlink():
                raise ValueError("symlink submission")
            if source.is_file():
                target = verifier / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                transferred[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        # Grader starts with the generic Python image, containing no starter files.
        with owned_container() as container:
            log = docker(
                "run",
                "--name",
                container,
                *common,
                "-e",
                "TEST_DIR=/tests",
                "-e",
                "VERIFIER_LOG_DIR=/logs/verifier",
                "-v",
                f"{verifier}:/app",
                "-v",
                f"{task.resolve() / 'tests'}:/tests:ro",
                "-v",
                f"{reward}:/logs/verifier",
                BASE_IMAGE,
                "/bin/bash",
                "/tests/test.sh",
                timeout=60,
                combined=True,
            )
        raw = (reward / "reward.txt").read_text().strip()
        if raw not in {"0", "1"}:
            raise ValueError("malformed container reward")
        return {
            "reward": int(raw),
            **test_results(log),
            "submission_sha256": transferred,
            "verifier_output": log,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    inventory = native_inventory(args.task_root, args.inventory)
    label = "synthetic-gap-" + uuid.uuid4().hex
    evidence = {
        "schema_version": 1,
        "split": SPLIT,
        "protocol": "docker-offline-file-handoff-v1",
        "base_image": BASE_IMAGE,
        "model_difficulty": "unmeasured",
        "cloud_execution_qualified": False,
        "rfc008_complete": False,
        "tasks": [],
    }
    for task_id, expected_hashes in sorted(inventory["task_file_hashes"].items()):
        with frozen_task(args.task_root / task_id, expected_hashes) as (task, digest):
            spec = json.loads((task / "spec.json").read_text())
            tag = f"openenv-synthetic:{label}-{task_id}"
            try:
                image_id = docker(
                    "build",
                    "--quiet",
                    "--network",
                    "none",
                    "-t",
                    tag,
                    task / "environment",
                ).splitlines()[-1]
                details = json.loads(docker("image", "inspect", tag))[0]
                with owned_container() as container:
                    visibility = docker(
                        "run",
                        "--name",
                        container,
                        "--network",
                        "none",
                        tag,
                        "/bin/sh",
                        "-c",
                        "test ! -e /tests && test ! -e /solution && python --version",
                    )
                records = []
                for name, script, expected in control_cases(task, spec):
                    if tree_digest(task) != digest:
                        raise ValueError("frozen task changed before container control")
                    result = run(task, spec, tag, script)
                    if result["reward"] != expected:
                        raise ValueError(
                            f"{task_id}/{name}\n{result['verifier_output']}"
                        )
                    if tree_digest(task) != digest:
                        raise ValueError("frozen task changed during container control")
                    records.append({"case": name, "task_sha256": digest, **result})
                evidence["tasks"].append(
                    {
                        "task_id": task_id,
                        "task_sha256": digest,
                        "local_image_id": image_id,
                        "architecture": details["Architecture"],
                        "python": visibility,
                        "private_paths_absent": True,
                        "cases": records,
                    }
                )
                validate_controls(evidence["tasks"][-1], task, spec)
                print(
                    f"{task_id}: {len(records)} offline container controls passed",
                    flush=True,
                )
            finally:
                docker("image", "rm", tag)
    write_json(args.output, evidence)


if __name__ == "__main__":
    main()
