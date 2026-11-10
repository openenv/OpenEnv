"""Validate a frozen cohort offline; --submit reserves and launches it once.

Only owned controllers are polled. Completed controllers remain unmeasured;
result collection and acceptance require separate evidence review.
"""

import argparse
import hashlib
import re
import time
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

from examples.coding_audit.hf import _deadline, _DeadlineExceeded

from . import cloud, full_run, spend

STAGES = "SCHEDULING RUNNING COMPLETED ERROR CANCELED DELETED".split()


def _plan(root, task):
    path = (root / cloud._safe_name(task["plan_path"])).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Plan must resolve inside the preparation directory")
    receipt = path.parent / "submission.json"
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("Submission receipt exists; manual review required")
    return path, cloud._read_plan(path, expected_sha256=task["plan_sha256"])


def _record(task, receipt):
    for key in ("job_id", "job_url", "status"):
        if isinstance(receipt.get(key), str):
            task["submission_status" if key == "status" else key] = receipt[key]


def _status(task, spec, owner, deadline, api, secrets, clock):
    if clock() >= deadline:
        return "TIMEOUT"
    try:
        with _deadline(deadline - clock()):
            job = api.inspect_job(
                job_id=task["job_id"],
                namespace=spec.get("namespace"),
                token=secrets["HF_SANDBOX_TOKEN"],
            )
    except _DeadlineExceeded:
        return "TIMEOUT"
    if (
        job.id != task["job_id"]
        or job.docker_image != spec["image"]
        or not owner
        or job.owner.name != owner
    ):
        raise ValueError("Controller identity mismatch")
    stage = job.status.stage
    if stage in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}:
        return stage
    if clock() >= deadline:
        return "TIMEOUT"
    return stage if stage in STAGES else "UNKNOWN"


def run(
    prepared_manifest,
    output,
    spend_ledger,
    reservation_id,
    budget_usd,
    *,
    submit=False,
    api=None,
    secrets=None,
    sleep=time.sleep,
    clock=time.monotonic,
):
    cap = full_run._budget(budget_usd)
    spend._text(reservation_id)
    if not str(spend_ledger or "").strip():
        raise ValueError("Supply a shared ledger and a reservation ID")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(prepared_manifest).resolve(strict=True)
    root, destination = source.parent, output / "manifest.json"
    source_bytes = source.read_bytes()
    binding = {
        "path": str(source),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "bytes": len(source_bytes),
    }
    manifest = dict(
        schema="openenv-prepared-batch-v1",
        status="preparing",
        stage="plans",
        prepared_manifest=binding,
        accepted=False,
        results_collected=False,
        budget_usd=budget_usd,
        tasks=[],
    )

    def save(**state):
        manifest.update(**state)
        full_run._write(destination, manifest)

    save()
    try:
        rows = cloud._loads(source_bytes)["tasks"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("A nonempty prepared cohort is required")
        configs, task_ids, run_ids = [], set(), set()
        for row in rows:
            if not isinstance(row["task_id"], str) or not row["task_id"].strip():
                raise ValueError("Task IDs must be nonempty strings")
            path, (plan, config, _, _) = _plan(root, row)
            if row["task_id"] in task_ids or plan["run_id"] in run_ids:
                raise ValueError("Task IDs, run IDs and plan paths must be distinct")
            task_ids.add(row["task_id"])
            run_ids.add(plan["run_id"])
            configs.append(config)
            task = {key: row[key] for key in ("task_id", "plan_path", "plan_sha256")}
            task.update(
                run_id=plan["run_id"],
                bundle_sha256=plan["bundle_sha256"],
                status="not_submitted",
                accepted=False,
            )
            manifest["tasks"].append(task)
        manifest["stage"] = "budget"
        manifest["estimate"] = full_run.estimate(configs)
        amount = manifest["estimate"]["reservation_usd_rounded_up"]
        manifest["within_budget"] = Fraction(amount) <= cap
        fingerprint = hashlib.sha256(
            "\0".join(task["bundle_sha256"] for task in manifest["tasks"]).encode()
        ).hexdigest()[:16]
        manifest["shared_spend"] = dict(
            ledger=str(Path(spend_ledger).absolute()),
            reservation_id=f"{reservation_id}:{fingerprint}",
            reserved_usd=amount,
            status="not_reserved",
        )
        manifest["status"] = "prepared"
        save()
        if not submit:
            save(stage="done")
            return manifest
        with _deadline(1):
            pass
        if not manifest["within_budget"]:
            raise ValueError("Aggregate reservation exceeds budget")
        if full_run._hash(source) != binding:
            raise ValueError("Preparation manifest changed")
        for task in manifest["tasks"]:
            _plan(root, task)
        manifest["stage"] = "reserve"
        manifest["shared_spend"]["status"] = "pending_or_unknown"
        save()
        spend.reserve(
            spend_ledger,
            manifest["shared_spend"]["reservation_id"],
            amount,
            "Frozen prepared cohort",
            evidence=str(destination),
            plan_costs={
                task["plan_sha256"]: full_run.estimate([config])[
                    "reservation_usd_exact"
                ]
                for task, config in zip(manifest["tasks"], configs, strict=True)
            },
        )
        manifest["shared_spend"]["status"] = "reserved"
        save()
        secrets = cloud._runtime_secrets(secrets)
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=secrets["HF_TOKEN"])
        job_ids = set()
        for offset in range(0, len(rows), 2):
            active = []
            for task in manifest["tasks"][offset : offset + 2]:
                if any(clock() >= entry[3] for entry in active):
                    raise TimeoutError("Controller deadline exceeded")
                if full_run._hash(source) != binding:
                    raise ValueError("Preparation manifest changed")
                path, (plan, config, _, _) = _plan(root, task)
                spec = plan["mcp_request"]["args"]
                task["status"] = "pending_or_unknown"
                save(stage="submit", status="submitting")
                deadline = (
                    clock()
                    + config["cloud"]["timeout_seconds"]
                    + full_run.RATEBOOK["startup_allowance_seconds"]
                )
                try:
                    receipt = cloud.submit(
                        path,
                        api=api,
                        secrets=secrets,
                        expected_plan_sha256=task["plan_sha256"],
                        spend_ledger=spend_ledger,
                        reservation_id=manifest["shared_spend"]["reservation_id"],
                    )
                except BaseException:
                    local = path.parent / "submission.json"
                    if local.is_file():
                        _record(task, cloud._loads(local.read_bytes()))
                    raise
                _record(task, receipt)
                save()
                if (
                    receipt["status"] != "submitted"
                    or receipt["run_id"] != task["run_id"]
                    or not task.get("job_id")
                ):
                    raise ValueError("Submission identity or outcome unknown")
                if task["job_id"] in job_ids:
                    raise ValueError("Duplicate controller Job ID")
                job_ids.add(task["job_id"])
                url = urlsplit(receipt["job_url"])
                parts = url.path.split("/")
                if (
                    url.scheme != "https"
                    or url.netloc != "huggingface.co"
                    or len(parts) != 4
                    or parts[:2] != ["", "jobs"]
                    or parts[-1] != task["job_id"]
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", parts[2])
                ):
                    raise ValueError("Controller owner URL is invalid")
                owner = spec.get("namespace") or parts[2]
                if owner != parts[2]:
                    raise ValueError(
                        "Controller owner differs from the launch namespace"
                    )
                task["status"] = "accepted_unmeasured"
                active.append((task, spec, owner, deadline))
                save()
            save(stage="poll", status="waiting")
            while active:
                for entry in active[:]:
                    task, spec, owner, deadline = entry
                    task["controller_status"] = "UNKNOWN"
                    stage = task["controller_status"] = _status(
                        task, spec, owner, deadline, api, secrets, clock
                    )
                    save()
                    if stage == "COMPLETED":
                        active.remove(entry)
                    elif stage not in {"SCHEDULING", "RUNNING"}:
                        raise RuntimeError("Controller failed or status unknown")
                if active:
                    sleep(max(0, min(30, min(entry[3] - clock() for entry in active))))
        manifest["status"] = "controllers_completed_unmeasured"
        save(stage="done")
    except BaseException as error:
        save(status="failed", error_type=type(error).__name__)
        raise RuntimeError(
            f"Prepared batch stopped during {manifest['stage']}; inspect manifest.json and per-plan receipts"
        ) from None
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    names = "prepared-manifest output spend-ledger reservation-id budget-usd"
    for name in names.split():
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    manifest = run(**vars(args))
    print(f"{manifest['status']}: {Path(args.output).absolute() / 'manifest.json'}")


if __name__ == "__main__":
    main()
