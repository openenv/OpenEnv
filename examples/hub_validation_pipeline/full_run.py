"""Freeze census metadata and prepare the reviewed cohort; --submit launches once.

Preparation acquires selected pinned public task trees, never paid compute. Only
cloud.submit publishes existing inputs/results. No catalog-wide execution claim.
"""

import argparse
import hashlib
import os
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import ceil
from pathlib import Path

from . import cloud, real_tasks, spend

MODEL_SOURCE = (
    "Qwen/Qwen3.8-27B",
    "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "deepinfra",
)
RATEBOOK = {
    "as_of": "2026-11-10",
    "input_usd_per_million": "0.15",
    "output_usd_per_million": "1.875",
    "cpu_basic_usd_per_hour": "0.01",
    "sandbox_port_usd_per_hour": "0.01",
    "child_lifetime_seconds": 86400,
    "startup_allowance_seconds": 300,
}
ASSUMPTIONS = [
    "Modeled launch reservation at recorded prices; not an account-enforced billing limit.",
    "Reservation covers the selected cohort, without builds; the shared ledger must account for prior/concurrent spending and retries.",
    "Every child reserves the SDK hard 86400-second lifetime plus 300-second startup; controllers reserve configured timeout plus 300 seconds, without an exposed port.",
    "Prior billed tokens stay below max_total_tokens before the last request; message bytes <= 4*total+8192, including tool/framing allowance; request context ceiling 262144.",
    "Source revision does not attest provider-served weights; rate/model/flavor changes require a different reviewed ratebook.",
    "Only the configured cohort has reviewed execution contracts; all remaining catalog tasks are metadata only. Native tasks are evaluation-only, never training data.",
    "An uncertain submission requires manual reconciliation; never automatically resume or retry.",
]


def _budget(value):
    try:
        number = Decimal(value) if type(value) is str else Decimal("NaN")
        if number.is_finite() and number > 0:
            return Fraction(number)
    except InvalidOperation:
        pass
    raise ValueError("budget_usd must be a finite positive decimal string")


def estimate(configs):
    """Return exact aggregate launch reservation and request/token ceilings."""
    if not configs:
        raise ValueError("A configured cohort is required")
    children = calls = inputs = outputs = controller_seconds = 0
    for raw in configs:
        config, limits = cloud._config(raw)
        if (
            limits["sandbox_remote_job_lifetime_seconds"]
            != RATEBOOK["child_lifetime_seconds"]
        ):
            raise ValueError("A different child lifetime requires a different ratebook")
        if any(config[key]["flavor"] != "cpu-basic" for key in ("cloud", "execution")):
            raise ValueError("A different flavor requires a different ratebook")
        for key in ("target", "generator"):
            model = config[key]
            if tuple(
                model[k] for k in ("model_id", "revision", "provider")
            ) != MODEL_SOURCE or model.get("provider_model_id"):
                raise ValueError(
                    "A different model/source/provider requires a different ratebook"
                )
            total, maximum = model["max_total_tokens"], model["max_tokens"]
            count = (
                limits["target_attempts"]
                if key == "target"
                else limits["generated_candidates"]
            )
            inputs += count * (
                min(262144, 4 * total + 8192) + (total if key == "target" else 0)
            )
            outputs += count * min(
                total, maximum * (model["max_steps"] if key == "target" else 1)
            )
        children += limits["sandbox_jobs_max"]
        calls += limits["inference_calls_max"]
        controller_seconds += (
            config["cloud"]["timeout_seconds"] + RATEBOOK["startup_allowance_seconds"]
        )
    inference = (
        inputs * Fraction(RATEBOOK["input_usd_per_million"])
        + outputs * Fraction(RATEBOOK["output_usd_per_million"])
    ) / 1000000
    cpu, port = (
        Fraction(RATEBOOK[key])
        for key in ("cpu_basic_usd_per_hour", "sandbox_port_usd_per_hour")
    )
    child_seconds = (
        RATEBOOK["child_lifetime_seconds"] + RATEBOOK["startup_allowance_seconds"]
    )
    compute = (
        children * child_seconds * (cpu + port) + controller_seconds * cpu
    ) / 3600
    cost = inference + compute
    micros = ceil(cost * 1000000)
    return {
        "sandbox_jobs_max": children,
        "controller_jobs_max": len(configs),
        "inference_calls_max": calls,
        "input_tokens_max": inputs,
        "output_tokens_max": outputs,
        "inference_usd_exact": str(inference),
        "compute_usd_exact": str(compute),
        "reservation_usd_exact": str(cost),
        "reservation_usd_rounded_up": f"{micros // 1000000}.{micros % 1000000:06d}",
    }


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(cloud._json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _hash(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _catalog(path, destination, kind):
    path = Path(path).absolute()
    with path.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Supply a small census summary, not a raw census")
    summary = cloud._loads(data)
    destination.write_bytes(data)
    references = (
        {
            key: summary[key]
            for key in ("compact_repo_inventory", "compact_identity_inventory")
        }
        if kind == "hf"
        else {
            key: summary["artifacts"][key]["path"]
            for key in ("task_identity_inventory", "dataset_version_membership_index")
        }
    )
    inventories = {}
    for name, reference in references.items():
        inventory = Path(reference)
        if not inventory.is_absolute():
            inventory = path.parent / inventory
        inventories[name] = (
            _hash(inventory)
            if inventory.is_file()
            else {"path": str(inventory), "status": "unavailable"}
        )
    return {
        **_hash(destination),
        "scope": summary["scope"],
        "counts": summary["counts"],
        "inventories": inventories,
        "runtime_qualified": False,
    }


def run(
    output,
    hf_census,
    harbor_census,
    budget_usd,
    *,
    panel_path,
    task_names=None,
    submit=False,
    spend_ledger=None,
    reservation_id=None,
):
    if (submit or spend_ledger is not None or reservation_id is not None) and (
        spend_ledger is None
        or (isinstance(spend_ledger, str) and not spend_ledger.strip())
        or not isinstance(reservation_id, str)
        or not reservation_id.strip()
    ):
        raise ValueError("Supply both a shared ledger and a reservation ID")
    cap = _budget(budget_usd)
    panel, panel_provenance = real_tasks._load_panel(panel_path)
    names = [task["name"] for task in real_tasks._select_tasks(panel, task_names)]
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema": "openenv-full-run-v1",
        "status": "preparing",
        "stage": "catalogs",
        "accepted": False,
        "rfc008_qualified": False,
        "trainable": False,
        "evaluation_only": True,
        "full_catalog_executed": False,
        "budget_usd": budget_usd,
        "ratebook": RATEBOOK,
        "assumptions": ASSUMPTIONS,
        "evaluation_panel": panel_provenance,
        "catalogs": {},
        "tasks": [],
    }
    _write(manifest_path, manifest)
    try:
        for kind, source in (("hf", hf_census), ("harbor", harbor_census)):
            manifest["catalogs"][kind] = _catalog(
                source, output / f"{kind}-census-summary.json", kind
            )
            _write(manifest_path, manifest)
        manifest["stage"] = "prepare"
        recipes = real_tasks._prepare(
            output / "recipes",
            panel=panel,
            panel_provenance=panel_provenance,
            task_names=names,
            bundle=True,
        )
        configs = []
        manifest["stage"] = "plans"
        for row in recipes["tasks"]:
            plan_sha256 = _hash(Path(row["bundle_plan"]))["sha256"]
            plan, config, _, _ = cloud._read_plan(
                row["bundle_plan"], expected_sha256=plan_sha256
            )
            configs.append(config)
            manifest["tasks"].append(
                {
                    "task_id": row["task_id"],
                    "plan": row["bundle_plan"],
                    "plan_sha256": plan_sha256,
                    "run_id": plan["run_id"],
                    "bundle_sha256": plan["bundle_sha256"],
                    "budget": plan["budget"],
                    "accepted": False,
                    "trainable": False,
                    "rfc008_qualified": False,
                    "submission": {"status": "not_submitted"},
                }
            )
            _write(manifest_path, manifest)
        manifest["stage"] = "budget"
        manifest["estimate"] = estimate(configs)
        manifest["within_budget"] = (
            Fraction(manifest["estimate"]["reservation_usd_exact"]) <= cap
        )
        manifest["status"] = "prepared"
        _write(manifest_path, manifest)
        if submit:
            if not manifest["within_budget"]:
                raise ValueError("Reservation exceeds budget")
            manifest["stage"] = "reserve"
            fingerprint = hashlib.sha256(
                "\0".join(task["bundle_sha256"] for task in manifest["tasks"]).encode()
            ).hexdigest()[:16]
            identifier = f"{reservation_id}:{fingerprint}"
            spend.reserve(
                spend_ledger,
                identifier,
                manifest["estimate"]["reservation_usd_rounded_up"],
                "Reviewed native Hub audit cohort",
                evidence=str(manifest_path),
            )
            manifest["shared_spend"] = {
                "ledger": str(Path(spend_ledger).absolute()),
                "reservation_id": identifier,
                "reserved_usd": manifest["estimate"]["reservation_usd_rounded_up"],
            }
            _write(manifest_path, manifest)
            manifest.update(stage="submit", status="submitting")
            for task in manifest["tasks"]:
                if _hash(Path(task["plan"]))["sha256"] != task["plan_sha256"]:
                    raise ValueError("Prepared plan changed before submission")
                task["submission"] = {
                    "status": "pending_or_unknown",
                    "receipt": str(Path(task["plan"]).parent / "submission.json"),
                }
                _write(manifest_path, manifest)
                receipt = cloud.submit(
                    task["plan"], expected_plan_sha256=task["plan_sha256"]
                )
                task["submission"].update(
                    {
                        key: receipt[key]
                        for key in (
                            "status",
                            "job_id",
                            "job_url",
                            "input_revision",
                            "receipt_sync_status",
                            "receipt_sync_error_type",
                        )
                        if key in receipt
                    }
                )
                _write(manifest_path, manifest)
            manifest["status"] = "submitted"
        manifest["stage"] = "done"
        _write(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        _write(manifest_path, manifest)
        raise RuntimeError(
            f"Full run stopped during {manifest['stage']}; inspect manifest.json and per-plan submission receipts"
        ) from None
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", dest="panel_path", type=Path, required=True)
    parser.add_argument("--hf-census", type=Path, required=True)
    parser.add_argument("--harbor-census", type=Path, required=True)
    parser.add_argument("--budget-usd", required=True)
    parser.add_argument("--task", action="append", dest="task_names")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--spend-ledger", type=Path)
    parser.add_argument("--reservation-id")
    args = parser.parse_args(argv)
    if args.submit and (args.spend_ledger is None or not args.reservation_id):
        parser.error("Paid submission requires --spend-ledger and --reservation-id")
    manifest = run(**vars(args))
    print(
        cloud._json_bytes(
            {
                "manifest": str(args.output.absolute() / "manifest.json"),
                "status": manifest["status"],
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()
