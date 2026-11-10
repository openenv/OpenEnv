"""Local-only checks for modeled reservation and durable one-shot orchestration."""

import hashlib
import json
import socket
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import pytest
from examples.hub_validation_pipeline import cloud, full_run, real_tasks, spend
from examples.hub_validation_pipeline.test_real_tasks import (
    toy_config,
    TOY_NAMES,
    TOY_SOURCE,
    write_panel,
)


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Live operations forbidden"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(real_tasks, "HfApi", forbidden)
    monkeypatch.setattr(cloud, "_runtime_secrets", forbidden)
    panel_path = write_panel(tmp_path)
    panel_sha256 = full_run._hash(panel_path)["sha256"]
    plans = {}

    def prepare(output, *, panel, panel_provenance, task_names, bundle):
        assert panel_provenance["sha256"] == panel_sha256
        tasks = {task["name"]: task for task in panel["tasks"]}
        assert bundle is True
        output.mkdir()
        rows = []
        for index, name in enumerate(task_names):
            config, budget = cloud._config(tasks[name]["config"])
            path = output / name / "plan.json"
            path.parent.mkdir()
            plan = {
                "run_id": f"coding-{index}",
                "bundle_sha256": str(index) * 64,
                "budget": budget,
            }
            path.write_bytes(cloud._json_bytes(plan))
            plans[str(path)] = (plan, config, b"", {})
            rows.append(
                {
                    "task_id": f"hf:{TOY_SOURCE['repo_id']}@{TOY_SOURCE['revision']}:fixtures/{name}",
                    "bundle_plan": str(path),
                }
            )
        result = {"tasks": rows}
        (output / "manifest.json").write_bytes(cloud._json_bytes(result))
        return result

    prepare = Mock(side_effect=prepare)
    monkeypatch.setattr(real_tasks, "_prepare", prepare)

    def read_plan(path, *, expected_sha256=None):
        if expected_sha256 is not None:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
                raise ValueError("Prepared plan changed before submission")
        return plans[str(path)]

    monkeypatch.setattr(cloud, "_read_plan", read_plan)
    submit = Mock(return_value={"status": "submitted", "job_id": "queued-job"})
    monkeypatch.setattr(cloud, "submit", submit)
    hf, harbor = tmp_path / "hf.json", tmp_path / "harbor.json"
    inventory = tmp_path / "inventory.json.gz"
    # Deliberately not JSON/gzip: inventory bytes must only be streamed/hashed.
    inventory.write_bytes(b"unparsed inventory" * 100000)
    missing = tmp_path / "missing.json"
    hf.write_text(
        json.dumps(
            {
                "scope": "Frozen loose native identities; no runtime qualification",
                "counts": {"repositories": 3, "native_tasks": 7},
                "compact_repo_inventory": str(inventory),
                "compact_identity_inventory": str(missing),
                "raw_census": str(tmp_path / "do-not-read-raw.json"),
            }
        )
    )
    harbor.write_text(
        json.dumps(
            {
                "scope": "Anonymous native registry metadata only",
                "counts": {
                    "dataset_versions_listed": 9,
                    "dataset_names": 5,
                    "unique_task_digests": 11,
                },
                "artifacts": {
                    key: {"path": str(inventory)}
                    for key in (
                        "task_identity_inventory",
                        "dataset_version_membership_index",
                    )
                },
            }
        )
    )
    return tmp_path / "run", hf, harbor, prepare, submit, plans


def assert_prepared(prepare, output, names):
    prepare.assert_called_once()
    assert prepare.call_args.args == (output / "recipes",)
    assert prepare.call_args.kwargs["task_names"] == names
    assert prepare.call_args.kwargs["bundle"] is True
    assert (
        prepare.call_args.kwargs["panel_provenance"]["sha256"]
        == full_run._hash(output.parent / "panel.json")["sha256"]
    )


@pytest.fixture
def paid_reservation(tmp_path):
    ledger = tmp_path / "spend.json"
    spend.init(ledger, "500")
    return {"spend_ledger": ledger, "reservation_id": "test-cohort"}


@pytest.mark.parametrize("supplied", [False, True], ids=["omitted", "exhausted"])
def test_paid_api_cannot_bypass_fully_reserved_global_ledger(local_run, supplied):
    output, hf, harbor, prepare, submit, _ = local_run
    ledger = output.parent / "spend.json"
    spend.init(ledger, "500")
    spend.reserve(ledger, "prior-unresolved", "500", "Prior unresolved work")
    arguments = (
        {"spend_ledger": ledger, "reservation_id": "toy-selection-v3"}
        if supplied
        else {}
    )
    error, message = (
        (RuntimeError, "during reserve")
        if supplied
        else (ValueError, "shared ledger and a reservation ID")
    )
    with pytest.raises(error, match=message):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            task_names=["toy-alpha"],
            submit=True,
            **arguments,
            panel_path=output.parent / "panel.json",
        )
    submit.assert_not_called()
    if not supplied:
        prepare.assert_not_called()
        assert not output.exists()
    assert spend.summary(ledger)["reserved_usd_exact"] == "500"
    assert spend.summary(ledger)["remaining_usd_exact"] == "0"


@pytest.mark.parametrize(
    "ledger_value,identifier",
    [
        (None, None),
        (None, "cohort"),
        ("", "cohort"),
        ("  ", "cohort"),
        ("initialized", None),
        ("initialized", ""),
        ("initialized", "  "),
    ],
)
def test_paid_api_requires_nonempty_shared_reservation_before_preparation(
    local_run, paid_reservation, monkeypatch, ledger_value, identifier
):
    output, hf, harbor, prepare, submit, _ = local_run
    monkeypatch.chdir(output.parent)
    ledger = paid_reservation["spend_ledger"]
    with pytest.raises(ValueError, match="shared ledger and a reservation ID"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            spend_ledger=ledger if ledger_value == "initialized" else ledger_value,
            reservation_id=identifier,
            panel_path=output.parent / "panel.json",
        )
    assert not output.exists()
    prepare.assert_not_called()
    submit.assert_not_called()
    assert spend.summary(ledger)["reserved_usd_exact"] == "0"


@pytest.mark.parametrize("empty_file", [False, True], ids=["missing", "empty"])
def test_paid_api_requires_initialized_ledger_before_submission(local_run, empty_file):
    output, hf, harbor, _, submit, _ = local_run
    ledger = output.parent / "spend.json"
    if empty_file:
        ledger.write_bytes(b"")
    with pytest.raises(RuntimeError, match="during reserve"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            spend_ledger=ledger,
            reservation_id="test-cohort",
            panel_path=output.parent / "panel.json",
        )
    submit.assert_not_called()


def test_exact_aggregate_reservation_and_normalized_generator():
    configs = [toy_config(name) for name in TOY_NAMES]
    for config in configs:
        config.pop("generator")
    result = full_run.estimate(configs)
    # Missing generator inherits the target's request limits.
    inputs_per_task = 2 * (4 * 256 + 8192 + 256) + (4 * 256 + 8192)
    outputs_per_task = 2 * 3 * 64 + 64
    assert result["input_tokens_max"] == 2 * inputs_per_task
    assert result["output_tokens_max"] == 2 * outputs_per_task
    assert result["controller_jobs_max"] == 2
    budgets = [cloud._config(config)[1] for config in configs]
    children = sum(budget["sandbox_jobs_max"] for budget in budgets)
    assert result["sandbox_jobs_max"] == children
    inference = (
        2 * inputs_per_task * Fraction("0.15")
        + 2 * outputs_per_task * Fraction("1.875")
    ) / 1000000
    compute = (children * 86700 * Fraction("0.02") + 2 * 360 * Fraction("0.01")) / 3600
    assert Fraction(result["inference_usd_exact"]) == inference
    assert Fraction(result["compute_usd_exact"]) == compute
    assert Fraction(result["reservation_usd_exact"]) == inference + compute
    assert all(isinstance(value, (str, int)) for value in result.values())


def test_operator_runtime_limits_drive_reservation():
    configs = [toy_config(name) for name in TOY_NAMES]
    base = full_run.estimate(configs)
    configs[0]["target"]["max_total_tokens"] *= 2
    changed = full_run.estimate(configs)
    assert changed["input_tokens_max"] > base["input_tokens_max"]
    assert Fraction(changed["reservation_usd_exact"]) > Fraction(
        base["reservation_usd_exact"]
    )


@pytest.mark.parametrize(
    "names", [[], ["unknown"], ["toy-alpha", "unknown"], ["toy-beta"] * 2]
)
def test_invalid_task_selection_is_rejected_before_preparation(local_run, names):
    output, hf, harbor, prepare, submit, _ = local_run
    ledger = output.parent / "spend.json"
    spend.init(ledger, "500")
    with pytest.raises(ValueError, match="distinct configured panel task names"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            task_names=names,
            submit=True,
            spend_ledger=ledger,
            reservation_id="invalid-cohort",
            panel_path=output.parent / "panel.json",
        )
    assert not output.exists()
    prepare.assert_not_called()
    submit.assert_not_called()
    assert spend.summary(ledger)["reserved_usd_exact"] == "0"


def test_selection_budgets_and_submits_only_selected_task(local_run):
    output, hf, harbor, prepare, submit, _ = local_run
    ledger = output.parent / "spend.json"
    spend.init(ledger, "30")
    spend.reserve(ledger, "prior-launch", "5", "Unreconciled prior launch")
    manifest = full_run.run(
        output,
        hf,
        harbor,
        "15",
        task_names=["toy-alpha"],
        submit=True,
        spend_ledger=ledger,
        reservation_id="toy-selection-v2",
        panel_path=output.parent / "panel.json",
    )
    assert_prepared(prepare, output, ["toy-alpha"])
    assert manifest["status"] == "submitted" and manifest["within_budget"] is True
    assert manifest == cloud._loads((output / "manifest.json").read_bytes())
    assert len(manifest["tasks"]) == 1
    task = manifest["tasks"][0]
    assert task["task_id"] == (
        f"hf:{TOY_SOURCE['repo_id']}@{TOY_SOURCE['revision']}:fixtures/toy-alpha"
    )
    assert task["submission"]["status"] == "submitted"
    submit.assert_called_once_with(
        task["plan"],
        expected_plan_sha256=task["plan_sha256"],
        spend_ledger=ledger,
        reservation_id=manifest["shared_spend"]["reservation_id"],
    )
    assert not (output / "recipes/toy-beta").exists()
    assert "toy-beta" not in (output / "manifest.json").read_text()
    estimate = manifest["estimate"]
    assert estimate == full_run.estimate([toy_config("toy-alpha")])
    assert estimate["controller_jobs_max"] == 1
    assert Fraction(spend.summary(ledger)["reserved_usd_exact"]) == (
        5 + Fraction(estimate["reservation_usd_rounded_up"])
    )
    assert manifest["shared_spend"]["reservation_id"].startswith("toy-selection-v2:")
    with pytest.raises(FileExistsError):
        full_run.run(
            output,
            hf,
            harbor,
            "15",
            task_names=["toy-alpha"],
            submit=True,
            spend_ledger=ledger,
            reservation_id="toy-selection-v2",
            panel_path=output.parent / "panel.json",
        )
    assert submit.call_count == prepare.call_count == 1


def test_selected_task_still_obeys_global_spend_cap(local_run):
    output, hf, harbor, _, submit, _ = local_run
    ledger = output.parent / "spend.json"
    spend.init(ledger, "15")
    spend.reserve(ledger, "prior-launch", "10", "Unreconciled prior launch")
    with pytest.raises(RuntimeError, match="during reserve"):
        full_run.run(
            output,
            hf,
            harbor,
            "15",
            task_names=["toy-alpha"],
            submit=True,
            spend_ledger=ledger,
            reservation_id="toy-selection-v2",
            panel_path=output.parent / "panel.json",
        )
    submit.assert_not_called()
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert manifest["within_budget"] is True
    assert len(manifest["tasks"]) == 1
    assert manifest["tasks"][0]["submission"]["status"] == "not_submitted"
    assert spend.summary(ledger)["reserved_usd_exact"] == "10"


@pytest.mark.parametrize(
    "budget", ["0", "-1", "NaN", "Infinity", "1/2", "invalid", 50.0, True, None]
)
def test_invalid_cap_is_rejected_before_preparation(
    local_run, paid_reservation, budget
):
    output, hf, harbor, prepare, submit, _ = local_run
    with pytest.raises(ValueError, match="decimal string"):
        full_run.run(
            output,
            hf,
            harbor,
            budget,
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    assert not output.exists()
    prepare.assert_not_called()
    submit.assert_not_called()


def test_prepare_only_freezes_small_summaries_and_hashes_inventories(local_run):
    output, hf, harbor, prepare, submit, _ = local_run
    manifest = full_run.run(
        output, hf, harbor, "500", panel_path=output.parent / "panel.json"
    )
    assert_prepared(prepare, output, list(TOY_NAMES))
    submit.assert_not_called()
    assert manifest == cloud._loads((output / "manifest.json").read_bytes())
    assert manifest["status"] == "prepared" and manifest["within_budget"] is True
    assert manifest["full_catalog_executed"] is False
    assert (
        manifest["accepted"]
        is manifest["rfc008_qualified"]
        is manifest["trainable"]
        is False
    )
    assert len(manifest["tasks"]) == 2
    for kind, source in (("hf", hf), ("harbor", harbor)):
        record = manifest["catalogs"][kind]
        assert Path(record["path"]).read_bytes() == source.read_bytes()
        assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert record["runtime_qualified"] is False
    inventories = manifest["catalogs"]["hf"]["inventories"]
    record = inventories["compact_repo_inventory"]
    assert (
        record["sha256"]
        == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    )
    assert inventories["compact_identity_inventory"]["status"] == "unavailable"
    assert manifest["catalogs"]["hf"]["counts"]["native_tasks"] == 7
    assert not (output.parent / "spend.json").exists()
    with pytest.raises(FileExistsError):
        full_run.run(output, hf, harbor, "500", panel_path=output.parent / "panel.json")
    prepare.assert_called_once()


@pytest.mark.parametrize(
    "cap", ["1", "0.0000000000000000000000000000000000000000000000000000000001"]
)
def test_aggregate_cap_stops_before_any_submit_and_preserves_prepared_manifest(
    local_run, paid_reservation, cap
):
    output, hf, harbor, _, submit, _ = local_run
    with pytest.raises(RuntimeError, match="during budget"):
        full_run.run(
            output,
            hf,
            harbor,
            cap,
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    submit.assert_not_called()
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert manifest["within_budget"] is False and manifest["stage"] == "budget"
    assert len(manifest["tasks"]) == 2 and (output / "recipes/manifest.json").exists()
    assert all(
        task["submission"]["status"] == "not_submitted" for task in manifest["tasks"]
    )


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("target", "model_id", "other/model"),
        ("generator", "provider", "other"),
        ("target", "revision", "a" * 40),
        ("generator", "provider_model_id", "different-model"),
        ("execution", "flavor", "cpu-upgrade"),
        ("cloud", "flavor", "cpu-upgrade"),
    ],
)
def test_unquoted_profiles_are_rejected(section, field, value):
    config = deepcopy(toy_config("toy-beta"))
    config[section][field] = value
    with pytest.raises(ValueError):
        full_run.estimate([config])


def test_partial_ambiguous_submission_is_durable_safe_and_never_resumed(
    local_run, paid_reservation
):
    output, hf, harbor, prepare, submit, _ = local_run
    observed = []

    def launch(path, *, expected_plan_sha256, spend_ledger, reservation_id):
        assert full_run._hash(Path(path))["sha256"] == expected_plan_sha256
        durable = cloud._loads((output / "manifest.json").read_bytes())
        assert durable["shared_spend"]["ledger"] == str(
            paid_reservation["spend_ledger"]
        )
        assert Fraction(
            spend.summary(paid_reservation["spend_ledger"])["reserved_usd_exact"]
        ) == Fraction(durable["estimate"]["reservation_usd_rounded_up"])
        assert spend_ledger == paid_reservation["spend_ledger"]
        assert reservation_id == durable["shared_spend"]["reservation_id"]
        index = len(observed)
        observed.append(path)
        assert durable["tasks"][index]["submission"]["status"] == "pending_or_unknown"
        if index:
            assert durable["tasks"][0]["submission"]["job_id"] == "job-first"
            (Path(path).parent / "submission.json").write_text(
                '{"status":"launch_unknown"}'
            )
            raise RuntimeError("sensitive-error-value-must-not-escape")
        return {
            "status": "submitted",
            "job_id": "job-first",
            "launch_spec": {"command": ["omitted"]},
        }

    submit.side_effect = launch
    with pytest.raises(RuntimeError, match="during submit") as failure:
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert manifest["status"] == "failed" and manifest["accepted"] is False
    assert manifest["tasks"][0]["submission"]["status"] == "submitted"
    assert manifest["tasks"][1]["submission"]["status"] == "pending_or_unknown"
    receipt = Path(manifest["tasks"][1]["submission"]["receipt"])
    assert json.loads(receipt.read_text())["status"] == "launch_unknown"
    assert (
        "sensitive-error-value"
        not in str(failure.value) + (output / "manifest.json").read_text()
    )
    assert "launch_spec" not in manifest["tasks"][0]["submission"]
    assert Fraction(
        spend.summary(paid_reservation["spend_ledger"])["reserved_usd_exact"]
    ) == Fraction(manifest["estimate"]["reservation_usd_rounded_up"])
    with pytest.raises(FileExistsError):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    assert submit.call_count == 2 and prepare.call_count == 1


def test_submitted_jobs_remain_unaccepted(local_run, paid_reservation):
    output, hf, harbor, _, submit, _ = local_run
    manifest = full_run.run(
        output,
        hf,
        harbor,
        "500",
        submit=True,
        **paid_reservation,
        panel_path=output.parent / "panel.json",
    )
    assert submit.call_count == 2 and manifest["status"] == "submitted"
    assert manifest["shared_spend"]["ledger"] == str(paid_reservation["spend_ledger"])
    assert (
        manifest["accepted"]
        is manifest["trainable"]
        is manifest["rfc008_qualified"]
        is False
    )
    assert all(
        task["accepted"] is task["trainable"] is task["rfc008_qualified"] is False
        for task in manifest["tasks"]
    )


def test_prepare_failure_preserves_own_manifest_and_partial_recipes(
    local_run, paid_reservation
):
    output, hf, harbor, prepare, submit, _ = local_run

    def fail(path, **kwargs):
        path.mkdir()
        (path / "partial.json").write_text("{}")
        raise ValueError("untrusted exception text")

    prepare.side_effect = fail
    with pytest.raises(RuntimeError, match="during prepare"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert manifest["stage"] == "prepare" and manifest["error_type"] == "ValueError"
    assert (output / "recipes/partial.json").exists()
    assert "untrusted exception text" not in (output / "manifest.json").read_text()
    submit.assert_not_called()


def test_second_plan_failure_preserves_first_plan_and_stops_all_submits(
    local_run, paid_reservation, monkeypatch
):
    output, hf, harbor, _, submit, plans = local_run

    def read(path, *, expected_sha256=None):
        if Path(path).parent.name == "toy-beta":
            raise ValueError("bad plan")
        return plans[str(path)]

    monkeypatch.setattr(cloud, "_read_plan", read)
    with pytest.raises(RuntimeError, match="during plans"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert len(manifest["tasks"]) == 1 and manifest["accepted"] is False
    assert (output / "recipes/toy-beta/plan.json").exists()
    submit.assert_not_called()


def test_changed_plan_cannot_reuse_reviewed_estimate(
    local_run, paid_reservation, monkeypatch
):
    output, hf, harbor, _, submit, plans = local_run
    estimate = full_run.estimate

    def change(configs):
        result = estimate(configs)
        Path(next(iter(plans))).write_text("changed")
        return result

    monkeypatch.setattr(full_run, "estimate", change)
    with pytest.raises(RuntimeError, match="during submit"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            **paid_reservation,
            panel_path=output.parent / "panel.json",
        )
    submit.assert_not_called()
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert Fraction(
        spend.summary(paid_reservation["spend_ledger"])["reserved_usd_exact"]
    ) == Fraction(manifest["estimate"]["reservation_usd_rounded_up"])


def test_raw_census_is_rejected_without_preparation(local_run):
    output, hf, harbor, prepare, _, _ = local_run
    hf.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(RuntimeError, match="during catalogs"):
        full_run.run(output, hf, harbor, "500", panel_path=output.parent / "panel.json")
    prepare.assert_not_called()


def test_cli_requires_explicit_budget_and_defaults_to_preparation(local_run, capsys):
    output, hf, harbor, _, submit, _ = local_run
    arguments = [
        "--panel",
        str(output.parent / "panel.json"),
        "--output",
        str(output),
        "--hf-census",
        str(hf),
        "--harbor-census",
        str(harbor),
    ]
    with pytest.raises(SystemExit):
        full_run.main(arguments)
    assert not output.exists()
    full_run.main(arguments + ["--budget-usd", "500"])
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"
    submit.assert_not_called()


@pytest.mark.parametrize("names", [["toy-alpha"], ["toy-beta", "toy-alpha"]])
def test_cli_passes_repeatable_task_selection(local_run, capsys, names):
    output, hf, harbor, prepare, submit, _ = local_run
    arguments = [
        "--panel",
        str(output.parent / "panel.json"),
        "--output",
        str(output),
        "--hf-census",
        str(hf),
        "--harbor-census",
        str(harbor),
        "--budget-usd",
        "500",
    ]
    for name in names:
        arguments.extend(["--task", name])
    full_run.main(arguments)
    assert_prepared(prepare, output, names)
    manifest = cloud._loads((output / "manifest.json").read_bytes())
    assert [Path(task["plan"]).parent.name for task in manifest["tasks"]] == names
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"
    submit.assert_not_called()


@pytest.mark.parametrize(
    "selection", [["--task", "unknown"], ["--task", ""], ["--task"]]
)
def test_cli_rejects_unknown_or_empty_task_before_preparation(local_run, selection):
    output, hf, harbor, prepare, submit, _ = local_run
    with pytest.raises((SystemExit, ValueError)):
        full_run.main(
            [
                "--panel",
                str(output.parent / "panel.json"),
                "--output",
                str(output),
                "--hf-census",
                str(hf),
                "--harbor-census",
                str(harbor),
                "--budget-usd",
                "500",
                *selection,
            ]
        )
    assert not output.exists()
    prepare.assert_not_called()
    submit.assert_not_called()


def test_shared_ledger_blocks_before_any_paid_submission(local_run):
    output, hf, harbor, _, submit, _ = local_run
    ledger = output.parent / "spend.json"
    spend.init(ledger, "1")
    with pytest.raises(RuntimeError, match="during reserve"):
        full_run.run(
            output,
            hf,
            harbor,
            "500",
            submit=True,
            spend_ledger=ledger,
            reservation_id="cohort-v2",
            panel_path=output.parent / "panel.json",
        )
    submit.assert_not_called()
    assert spend.summary(ledger)["reserved_usd_exact"] == "0"


def test_shared_reservation_is_bound_to_the_immutable_cohort(local_run):
    output, hf, harbor, _, submit, _ = local_run
    ledger = output.parent / "spend.json"
    spend.init(ledger, "500")
    manifest = full_run.run(
        output,
        hf,
        harbor,
        "500",
        submit=True,
        spend_ledger=ledger,
        reservation_id="cohort-v2",
        panel_path=output.parent / "panel.json",
    )
    assert submit.call_count == 2
    assert manifest["shared_spend"]["reservation_id"].startswith("cohort-v2:")
    assert Fraction(spend.summary(ledger)["reserved_usd_exact"]) == Fraction(
        manifest["estimate"]["reservation_usd_rounded_up"]
    )


def test_paid_cli_requires_shared_reservation(local_run):
    output, hf, harbor, _, submit, _ = local_run
    with pytest.raises((SystemExit, ValueError)):
        full_run.main(
            [
                "--panel",
                str(output.parent / "panel.json"),
                "--output",
                str(output),
                "--hf-census",
                str(hf),
                "--harbor-census",
                str(harbor),
                "--budget-usd",
                "500",
                "--submit",
            ]
        )
    assert not output.exists()
    submit.assert_not_called()


def test_full_run_reads_and_binds_one_panel_snapshot(local_run, monkeypatch):
    output, hf, harbor, prepare, _, _ = local_run
    panel_path = output.parent / "panel.json"
    original_bytes = panel_path.read_bytes()
    read_file = cloud._text_file
    reads = []

    def replace_after_read(path):
        result = read_file(path)
        if path == panel_path:
            reads.append(path)
            panel_path.write_text('{"replacement": true}')
        return result

    monkeypatch.setattr(cloud, "_text_file", replace_after_read)
    manifest = full_run.run(output, hf, harbor, "500", panel_path=panel_path)
    assert reads == [panel_path]
    assert (
        manifest["evaluation_panel"]["sha256"]
        == hashlib.sha256(original_bytes).hexdigest()
    )
    assert prepare.call_args.kwargs["panel_provenance"] == manifest["evaluation_panel"]
    assert prepare.call_args.kwargs["panel"]["tasks"][0]["config"] == toy_config()


def test_full_run_requires_explicit_panel_before_preparation(local_run):
    output, hf, harbor, prepare, submit, _ = local_run
    with pytest.raises(TypeError, match="panel_path"):
        full_run.run(output, hf, harbor, "500")
    with pytest.raises(SystemExit):
        full_run.main(
            [
                "--output",
                str(output),
                "--hf-census",
                str(hf),
                "--harbor-census",
                str(harbor),
                "--budget-usd",
                "500",
            ]
        )
    assert not output.exists()
    prepare.assert_not_called()
    submit.assert_not_called()
