import pytest
from openenv.harbor.models import HarborRolloutResult
from openenv.harbor.qualification import evaluate_eval_capture, qualification_rows
from openenv.harbor.seams import SEAMS


def test_all_current_adapters_start_unqualified_for_every_provider():
    rows = qualification_rows(list(SEAMS))
    assert len(rows) == 29
    assert all(row[1:] == ["not_run"] * 4 for row in rows)


def test_eval_export_rejection_is_expected_and_zero_reward_is_valid():
    result = HarborRolloutResult(
        rollout_type="eval", capture_level="text", n_turns=2, reward=0.0
    )
    assert all(evaluate_eval_capture(result).values())
    result.reward = None
    assert not evaluate_eval_capture(result)["verifier_graded"]


def test_no_capture_cannot_be_certified_as_success():
    result = HarborRolloutResult(rollout_type="eval", capture_level="text", reward=0.0)
    assert not evaluate_eval_capture(result)["model_calls_captured"]


def test_pass_requires_evidence_and_is_not_promoted_to_optimizer():
    cell = {
        "harness": "opencode",
        "provider": "vllm",
        "status": "capture_and_reader_pass",
    }
    with pytest.raises(ValueError, match="evidence"):
        qualification_rows(["opencode"], {"cells": [cell]})
    cell["evidence"] = ["two-task-run.json"]
    assert (
        qualification_rows(["opencode"], {"cells": [cell]})[0][-1]
        == "capture_and_reader_pass"
    )
    with pytest.raises(ValueError, match="duplicate"):
        qualification_rows(["opencode"], {"cells": [cell, cell]})


def test_optimizer_status_requires_proof_for_current_captures():
    from openenv.harbor.qualification import qualification_details

    cell = {
        "harness": "opencode",
        "provider": "vllm",
        "status": "optimizer_pass",
        "evidence": ["capture.json"],
        "optimizer_validated": True,
    }
    with pytest.raises(ValueError, match="matching, scoped"):
        qualification_rows(["opencode"], {"cells": [cell]})
    proof = {
        "matches_current_captures": True,
        "result": "result.json",
        "inputs": "inputs.json",
        "scope": "diagnostic replay; no weight sync",
        "model": "test-model",
        "revision": "pinned",
        "rows": 2,
    }
    cell["optimizer_evidence"] = proof
    assert (
        qualification_rows(["opencode"], {"cells": [cell]})[0][-1] == "optimizer_pass"
    )
    assert (
        "diagnostic replay; no weight sync"
        in qualification_details({"cells": [cell]})[0][7]
    )
    proof["matches_current_captures"] = False
    with pytest.raises(ValueError, match="matching, scoped"):
        qualification_rows(["opencode"], {"cells": [cell]})
    cell["status"] = "capture_and_reader_pass"
    cell["optimizer_validated"] = False
    assert qualification_details({"cells": [cell]})[0][7].startswith(
        "previous captures only:"
    )


def test_unrecognized_provider_is_not_silently_hidden():
    with pytest.raises(ValueError, match="provider"):
        qualification_rows([], {"cells": [{"harness": "opencode", "provider": "typo"}]})


def test_maturity_requires_all_providers_and_current_optimizer():
    from openenv.harbor.qualification import harness_maturity_rows, PROVIDERS

    cells = [
        {"harness": "example", "provider": provider, "status": "failed"}
        for provider in PROVIDERS
    ]
    report = {"cells": cells}
    assert harness_maturity_rows(["example"], report)[0][1] == "unstable"
    assert harness_maturity_rows(["unmeasured"], report)[0][1] == "experimental"
    for cell in cells:
        cell.update(status="eval_pass", evidence=["capture.json"])
    cells[-1]["status"] = "capture_and_reader_pass"
    assert harness_maturity_rows(["example"], report)[0][1] == "experimental"
    cells[-1].update(
        status="optimizer_pass",
        optimizer_validated=True,
        optimizer_evidence={
            "matches_current_captures": True,
            "result": "result.json",
            "inputs": "inputs.json",
            "scope": "diagnostic",
            "model": "model",
            "revision": "pinned",
            "rows": 2,
        },
    )
    assert harness_maturity_rows(["example"], report)[0][1] == "stable"
    cells[1]["status"] = "failed"
    assert harness_maturity_rows(["example"], report)[0][1] == "experimental"
