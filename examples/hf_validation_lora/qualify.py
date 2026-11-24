"""Qualify an explicitly scoped source bootstrap, retaining raw validation evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.basic import (
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.report import CheckResult, ValidationReportV2
from openenv.validation.runner import source_digest
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.collector import collect_runtime_evidence
from openenv.validation.runtime.contracts import IsolationEvidence, RuntimePlan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["math", "wordle"], required=True)
    a = p.parse_args()
    math = a.env == "math"
    root = Path("/opt/scott" if math else "/opt/wordle")
    sha = (
        "4de8125d314fca0a3f137fa1bab04b966a3efc15"
        if math
        else "a6ca4a7b3ab22052df265d358f41de34ffd351cc"
    )
    repo = "FineEnvs/ScottWuMath" if math else "FineEnvs/wordle-openenv"
    directory = Path("/tmp/qualification-" + a.env)
    directory.mkdir(exist_ok=True)
    manifest = NormalizedManifest.model_validate(
        {
            "manifest_schema_version": "1",
            "name": a.env,
            "signature": "openenv.yaml",
            "reward": {
                "range": [0, 1 if math else 1.5],
                "oracle_tolerance": 0,
                "floor_margin": 0.5,
            },
            "resources": {
                "cpu": 2,
                "memory_mb": 16000,
                "disk_mb": 50000,
                "episode_timeout_s": 30,
            },
            "capabilities": {"verifier": {"kind": "reward_channel"}},
            "types": {"tags": ["math" if math else "wordle"]},
        }
    )
    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {
                "episode_id": "qualification-fixed",
                "seed": 20261104,
                "options": {"split": "train", "index": 0}
                if math
                else {"answer": "apple"},
            },
            "actions": [{"expression": "10 + 13 - 3 + 4"}]
            if math
            else [{"tool_name": "guess", "arguments": {"word": "apple"}}],
        }
    )
    base = "http://127.0.0.1:" + ("8012" if math else "8001")
    sequential = collect_runtime_evidence(base, plan, episode_timeout_s=30, episodes=2)
    fresh = collect_runtime_evidence(base, plan, episode_timeout_s=30)
    # The complete two-episode stream is preserved, and a separately collected first
    # episode is used by basic graders and the isolation comparison.
    reference = collect_runtime_evidence(base, plan, episode_timeout_s=30)
    subject = Subject(
        root=root,
        manifest=manifest,
        image_ref=None,
        running=None,
        outputs_dir=directory,
        runtime_evidence=reference,
        isolation_evidence=IsolationEvidence(sequential=sequential, fresh=fresh),
    )
    results = [
        CheckResult(
            check_id="runtime.startup",
            status="fail" if reference.failure_reason else "pass",
            evidence=[
                reference.failure_reason
                or "schema and reset/step/state completed through internal WebSocket"
            ],
            duration_s=0,
        )
    ]
    results += [
        g().run(subject)
        for g in [
            RewardWellFormedGrader,
            ObservationSchemaGrader,
            StateContractGrader,
            EpisodeIsolationGrader,
        ]
    ]
    for name in [
        "network_policy",
        "host_containment",
        "resource_bounds",
        "oracle_containment",
        "seed_control",
        "episode_determinism",
        "rubric_introspectable",
        "trajectory_record",
        "reward_attribution",
        "tool_declaration_accuracy",
        "task_declaration_accuracy",
    ]:
        results.append(
            CheckResult(
                check_id="runtime." + name,
                status="skip",
                evidence=["not qualified by this source-bootstrap experiment"],
                duration_s=0,
            )
        )
    report = ValidationReportV2(
        report_schema_version="2",
        target="source-bootstrap:spaces/" + repo + "@" + sha,
        source_digest=source_digest(root),
        signature="openenv.yaml",
        manifest=manifest,
        policy_version="v2",
        lane="local",
        levels_run=[2],
        results=results,
        verdict="fail"
        if any(x.status.value in {"fail", "error"} for x in results)
        else "warn",
    )
    write_runtime_bundle(
        directory,
        report,
        plan=plan,
        evidence=reference,
        isolation=subject.isolation_evidence,
        provider={
            "name": "dedicated-hf-source-bootstrap",
            "scope": "internal protocol qualification; published Space image and resource enforcement unqualified",
        },
        cleanup={
            "required": True,
            "state": "owned sandbox retained for next experiment stage; external teardown record required",
        },
    )
    provenance = {
        "repo": repo,
        "revision": sha,
        "scope": "researcher overlay only; original package CLI refusal remains",
        "base_image": "python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28",
        "framework": "986af4195a1339b9ccb8d84b82cba1cb622c3c46",
        "dependency_mode": "framework override"
        if math
        else "original uv.lock frozen server; separate baseline validator",
        "source_hashes": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in root.rglob("*")
            if f.is_file()
            and not any(v in f.parts for v in [".venv", ".cache", "__pycache__"])
            and ".egg-info" not in str(f)
        },
    }
    (directory / "experiment-provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
