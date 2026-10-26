import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "validation"
VALID_MANIFEST_FIXTURES = [
    "served_min_pass",
    "no_oracle",
    "harbor_task_min",
    "posttrain_task_min",
]

INVALID_MANIFEST_FIXTURES = [
    "broken_manifest",
    "unpinned_judge",
]

# Must match severity-v1.json so the policy file cannot drift silently.
EXPECTED_POLICY = {
    "static.manifest": (1, "local", "fail"),
    "static.reproducible_build": (1, "local", "fail"),
    "static.image_hygiene": (1, "local", "warn"),
    "static.layout": (1, "local", "warn"),
    "static.sbom": (1, "local", "fail"),
    "static.oci_labels": (1, "local", "fail"),
    "static.resource_declaration": (1, "local", "fail"),
    "static.timeout_ceiling": (1, "local", "fail"),
    "static.dependency_pinning": (1, "local", "fail"),
    "static.task_distribution_pinning": (1, "local", "fail"),
    "runtime.reward_well_formed": (2, "local", "fail"),
    "runtime.rubric_introspectable": (2, "local", "warn"),
    "runtime.observation_schema": (2, "local", "fail"),
    "runtime.state_contract": (2, "local", "fail"),
    "runtime.trajectory_record": (2, "local", "warn"),
    "runtime.reward_attribution": (2, "local", "warn"),
    "runtime.tool_declaration_accuracy": (2, "local", "fail"),
    "runtime.task_declaration_accuracy": (2, "local", "fail"),
    "runtime.seed_control": (2, "local", "fail"),
    "runtime.episode_determinism": (2, "local", "fail"),
    "runtime.network_policy": (2, "local", "fail"),
    "runtime.host_containment": (2, "local", "fail"),
    "runtime.resource_bounds": (2, "local", "fail"),
    "runtime.episode_isolation": (2, "local", "fail"),
    "runtime.oracle_containment": (2, "local", "fail"),
    "semantic.oracle_max": (3, "local", "fail"),
    "semantic.floor_gap": (3, "local", "fail"),
    "semantic.canary_floor": (3, "local", "fail"),
    "semantic.no_solution_leakage": (3, "local", "fail"),
    "semantic.verifier_determinism": (3, "local", "fail"),
    "semantic.verifier_portability": (3, "local", "fail"),
    "semantic.resource_envelope": (3, "local", "fail"),
    "semantic.replayability": (3, "local", "fail"),
    "hub.layer_isolation": (1, "hub", "warn"),
    "hub.time_to_first_work": (2, "hub", "warn"),
    "hub.cosign_signature": (1, "hub", "warn"),
    "hub.cross_host_determinism": (2, "hub", "fail"),
    "hub.immutable_versioning": (1, "hub", "fail"),
    "statistical.reward_reachability": (4, "hub", "fail"),
    "statistical.difficulty_separation": (4, "hub", "fail"),
    "statistical.headroom": (4, "hub", "warn"),
    "statistical.variance_bound": (4, "hub", "fail"),
    "statistical.training_signal": (4, "hub", "advisory"),
    "statistical.adversarial_floor": (4, "hub", "fail"),
    "statistical.gameability_gap": (4, "hub", "warn"),
}


def load_fixture_manifest(name: str) -> dict:
    return json.loads((FIXTURES / name / "normalized_manifest.json").read_text())
