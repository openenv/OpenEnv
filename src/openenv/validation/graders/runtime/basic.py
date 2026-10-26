"""Pure graders over independently collected, uncoerced wire evidence."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

from ...report import CheckResult
from ...runtime.artifacts import _redact
from ...types import CheckStatus, Level


def _remote_reference(value) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                return True
            if _remote_reference(child):
                return True
    elif isinstance(value, list):
        return any(_remote_reference(child) for child in value)
    return False


class _RuntimeGrader:
    level = Level.RUNTIME
    requires_capabilities = frozenset()
    requires_provider = frozenset()
    depends_on = ("runtime.startup",)

    def applies_to(self, manifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        evidence = subject.runtime_evidence
        if evidence is None:
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.SKIP,
                evidence=["runtime evidence is unavailable"],
                duration_s=0,
            )
        if evidence.failure_reason:
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.SKIP,
                evidence=["unmet dependency: runtime.startup (collection incomplete)"],
                duration_s=0,
            )
        if not any(row.operation == "step" for row in evidence.exchanges):
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.SKIP,
                evidence=["no step was observed; the episode contract is incomplete"],
                duration_s=0,
            )
        problems = []
        try:
            problems.extend(self.check(subject, evidence))
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            problems.append("malformed runtime evidence")
        return CheckResult(
            check_id=self.check_id,
            status=CheckStatus.FAIL if problems else CheckStatus.PASS,
            measured={"exchanges": len(evidence.exchanges)},
            evidence=problems[:20] or ["observed runtime contract holds"],
            remediation=(
                "Fix the reported runtime contract violations." if problems else None
            ),
            duration_s=time.monotonic() - started,
        )


class RewardWellFormedGrader(_RuntimeGrader):
    """Require numeric step rewards for Level Two, beyond the core API contract."""

    check_id = "runtime.reward_well_formed"

    def check(self, subject, evidence) -> list[str]:
        problems = []
        observations = 0
        low, high = subject.manifest.reward.range
        for index, exchange in enumerate(evidence.exchanges):
            if exchange.operation not in {"reset", "step"}:
                continue
            observations += 1
            data = json.loads(exchange.response_json)["data"]
            if not isinstance(data, dict):
                problems.append(f"exchange {index}: malformed observation envelope")
                continue
            if "reward" not in data:
                problems.append(f"exchange {index}: missing reward")
                continue
            reward = data["reward"]
            if reward is None and exchange.operation == "reset":
                continue
            if (
                type(reward) not in (int, float)
                or (type(reward) is float and not math.isfinite(reward))
                or not low <= reward <= high
            ):
                problems.append(f"exchange {index}: reward must be finite and in range")
        if observations == 0:
            problems.append("no observations were measured")
        return problems


class ObservationSchemaGrader(_RuntimeGrader):
    """Validate raw envelopes and reconstructed observations against the schema."""

    check_id = "runtime.observation_schema"

    def check(self, subject, evidence) -> list[str]:
        if evidence.observation_schema_json is None:
            return ["observation schema unavailable"]
        schema = json.loads(evidence.observation_schema_json)
        # Submitted schemas must not cause host-side HTTP/file retrieval.
        if _remote_reference(schema):
            return ["observation schema has a non-local reference"]
        if evidence.reset_observation_schema_json is not None:
            reset_schema = json.loads(evidence.reset_observation_schema_json)
            if _remote_reference(reset_schema):
                return ["reset observation schema has a non-local reference"]
        problems = []
        observations = []
        count = 0
        for index, exchange in enumerate(evidence.exchanges):
            if exchange.operation not in {"reset", "step"}:
                continue
            count += 1
            response = json.loads(exchange.response_json)
            data = response["data"]
            if (
                response.get("type") != "observation"
                or not isinstance(data, dict)
                or not isinstance(data.get("observation"), dict)
                or type(data.get("done")) is not bool
                or "reward" not in data
            ):
                problems.append(f"exchange {index}: malformed observation envelope")
                continue
            # Preserve the bounded wire representation across the subprocess
            # boundary: normalizing numbers such as 1e9 can inflate a valid
            # episode beyond the worker's input limit.
            observations.append(
                {
                    "index": index,
                    "operation": exchange.operation,
                    "response_json": exchange.response_json,
                }
            )
        if count == 0:
            problems.append("no observations were measured")
        # A subject-supplied regex or recursive schema can exhaust CPU. Keep all
        # schema evaluation in a disposable interpreter with a hard wall deadline.
        worker = Path(__file__).resolve().parents[2] / "runtime" / "schema_worker.py"
        try:
            checked = subprocess.run(
                [sys.executable, "-I", str(worker)],
                input=json.dumps(
                    {
                        "schema_json": evidence.observation_schema_json,
                        "reset_schema_json": evidence.reset_observation_schema_json,
                        "observations": observations,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
                env={},
            )
            if checked.returncode:
                problems.append(
                    "observation schema evaluation exceeded its resource budget"
                )
            else:
                problems.extend(_redact(json.loads(checked.stdout)))
        except subprocess.TimeoutExpired:
            problems.append("observation schema evaluation exceeded its time budget")
        return problems


class StateContractGrader(_RuntimeGrader):
    """Verify episode identity and step counts in the same replayed session."""

    check_id = "runtime.state_contract"

    def check(self, subject, evidence) -> list[str]:
        problems = []
        episode_id = None
        steps = 0
        states = 0
        for index, exchange in enumerate(evidence.exchanges):
            if exchange.operation == "reset":
                episode_id = json.loads(exchange.request_json)["data"]["episode_id"]
                steps = 0
            elif exchange.operation == "step":
                steps += 1
            elif exchange.operation == "state":
                states += 1
                response = json.loads(exchange.response_json)
                data = response["data"]
                if not isinstance(data, dict):
                    problems.append(f"exchange {index}: malformed state envelope")
                    continue
                if (
                    response.get("type") != "state"
                    or data.get("episode_id") != episode_id
                ):
                    problems.append(f"exchange {index}: episode_id differs from reset")
                if (
                    type(data.get("step_count")) is not int
                    or data["step_count"] != steps
                ):
                    problems.append(f"exchange {index}: incorrect step_count")
        if states == 0:
            problems.append("no state snapshots were measured")
        return problems
