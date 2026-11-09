"""Bounded fresh-session and independently verified fresh-container experiments."""

import json
import secrets
import time
import uuid
from dataclasses import replace

from ..types import ProviderCapability
from .collector import collect_runtime_evidence, RuntimeCollectionInterrupted
from .contracts import ReplayEvidence, RuntimeEvidence

REPLAY_BUDGET_SECONDS = 300.0
MAX_REPLAY_BYTES = 32 * 1024 * 1024
JUDGED_REPLAYS = 20


def _evidence_bytes(evidence):
    return sum(
        len(value.encode("utf-8"))
        for value in (
            evidence.observation_schema_json or "",
            evidence.reset_observation_schema_json or "",
            evidence.telemetry_json or "",
            *(
                value
                for row in evidence.exchanges
                for value in (row.request_json, row.response_json)
            ),
        )
    )


def collect_replays(
    provider, running, spec, plan, evidence, *, capabilities, deadline=None
):
    """
    Collect the fixed replay schedule without replacing the primary transcript.

    Args:
        provider (`ValidationProvider`):
            Provider used for the original subject and optional fresh container.
        running (`RunningSubject`):
            The original subject; its lifetime remains owned by the caller.
        spec (`LaunchSpec`):
            Original immutable image, resource limits and explicit authorization.
        plan (`RuntimePlan`):
            Fixed episode identity, seed, options and public action sequence.
        evidence (`RuntimeEvidence`):
            Independently collected primary episode.
        capabilities (`CapabilitiesSpec`):
            Manifest capabilities; judged rewards require 20 identical-input samples.
        deadline (`float`, *optional*):
            Absolute monotonic deadline set before primary collection. Defaults to
            300 seconds from this call; every collection and launch shares it.

    Returns:
        `RuntimeEvidence`: the original episode plus immutable replay evidence.
    """
    retained_bytes = _evidence_bytes(evidence)
    budget_failure = f"total retained replay evidence exceeds {MAX_REPLAY_BYTES} bytes"
    if retained_bytes > MAX_REPLAY_BYTES:
        # Normal collector bounds keep a primary episode below this limit. A
        # malformed oversized input must not bypass the final artifact budget.
        return RuntimeEvidence(
            failure_phase="replay evidence budget",
            failure_reason=budget_failure,
            replay_failure_reason=budget_failure,
        )
    if evidence.failure_reason:
        return evidence
    deadline = min(
        deadline if deadline is not None else float("inf"),
        time.monotonic() + REPLAY_BUDGET_SECONDS,
    )
    replays = []
    phase = "fresh session"
    failure = None

    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError("total replay deadline exceeded")
        return budget

    def collect(target, reset_plan, token):
        return collect_runtime_evidence(
            target.base_url,
            reset_plan,
            episode_timeout_s=min(spec.resources.episode_timeout_s, remaining()),
            validation_token=token,
        )

    def retain(scope, sample, **metadata):
        nonlocal retained_bytes, failure
        size = _evidence_bytes(sample)
        if retained_bytes + size > MAX_REPLAY_BYTES:
            failure = budget_failure
            if scope == "container":
                # Keep inspected identity and cleanup even when the transcript
                # cannot be retained. An empty sample cannot pass determinism.
                replays.append(ReplayEvidence(scope, RuntimeEvidence(), **metadata))
            return False
        retained_bytes += size
        replays.append(ReplayEvidence(scope, sample, **metadata))
        return True

    def session(scope, reset_plan):
        try:
            sample = collect(
                running, reset_plan, spec.env_vars.get("OPENENV_VALIDATION_TOKEN")
            )
        except RuntimeCollectionInterrupted as exc:
            retain(scope, exc.evidence)
            raise
        if not retain(scope, sample):
            raise ValueError("replay evidence byte budget exceeded")
        if sample.failure_reason:
            raise RuntimeError("replay collection failed")

    try:
        session("session", plan)
        phase = "different-seed session"
        changed_seed = plan.reset.model_copy(
            update={"seed": (plan.reset.seed + 1) % 2**32}
        )
        session("seed", plan.model_copy(update={"reset": changed_seed}))
        if ProviderCapability.FRESH_CONTAINER not in provider.capabilities:
            failure = "missing provider capability: fresh_container"
        else:
            phase = "original container inspection"
            remaining()
            original = running.inspect()
            if (
                not isinstance(original.get("container_id"), str)
                or not original["container_id"]
                or original.get("image_id") != spec.image_ref
            ):
                raise ValueError("original container identity unavailable")
            phase = "fresh container start"
            fresh_spec = spec.model_copy(
                update={
                    "run_id": "validation-replay-" + uuid.uuid4().hex,
                    "startup_timeout_s": min(spec.startup_timeout_s, remaining()),
                    "env_vars": {
                        **spec.env_vars,
                        "OPENENV_VALIDATION_TOKEN": secrets.token_urlsafe(32),
                    },
                }
            )
            try:
                fresh = provider.start(fresh_spec)
            except (Exception, KeyboardInterrupt):
                # No returned handle means this layer cannot verify teardown,
                # including when the provider's startup cleanup itself failed.
                retain(
                    "container",
                    RuntimeEvidence(
                        failure_phase=phase,
                        failure_reason="fresh container start failed; teardown could not be confirmed",
                    ),
                    cleanup_complete=False,
                )
                raise
            inspection_json = None
            sample = None
            cleanup = False
            try:
                phase = "fresh container inspection"
                remaining()
                inspection = fresh.inspect()
                inspection_json = json.dumps(inspection, allow_nan=False)
                if (
                    not isinstance(inspection.get("container_id"), str)
                    or not inspection["container_id"]
                    or inspection["container_id"] == original["container_id"]
                    or inspection.get("image_id") != spec.image_ref
                ):
                    raise ValueError("fresh container identity did not match")
                phase = "fresh container replay"
                sample = collect(
                    fresh, plan, fresh_spec.env_vars["OPENENV_VALIDATION_TOKEN"]
                )
            except RuntimeCollectionInterrupted as exc:
                sample = exc.evidence
                raise
            finally:
                cleanup_interrupted = False
                try:
                    fresh.stop()
                    cleanup = True
                except KeyboardInterrupt:
                    cleanup_interrupted = True
                    failure = "fresh container cleanup interrupted"
                except Exception:
                    failure = "fresh container cleanup failed"
                if sample is None:
                    sample = RuntimeEvidence(
                        failure_phase=phase, failure_reason=f"{phase} did not complete"
                    )
                retained = retain(
                    "container",
                    sample,
                    provider_json=inspection_json,
                    cleanup_complete=cleanup,
                )
                if cleanup_interrupted:
                    raise KeyboardInterrupt
            if not cleanup:
                raise RuntimeError("fresh container cleanup failed")
            if not retained:
                raise ValueError("replay evidence byte budget exceeded")
            if sample.failure_reason:
                raise RuntimeError("fresh container replay failed")
            phase = "judged replay sampling"
            if capabilities.llm_judged:
                # Primary + session + container are already three identical inputs.
                for _ in range(JUDGED_REPLAYS - 3):
                    session("session", plan)
    except KeyboardInterrupt:
        interrupted = replace(
            evidence,
            replays=tuple(replays),
            replay_failure_reason=failure or f"{phase} interrupted",
        )
        raise RuntimeCollectionInterrupted(interrupted) from None
    except Exception as exc:
        failure = failure or f"{phase} failed ({type(exc).__name__})"
    return replace(evidence, replays=tuple(replays), replay_failure_reason=failure)
