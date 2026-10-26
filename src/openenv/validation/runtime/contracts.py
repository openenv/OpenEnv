"""Versioned runtime inputs and execution evidence, independent of a provider."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from ..manifest import ExecutionDeclaration, NetworkPolicy, ResourceDeclaration

MAX_PLAN_BYTES = 65_536
MAX_PLAN_ACTIONS = 100
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 10_000


class RuntimePlanError(ValueError):
    """A public runtime plan cannot be read safely or violates its data contract."""


def _check_json(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> None:
    """Reject executable objects, non-finite numbers and excessive nesting."""
    if budget is None:
        budget = [MAX_JSON_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_JSON_DEPTH:
        raise ValueError("runtime JSON exceeds the node or nesting limit")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("runtime JSON numbers must be finite")
        return
    if type(value) is str:
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("runtime JSON object keys must be strings")
        children = value.values()
    elif type(value) is list:
        children = value
    else:
        raise ValueError("runtime inputs must contain only JSON data")
    for child in children:
        _check_json(child, depth=depth + 1, budget=budget)


class RuntimeReset(BaseModel):
    """
    Explicit reset inputs for one measured episode.

    Attributes:
        episode_id (`str`):
            Requested episode identity; verified by the state contract grader.
        seed (`int`):
            Requested seed. Acceptance alone does not establish determinism.
        options (`dict[str, Any]`, *optional*):
            Additional public reset arguments, never replacements for the seed or ID.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    episode_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=2**32 - 1)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def _public_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        if {"episode_id", "seed"} & value.keys():
            raise ValueError("reset options cannot override episode_id or seed")
        _check_json(value)
        return value


class RuntimePlan(BaseModel):
    """
    Bounded, data-only runtime inputs. Capability declarations stay in the manifest.

    Attributes:
        plan_schema_version (`str`):
            The pinned sidecar schema version, `"1"`.
        reset ([`~openenv.validation.runtime.contracts.RuntimeReset`]):
            Inputs to the first reset on the measured WebSocket session.
        actions (`list[dict[str, Any]]`):
            Between one and 100 public actions, applied in order until termination.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    plan_schema_version: Literal["1"]
    reset: RuntimeReset
    actions: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_PLAN_ACTIONS)

    @field_validator("actions")
    @classmethod
    def _data_actions(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _check_json(value)
        return value

    @model_validator(mode="after")
    def _total_size(self) -> "RuntimePlan":
        size = len(self.model_dump_json().encode("utf-8"))
        if size > MAX_PLAN_BYTES:
            raise ValueError(f"runtime plan exceeds {MAX_PLAN_BYTES} bytes")
        return self


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate runtime JSON key")
        result[key] = value
    return result


def load_runtime_plan(root: Path, execution: ExecutionDeclaration) -> RuntimePlan:
    """
    Read a bounded runtime plan without importing any subject code.

    Args:
        root (`Path`):
            Package root; the resolved plan must stay inside this directory.
        execution ([`~openenv.validation.manifest.ExecutionDeclaration`]):
            Manifest-owned location of the JSON sidecar.

    Returns:
        [`~openenv.validation.runtime.contracts.RuntimePlan`]: validated public inputs.

    Raises:
        [`~openenv.validation.runtime.contracts.RuntimePlanError`]:
            Missing, escaped, oversized, ambiguous or invalid input.
    """
    try:
        package_root = Path(root).resolve(strict=True)
        path = (package_root / execution.probe_path).resolve(strict=True)
        if not path.is_relative_to(package_root) or not path.is_file():
            raise ValueError("runtime plan must be a regular file inside the package")
        with path.open("rb") as source:
            payload = source.read(MAX_PLAN_BYTES + 1)
        if len(payload) > MAX_PLAN_BYTES:
            raise ValueError(f"runtime plan exceeds {MAX_PLAN_BYTES} bytes")
        raw = json.loads(payload, object_pairs_hook=_unique_object)
        _check_json(raw)
        return RuntimePlan.model_validate(raw)
    except ValidationError as exc:
        fields = RuntimePlan.model_fields.keys() | RuntimeReset.model_fields.keys()
        errors = []
        for error in exc.errors(
            include_input=False, include_context=False, include_url=False
        )[:5]:
            location = ".".join(
                str(part) if isinstance(part, int) or part in fields else "<field>"
                for part in error["loc"]
            )
            # These messages come from the fixed schema and validators; raw inputs,
            # exception context and subject-controlled field names are omitted.
            errors.append(f"{location or '<root>'}: {error['type']} ({error['msg']})")
        if exc.error_count() > 5:
            errors.append(f"... ({exc.error_count()} errors total)")
        raise RuntimePlanError(
            "runtime plan schema validation failed: " + "; ".join(errors)
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimePlanError(
            f"invalid runtime plan JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except UnicodeError as exc:
        raise RuntimePlanError("runtime plan contains invalid text encoding") from exc
    except OSError as exc:
        raise RuntimePlanError(
            f"runtime plan could not be read ({type(exc).__name__})"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise RuntimePlanError(f"invalid runtime plan: {exc}") from exc


class LaunchSpec(BaseModel):
    """
    Explicit validation-only launch request; no caller environment is inherited.

    Attributes:
        image_ref (`str`):
            Immutable local image ID or repository digest.
        resources ([`~openenv.validation.manifest.ResourceDeclaration`]):
            Subject CPU, memory, writable-storage and episode budgets.
        network ([`~openenv.validation.manifest.NetworkPolicy`]):
            Requested policy; unsupported enforcement must be refused.
        run_id (`str`):
            Unique run-owned resource label, safe for provider names.
        startup_timeout_s (`float`, *optional*, defaults to 30):
            Independent readiness deadline, at most 300 seconds.
        env_vars (`dict[str, str]`, *optional*):
            Explicit environment only; providers do not forward host credentials.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_ref: str = Field(pattern=r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
    resources: ResourceDeclaration
    network: NetworkPolicy
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    startup_timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    env_vars: dict[str, str] = Field(default_factory=dict, max_length=64)

    @field_validator("env_vars")
    @classmethod
    def _explicit_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for name, content in value.items():
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError("invalid environment variable name")
            if "\x00" in content or len(content.encode("utf-8")) > 8192:
                raise ValueError("environment value contains NUL or exceeds 8192 bytes")
        return value

    @field_validator("resources")
    @classmethod
    def _finite_resources(cls, value: ResourceDeclaration) -> ResourceDeclaration:
        if not math.isfinite(value.cpu) or not math.isfinite(value.episode_timeout_s):
            raise ValueError("runtime resource limits must be finite")
        return value


@dataclass(frozen=True)
class WireExchange:
    """
    One immutable raw exchange, captured before client defaults or type coercion.

    Attributes:
        operation (`str`):
            The reset, step or state operation performed on the measured session.
        request_json (`str`):
            Original serialized request; parse only when grading.
        response_json (`str`):
            Original serialized response, including malformed JSON for diagnostics.
    """

    operation: Literal["reset", "step", "state"]
    request_json: str
    response_json: str


@dataclass(frozen=True)
class RuntimeEvidence:
    """
    Collector evidence shared by graders without permitting mutation of the episode.

    Attributes:
        exchanges (`tuple[WireExchange, ...]`):
            Ordered operations on one WebSocket session.
        observation_schema_json (`str`, *optional*):
            The advertised observation schema, exactly `/schema`'s observation value.
        failure_phase (`str`, *optional*):
            Collector phase that failed; a truncated transcript cannot pass silently.
        failure_reason (`str`, *optional*):
            Credential-safe explanation of the collection failure.
        reset_observation_schema_json (`str`, *optional*):
            Explicit `/schema` reset_observation value; absent means use the step schema.
    """

    exchanges: tuple[WireExchange, ...] = ()
    observation_schema_json: str | None = None
    failure_phase: str | None = None
    failure_reason: str | None = None
    reset_observation_schema_json: str | None = None


class ProbeOutcome(BaseModel):
    """
    One connection attempt, classified without trusting its error text.

    Attributes:
        result (`str`):
            `"reachable"`, `"denied"` (a network-level refusal such as ENETUNREACH),
            or `"inconclusive"` (timeouts, permission errors and anything else).
        errno (`int`, *optional*):
            The OS error number behind a denial or an inconclusive attempt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: Literal["reachable", "denied", "inconclusive"]
    errno: int | None = None


class NetworkProbe(BaseModel):
    """
    The same probe run from a control namespace and from the subject's namespace.

    Attributes:
        kind (`str`):
            `"tcp"`, `"udp"` or `"icmp"` towards a validator-owned sink.
        control ([`~openenv.validation.runtime.contracts.ProbeOutcome`]):
            Outcome from a validator-owned container with ordinary networking.
        subject ([`~openenv.validation.runtime.contracts.ProbeOutcome`]):
            Outcome from the subject's network namespace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tcp", "udp", "icmp"]
    control: ProbeOutcome
    subject: ProbeOutcome


class NamespaceEvidence(BaseModel):
    """
    Effective interfaces and routes observed inside the subject's network namespace.

    Attributes:
        interfaces (`list[str]`):
            Interface names in the namespace.
        default_route (`bool`):
            Whether an IPv4 default route exists.
        ipv6_non_loopback (`int`):
            IPv6 addresses on interfaces other than `lo`.
        listening_ports (`list[int]`):
            TCP ports listening in the namespace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    interfaces: list[str]
    default_route: bool
    ipv6_non_loopback: int = Field(ge=0)
    listening_ports: list[int] = Field(default_factory=list)


class NetworkEvidence(BaseModel):
    """
    Provider-neutral evidence for `runtime.network_policy`.

    Attributes:
        schema_version (`int`):
            Evidence format version.
        requested_mode (`str`):
            The declared mode the subject was launched under.
        subject_network_mode (`str`):
            The engine's inspected network mode for the subject.
        namespace ([`~openenv.validation.runtime.contracts.NamespaceEvidence`]):
            What the subject's namespace contains.
        probes (`list[NetworkProbe]`):
            Controlled reachability measurements.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    requested_mode: Literal["public", "no-network"]
    subject_network_mode: str
    namespace: NamespaceEvidence
    probes: list[NetworkProbe]
