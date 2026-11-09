"""Pure Harbor task parsing and the explicit, operator-supplied pilot contract."""

import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..manifest import (
    ManifestError,
    NormalizedManifestV2,
    ResourceDeclaration,
    RewardDeclaration,
    TypeSpec,
)
from ..types import SignatureKind

HARBOR_PROFILE = "harbor-oracle-floor-v1"


class HarborContract(BaseModel):
    """Trusted declarations absent from Harbor's task format; never inferred."""

    model_config = ConfigDict(extra="forbid")

    profile: Literal["harbor-oracle-floor-v1"]
    reward_key: str = Field(min_length=1)
    reward: RewardDeclaration
    resources: ResourceDeclaration
    types: TypeSpec
    original_source_digest: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    requires_credentials: bool = Field(default=False, strict=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def read_harbor_contract(package_root: Path) -> HarborContract:
    """Read `validation/openenv.json` without importing or executing task code."""
    try:
        raw = json.loads(
            (package_root / "validation/openenv.json").read_text(),
            object_pairs_hook=_unique_object,
        )
        if not isinstance(raw, dict):
            raise ValueError("contract must be an object")
        if (
            not {"range", "oracle_tolerance", "floor_margin"}
            <= raw.get("reward", {}).keys()
        ):
            raise ValueError(
                "reward range, oracle_tolerance and floor_margin must be explicit"
            )
        for section in ("reward", "resources"):
            pending = [raw.get(section)]
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, bool) or (
                    isinstance(item, float) and not math.isfinite(item)
                ):
                    raise ValueError(
                        "numeric declarations must be finite and nonboolean"
                    )
        return HarborContract.model_validate(raw)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ManifestError(
            [f"invalid trusted Harbor contract ({type(exc).__name__})"],
            remediation="supply explicit reward/resources in validation/openenv.json",
        ) from exc


def read_harbor_task(package_root: Path):
    """Use the pinned Harbor data loader; constructing a Trial is runtime-only."""
    try:
        if version("harbor") != "0.25.0":
            raise ValueError("the native pilot requires Harbor 0.25.0")
        from harbor.models.task.task import Task

        return Task(package_root)
    except (
        ImportError,
        PackageNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise ManifestError(
            [f"invalid Harbor 0.25.0 task ({type(exc).__name__})"],
            remediation="install Harbor 0.25.0 and supply a valid task.toml/task layout",
        ) from exc


class HarborTaskParser:
    """Normalize declarations only; no environment, Trial, build or subprocess."""

    signature = SignatureKind.HARBOR_TASK

    def parse(self, package_root: Path) -> NormalizedManifestV2:
        contract = read_harbor_contract(package_root)
        task = read_harbor_task(package_root)
        oracle = (
            {"form": "script", "location": "solution/solve.sh"}
            if (package_root / "solution/solve.sh").is_file()
            else None
        )
        try:
            return NormalizedManifestV2.model_validate(
                {
                    "manifest_schema_version": "2",
                    "signature": self.signature,
                    "name": task.name,
                    "version": task.config.task.version if task.config.task else None,
                    "reward": contract.reward.model_dump(),
                    "resources": contract.resources.model_dump(),
                    "types": contract.types.model_dump(),
                    "network": {"mode": task.config.environment.network_mode.value},
                    "capabilities": {
                        "verifier": {"kind": "script", "entry": "tests/test.sh"},
                        "oracle": oracle,
                    },
                    "execution": {
                        "kind": "harbor_task",
                        "agent_boundary": "process",
                        "profile": contract.profile,
                        "requires_credentials": contract.requires_credentials,
                    },
                }
            )
        except ValidationError as exc:
            raise ManifestError(
                ["Harbor declarations do not match the manifest schema"]
            ) from exc
