"""Pure Harbor task parsing and the explicit, operator-supplied pilot contract."""

import json
import math
import stat
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError

from ..manifest import (
    ManifestError,
    NormalizedManifestV2,
    ResourceDeclaration,
    RewardDeclaration,
    TypeSpec,
)
from ..types import SignatureKind

HARBOR_PROFILE = "harbor-semantic-v1"
MAX_CANARIES = 8


class HarborContract(BaseModel):
    """Trusted declarations absent from Harbor's task format; never inferred."""

    model_config = ConfigDict(extra="forbid")

    profile: Literal["harbor-semantic-v1"]
    reward_key: str = Field(min_length=1)
    reward: RewardDeclaration
    resources: ResourceDeclaration
    types: TypeSpec
    original_source_digest: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    requires_credentials: bool = Field(default=False, strict=True)
    # Directory of known-wrong `solve.sh` replacements for `semantic.canary_floor`.
    canaries: str | None = None

    @field_validator("canaries")
    @classmethod
    def _package_relative(cls, value):
        path = PurePosixPath(value)
        if (
            "\\" in value
            or ":" in value
            or "\x00" in value
            or path.is_absolute()
            or ".." in path.parts
            or path == PurePosixPath(".")
        ):
            raise ValueError("canaries must be a portable package-relative directory")
        return value


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


def canary_scripts(package_root: Path, directory: str) -> list[str]:
    """
    List the declared canary scripts without reading or running them.

    Args:
        package_root (`Path`):
            The task directory.
        directory (`str`):
            Package-relative directory holding one `*.sh` file per canary.

    Returns:
        `list[str]`: package-relative script paths, sorted by name.
    """
    root = package_root / directory
    if root.is_symlink() or not root.is_dir():
        raise ValueError("the canaries directory does not exist")
    scripts = []
    for path in sorted(root.iterdir()):
        info = path.lstat()
        if path.suffix != ".sh" or not stat.S_ISREG(info.st_mode):
            raise ValueError("the canaries directory may hold only regular *.sh files")
        scripts.append((PurePosixPath(directory) / path.name).as_posix())
    if not 1 <= len(scripts) <= MAX_CANARIES:
        raise ValueError(f"declare between 1 and {MAX_CANARIES} canaries")
    return scripts


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
        if contract.canaries is not None:
            try:
                canary_scripts(package_root, contract.canaries)
            except (OSError, ValueError) as exc:
                raise ManifestError(
                    [f"invalid canaries declaration ({exc})"],
                    remediation="declare a directory of 1 to 8 *.sh canary scripts",
                ) from exc
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
                        "canaries": contract.canaries,
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
