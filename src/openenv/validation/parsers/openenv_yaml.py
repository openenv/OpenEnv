"""Parser for the served OpenEnv format (`openenv.yaml`)."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from ..manifest import ManifestError, NormalizedManifest, NormalizedManifestV2
from ..types import SignatureKind

VALIDATION_BLOCK_REMEDIATION = (
    "add a `validation:` block to openenv.yaml declaring reward (range, "
    "oracle_tolerance, floor_margin), resources (cpu, memory_mb, disk_mb, "
    "episode_timeout_s), capabilities (verifier, oracle, ...), and types (tags); "
    "see the manifest schema at src/openenv/validation/schemas/manifest.schema.json"
)

SCHEMA_REMEDIATION = (
    "fix the `validation:` block in openenv.yaml so it matches the normalized "
    "manifest schema at src/openenv/validation/schemas/manifest.schema.json"
)


def _format_validation_error(exc: ValidationError) -> list[str]:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "manifest"
        lines.append(f"{loc}: {err['msg']}")
    return lines


class OpenEnvYamlParser:
    """
    Parses `openenv.yaml` into the normalized manifest.

    A parse is a pure read: it never imports or executes package code. Top-level
    `name`/`version` identify the environment; everything validation-specific lives
    under the `validation:` block, which maps onto
    [`~openenv.validation.manifest.NormalizedManifest`] sections.
    """

    signature = SignatureKind.OPENENV_SERVED

    def parse(self, package_root: Path) -> NormalizedManifest | NormalizedManifestV2:
        """
        Parse a served-environment package.

        Args:
            package_root (`Path`):
                Directory containing `openenv.yaml`.

        Returns:
            [`~openenv.validation.manifest.NormalizedManifest`]: the normalized manifest.
        """
        source = Path(package_root) / "openenv.yaml"
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ManifestError([f"openenv.yaml is not valid YAML: {exc}"]) from exc
        if not isinstance(raw, dict):
            raise ManifestError(["openenv.yaml must be a YAML mapping"])

        validation = raw.get("validation")
        if validation is None:
            raise ManifestError(
                ["openenv.yaml has no `validation:` block"],
                remediation=VALIDATION_BLOCK_REMEDIATION,
            )
        if not isinstance(validation, dict):
            raise ManifestError(["`validation:` must be a mapping"])

        has_execution = "execution" in validation
        data: dict = {
            "manifest_schema_version": "2" if has_execution else "1",
            "signature": SignatureKind.OPENENV_SERVED,
            "version": raw.get("version"),
            "judge": validation.get("judge"),
            "task_distribution": validation.get("task_distribution"),
        }
        for key, value in (
            ("name", raw.get("name")),
            ("reward", validation.get("reward")),
            ("resources", validation.get("resources")),
            ("capabilities", validation.get("capabilities")),
            ("types", validation.get("types")),
            ("network", validation.get("network")),
        ):
            if value is not None:
                data[key] = value

        if has_execution:
            data["execution"] = validation["execution"]

        try:
            model = NormalizedManifestV2 if has_execution else NormalizedManifest
            return model.model_validate(data)
        except ValidationError as exc:
            raise ManifestError(
                _format_validation_error(exc),
                remediation=SCHEMA_REMEDIATION,
            ) from exc
