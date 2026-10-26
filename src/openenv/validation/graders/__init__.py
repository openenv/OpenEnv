"""Grader protocol and registry.

Selection reads the manifest only — the signature never reaches
[`~openenv.validation.graders.GraderRegistry.select`]. Write a grader once and it works
with every format whose parser supplies the manifest fields.
"""

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..manifest import CapabilitiesSpec, NormalizedManifest
from ..providers import RunningSubject
from ..report import CheckResult
from ..runtime.contracts import RuntimeEvidence
from ..types import Level, ProviderCapability

ENTRY_POINT_GROUP = "openenv.validation.graders"
"""Third-party graders register under this entry-point group — zero core changes."""


@dataclass(frozen=True)
class Subject:
    """
    The one argument every grader receives.

    Attributes:
        root (`Path`):
            Package source tree.
        manifest ([`~openenv.validation.manifest.NormalizedManifest`]):
            The parsed, normalized manifest.
        image_ref (`str`, *optional*):
            Built image reference; `None` under `--skip-build`.
        running ([`~openenv.validation.providers.RunningSubject`], *optional*):
            The started sandbox; `None` at the static level.
        outputs_dir (`Path`):
            Where trajectory records and replay artifacts are written.
        network_evidence_json (`str`, *optional*):
            Provider-measured [`~openenv.validation.runtime.contracts.NetworkEvidence`].
    """

    root: Path
    manifest: NormalizedManifest
    image_ref: str | None
    running: RunningSubject | None
    outputs_dir: Path
    runtime_evidence: RuntimeEvidence | None = None
    network_evidence_json: str | None = None


@runtime_checkable
class Grader(Protocol):
    """
    One check. Emits a [`~openenv.validation.report.CheckResult`] with a
    [`~openenv.validation.types.CheckStatus`]; never a severity.

    Attributes:
        check_id (`str`):
            Stable, policy-addressed id, e.g. `"semantic.oracle_max"`.
        level ([`~openenv.validation.types.Level`]):
            The validation level this check belongs to.
        requires_capabilities (`frozenset[str]`):
            `CapabilitiesSpec` field names that must be truthy for this grader to apply.
        requires_provider (`frozenset` of [`~openenv.validation.types.ProviderCapability`]):
            Provider capabilities this grader needs; missing ones SKIP the check.
        depends_on (`tuple[str, ...]`):
            Check ids; any non-PASS dependency SKIPs this check with the dependency
            named (this encodes "L2 determinism is a precondition of L3 semantics").
    """

    check_id: str
    level: Level
    requires_capabilities: frozenset[str]
    requires_provider: frozenset[ProviderCapability]
    depends_on: tuple[str, ...]

    def applies_to(self, manifest: NormalizedManifest) -> bool: ...

    def run(self, subject: Subject) -> CheckResult: ...


class GraderRegistry:
    """
    Holds all known graders and selects the applicable subset per run.

    Core graders are registered directly; third parties register via the
    `"openenv.validation.graders"` entry-point group.
    """

    def __init__(self) -> None:
        self._graders: dict[str, Grader] = {}

    def register(self, grader: Grader) -> None:
        """
        Register a grader by its check id.

        Args:
            grader ([`~openenv.validation.graders.Grader`]):
                The grader to register. Its `requires_capabilities` must name real
                `CapabilitiesSpec` fields — a typo'd capability is rejected here
                rather than crashing (or silently deselecting) at selection time.
        """
        if grader.check_id in self._graders:
            raise ValueError(
                f"grader already registered for check id {grader.check_id!r}"
            )
        unknown = set(grader.requires_capabilities) - set(CapabilitiesSpec.model_fields)
        if unknown:
            raise ValueError(
                f"grader {grader.check_id!r} requires unknown capabilities: "
                f"{sorted(unknown)}; valid names are CapabilitiesSpec fields"
            )
        self._graders[grader.check_id] = grader

    def load_entry_points(self) -> int:
        """
        Register third-party graders from the entry-point group.

        An entry point may name a grader instance or a zero-argument grader class;
        classes are instantiated. (An `isinstance(..., Grader)` test cannot make the
        distinction — a grader class with class-level attributes already satisfies
        the runtime-checkable protocol.)

        Returns:
            `int`: the number of graders loaded.
        """
        loaded = 0
        discovered = entry_points()
        if hasattr(discovered, "select"):
            selected = discovered.select(group=ENTRY_POINT_GROUP)
        else:
            selected = discovered.get(ENTRY_POINT_GROUP, ())

        for ep in selected:
            candidate = ep.load()
            grader = candidate() if isinstance(candidate, type) else candidate
            self.register(grader)
            loaded += 1
        return loaded

    def get(self, check_id: str) -> Grader | None:
        """Return the registered implementation, or `None` for a reserved check."""
        return self._graders.get(check_id)

    def select(self, manifest: NormalizedManifest, max_level: Level) -> list[Grader]:
        """
        Select the graders that apply to a manifest, up to a level ceiling.

        Selection reads the manifest only: capabilities select contract graders, type
        tags select domain graders. The signature never reaches this method's inputs.

        Args:
            manifest ([`~openenv.validation.manifest.NormalizedManifest`]):
                The parsed manifest.
            max_level ([`~openenv.validation.types.Level`]):
                Ceiling; graders above it are not selected.

        Returns:
            `list` of [`~openenv.validation.graders.Grader`], ordered by (level, check id).
        """
        selected = []
        for grader in self._graders.values():
            if grader.level > max_level:
                continue
            if not all(
                getattr(manifest.capabilities, field)
                for field in grader.requires_capabilities
            ):
                continue
            if not grader.applies_to(manifest):
                continue
            selected.append(grader)
        return sorted(selected, key=lambda g: (g.level, g.check_id))
