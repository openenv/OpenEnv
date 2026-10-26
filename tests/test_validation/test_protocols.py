from pathlib import Path

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders import (
    ENTRY_POINT_GROUP,
    Grader,
    GraderRegistry,
    Subject,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.parsers import Parser, ParserRegistry
from openenv.validation.providers import ExecResult, RunningSubject, ValidationProvider
from openenv.validation.report import CheckResult
from openenv.validation.signature import UnsupportedPackageError
from openenv.validation.types import (
    CheckStatus,
    Level,
    ProviderCapability,
    SignatureKind,
)


class FakeParser:
    signature = SignatureKind.OPENENV_SERVED

    def parse(self, package_root: Path) -> NormalizedManifest:
        return NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        )


class FakeGrader:
    def __init__(
        self,
        check_id="static.manifest",
        level=Level.STATIC,
        requires_capabilities=frozenset(),
        applies=True,
    ):
        self.check_id = check_id
        self.level = level
        self.requires_capabilities = requires_capabilities
        self.requires_provider = frozenset()
        self.depends_on = ()
        self._applies = applies

    def applies_to(self, manifest: NormalizedManifest) -> bool:
        return self._applies

    def run(self, subject: Subject) -> CheckResult:
        return CheckResult(
            check_id=self.check_id, status=CheckStatus.PASS, duration_s=0.0
        )


class FakeEntryPoints(list):
    def select(self, **params):
        assert params == {"group": ENTRY_POINT_GROUP}
        return self


class FakeRunningSubject:
    base_url = "http://localhost:8000"

    def inspect(self):
        return {}

    def logs(self, max_bytes=65536):
        return ""

    def exec(self, argv: list, timeout_s: float) -> ExecResult:
        return ExecResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def stop(self) -> None:
        pass


class FakeProvider:
    name = "fake"
    supported_network_modes = frozenset({"public"})
    capabilities = frozenset(
        {ProviderCapability.NETWORK_POLICY, ProviderCapability.EXEC}
    )

    def build(self, root, execution):
        return "sha256:" + "a" * 64

    def start(self, spec):
        return FakeRunningSubject()


@pytest.fixture()
def manifest():
    return NormalizedManifest.model_validate(load_fixture_manifest("served_min_pass"))


def test_fakes_conform_to_the_protocols():
    assert isinstance(FakeParser(), Parser)
    assert isinstance(FakeGrader(), Grader)
    assert isinstance(FakeRunningSubject(), RunningSubject)
    assert isinstance(FakeProvider(), ValidationProvider)


def test_parser_registry_dispatches_by_signature():
    registry = ParserRegistry()
    parser = FakeParser()
    registry.register(parser)
    assert registry.parser_for(SignatureKind.OPENENV_SERVED) is parser


def test_unregistered_signature_raises_parser_not_implemented():
    registry = ParserRegistry()
    registry.register(FakeParser())
    with pytest.raises(UnsupportedPackageError) as exc_info:
        registry.parser_for(SignatureKind.POSTTRAIN_TASK)
    assert exc_info.value.category == "parser-not-implemented"


def test_grader_registry_rejects_duplicate_check_ids():
    registry = GraderRegistry()
    registry.register(FakeGrader())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeGrader())


def test_selection_respects_the_level_ceiling(manifest):
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="static.manifest", level=Level.STATIC))
    registry.register(FakeGrader(check_id="semantic.oracle_max", level=Level.SEMANTIC))
    selected = registry.select(manifest, max_level=Level.STATIC)
    assert [g.check_id for g in selected] == ["static.manifest"]


def test_selection_requires_truthy_capabilities(manifest):
    registry = GraderRegistry()
    registry.register(
        FakeGrader(
            check_id="runtime.task_declaration_accuracy",
            level=Level.RUNTIME,
            requires_capabilities=frozenset({"task_api"}),
        )
    )
    assert manifest.capabilities.task_api is False
    assert registry.select(manifest, max_level=Level.SEMANTIC) == []


def test_selection_respects_applies_to(manifest):
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="static.manifest", applies=False))
    assert registry.select(manifest, max_level=Level.SEMANTIC) == []


def test_selection_is_ordered_by_level_then_check_id(manifest):
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="semantic.oracle_max", level=Level.SEMANTIC))
    registry.register(FakeGrader(check_id="static.manifest", level=Level.STATIC))
    registry.register(FakeGrader(check_id="static.layout", level=Level.STATIC))
    selected = registry.select(manifest, max_level=Level.SEMANTIC)
    assert [g.check_id for g in selected] == [
        "static.layout",
        "static.manifest",
        "semantic.oracle_max",
    ]


def test_selection_never_sees_the_signature(manifest):
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="static.manifest", level=Level.STATIC))
    as_harbor = manifest.model_copy(update={"signature": SignatureKind.HARBOR_TASK})
    served_ids = [
        g.check_id for g in registry.select(manifest, max_level=Level.SEMANTIC)
    ]
    harbor_ids = [
        g.check_id for g in registry.select(as_harbor, max_level=Level.SEMANTIC)
    ]
    assert served_ids == harbor_ids


def test_subject_is_frozen(manifest, tmp_path):
    subject = Subject(
        root=tmp_path,
        manifest=manifest,
        image_ref=None,
        running=None,
        outputs_dir=tmp_path / "outputs",
    )
    with pytest.raises(Exception):
        subject.image_ref = "other"  # type: ignore[misc]


def test_provider_start_returns_a_running_subject():
    running = FakeProvider().start("img:latest")
    assert isinstance(running, RunningSubject)
    assert running.exec(["true"], timeout_s=1.0).exit_code == 0


def test_register_rejects_unknown_capability_names():
    registry = GraderRegistry()
    with pytest.raises(ValueError, match="telepathy"):
        registry.register(FakeGrader(requires_capabilities=frozenset({"telepathy"})))


def test_entry_point_grader_classes_are_instantiated(manifest, monkeypatch):
    class FakeEntryPoint:
        name = "fake-class"

        def load(self):
            return FakeGrader

    import openenv.validation.graders as graders_module

    monkeypatch.setattr(
        graders_module, "entry_points", lambda: FakeEntryPoints([FakeEntryPoint()])
    )
    registry = GraderRegistry()
    assert registry.load_entry_points() == 1
    (grader,) = registry.select(manifest, max_level=Level.SEMANTIC)
    assert grader.run(None).status is CheckStatus.PASS


def test_entry_point_grader_instances_register_as_is(manifest, monkeypatch):
    instance = FakeGrader()

    class FakeEntryPoint:
        name = "fake-instance"

        def load(self):
            return instance

    import openenv.validation.graders as graders_module

    monkeypatch.setattr(
        graders_module, "entry_points", lambda: FakeEntryPoints([FakeEntryPoint()])
    )
    registry = GraderRegistry()
    assert registry.load_entry_points() == 1
    assert registry.select(manifest, max_level=Level.SEMANTIC) == [instance]


def test_entry_point_loading_supports_the_legacy_mapping_api(manifest, monkeypatch):
    instance = FakeGrader()

    class FakeEntryPoint:
        def load(self):
            return instance

    import openenv.validation.graders as graders_module

    monkeypatch.setattr(
        graders_module,
        "entry_points",
        lambda: {ENTRY_POINT_GROUP: [FakeEntryPoint()]},
    )
    registry = GraderRegistry()
    assert registry.load_entry_points() == 1
    assert registry.select(manifest, max_level=Level.SEMANTIC) == [instance]
