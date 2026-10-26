# SPDX-License-Identifier: BSD-3-Clause

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from openenv.discovery import (
    build_catalog,
    CatalogError,
    CatalogSnapshot,
    compare_catalogs,
    load_catalog,
    resolve_entry,
    search_catalog,
    write_catalog,
)
from openenv.discovery.metadata import DiscoveryDeclaration


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Catalog tests",
            "-c",
            "user.email=catalog@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(repository: Path) -> str:
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "Update fixture")
    return git(repository, "rev-parse", "HEAD")


def environment(repository: Path, name: str, description: str) -> Path:
    directory = repository / "envs" / name
    directory.mkdir(parents=True)
    (directory / "openenv.yaml").write_text(
        f"spec_version: 1\nname: {name}\nruntime: fastapi\napp: server.app:app\n"
    )
    (directory / "pyproject.toml").write_text(
        f'[project]\nname = "openenv-{name}"\n'
        f"description = {json.dumps(description)}\n"
        'dependencies = ["openenv>=0.3.1"]\n'
    )
    (directory / "__init__.py").write_text(
        "raise RuntimeError('candidate code must never be imported by discovery')\n"
    )
    return directory


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    environment(repository, "echo_env", "Echo messages to test client tool calls.")
    environment(repository, "chess_env", "Play chess against a configurable opponent.")
    (repository / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nlicense="BSD-3-Clause"\nlicense-files=["LICENSE"]\n'
    )
    (repository / "LICENSE").write_text("BSD 3-Clause License\n")
    commit(repository)
    return repository


def build(repository: Path):
    return build_catalog(
        repository,
        repository_uri="https://github.com/example/environments.git",
        publisher="example.org",
        namespace="openenv",
    )


@pytest.fixture(scope="module")
def maintained_inventory(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CatalogSnapshot, list[str]]:
    source_root = Path(__file__).resolve().parents[2]
    repository = tmp_path_factory.mktemp("maintained-discovery")
    git(repository, "init", "-q")

    def copy_metadata(relative: str) -> None:
        source = source_root / relative
        assert not source.is_symlink(), f"Metadata must not be a symlink: {relative}"
        if source.is_file():
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    for filename in ("pyproject.toml", "LICENSE"):
        copy_metadata(filename)
    paths = []
    for manifest in sorted((source_root / "envs").glob("*/openenv.yaml")):
        path = manifest.parent.relative_to(source_root).as_posix()
        paths.append(path)
        for filename in (
            "openenv.yaml",
            "pyproject.toml",
            "README.md",
            "discovery.json",
        ):
            copy_metadata(f"{path}/{filename}")
        declaration_path = source_root / path / "discovery.json"
        if declaration_path.is_file():
            declaration = DiscoveryDeclaration.model_validate_json(
                declaration_path.read_text()
            )
            if declaration.agent_tools is not None:
                copy_metadata(declaration.agent_tools.source)
        (repository / path / "__init__.py").write_text(
            "raise RuntimeError('discovery must not import candidate environments')\n"
        )
    commit(repository)
    return build(repository), paths


@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("envs/coding_env", "evaluate Python snippets and inspect standard error"),
        ("envs/browsergym_env", "fill forms and click page controls with a web agent"),
        ("envs/calendar_env", "schedule appointments and edit calendar events"),
        ("envs/chess_env", "practice legal UCI moves against an opponent"),
        (
            "envs/reasoning_gym_env",
            "single question episodes with dataset answer scoring",
        ),
    ],
)
def test_maintained_declarations_support_task_selection_without_imports(
    maintained_inventory: tuple[CatalogSnapshot, list[str]], path: str, query: str
) -> None:
    snapshot, paths = maintained_inventory
    assert snapshot.complete
    assert snapshot.inventory.paths == paths
    entry = next(item for item in snapshot.entries if item.data.source.path == path)
    assert 2 <= len(entry.representative_queries) <= 5
    assert (
        entry.metadata["provenance"]["description"]
        == f"{path}/discovery.json#description"
    )
    assert query not in entry.representative_queries
    matches = search_catalog(snapshot, query)
    assert matches
    assert matches[0].entry.identifier == entry.identifier
    assert resolve_entry(snapshot, entry.identifier).data == entry.data
    assert entry.data.source.revision == snapshot.source.revision


def test_calendar_declares_only_source_evidenced_event_tools() -> None:
    source_root = Path(__file__).resolve().parents[2]
    declaration_path = source_root / "envs/calendar_env/discovery.json"
    assert declaration_path.is_file()
    declaration = DiscoveryDeclaration.model_validate_json(declaration_path.read_text())
    assert declaration.agent_tools is not None
    assert declaration.agent_tools.protocol == "mcp"
    source = source_root / declaration.agent_tools.source
    definitions = ast.parse(source.read_text())
    assignment = next(
        node
        for node in definitions.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EVENTS_TOOLS"
            for target in node.targets
        )
    )
    tools = ast.literal_eval(assignment.value)
    assert set(declaration.agent_tools.names) <= {tool["name"] for tool in tools}
    assert not {"reset", "step", "state", "get_state"}.intersection(
        declaration.agent_tools.names
    )


def test_real_git_inventory_round_trips_and_selects_by_task(
    repository: Path, tmp_path: Path
) -> None:
    snapshot = build(repository)
    assert snapshot.complete
    assert snapshot.inventory.paths == ["envs/chess_env", "envs/echo_env"]
    assert len(snapshot.entries) == 2

    path = tmp_path / "catalog.json"
    write_catalog(snapshot, path)
    loaded = load_catalog(path)
    matches = search_catalog(loaded, "test client tool calls")
    assert matches[0].entry.data.name == "echo_env"
    entry = resolve_entry(loaded, matches[0].entry.identifier)
    assert entry.data.source.uri == "https://github.com/example/environments.git"
    assert entry.data.source.path == "envs/echo_env"
    assert entry.data.source.revision == git(repository, "rev-parse", "HEAD")
    assert entry.data.artifacts[0].path == "envs/echo_env"
    assert entry.data.artifacts[0].revision == entry.data.source.revision


def test_build_reads_only_the_selected_commit(repository: Path) -> None:
    first = build(repository)
    (repository / "envs/echo_env/pyproject.toml").write_text(
        '[project]\nname="changed"\ndescription="uncommitted description"\n'
    )
    environment(repository, "untracked_env", "Untracked environment.")
    assert build(repository).model_dump() == first.model_dump()


def test_two_builds_of_one_revision_are_byte_identical(
    repository: Path, tmp_path: Path
) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_catalog(build(repository), first)
    write_catalog(build(repository), second)
    assert first.read_bytes() == second.read_bytes()


def test_descriptions_and_requirements_are_source_declarations(
    repository: Path,
) -> None:
    entry = next(
        item for item in build(repository).entries if item.data.name == "echo_env"
    )
    assert entry.data.description == "Echo messages to test client tool calls."
    assert entry.data.manifest_spec_version == 1
    assert entry.data.framework_requirement == "openenv>=0.3.1"
    assert entry.data.license == "BSD-3-Clause"
    assert entry.data.license_url.endswith(f"/{entry.data.source.revision}/LICENSE")
    assert len(entry.data.interfaces) == 1
    assert entry.data.interfaces[0].role == "orchestration"
    assert entry.data.interfaces[0].protocol == "openenv"
    assert "version" not in entry.data.interfaces[0].model_dump(exclude_none=True)
    assert entry.metadata["provenance"]["description"] == (
        "envs/echo_env/pyproject.toml#project.description"
    )


def test_reviewable_declarations_provide_search_hints_and_tools(
    repository: Path,
) -> None:
    declaration = {
        "description": "A minimal client smoke test for message-based tools.",
        "tags": ["smoke-test"],
        "representative_queries": [
            "test a tool-calling client",
            "find a minimal client smoke test",
        ],
        "agent_tools": {
            "protocol": "mcp",
            "names": ["echo_message", "echo_with_length"],
            "source": "envs/echo_env/server.py",
        },
    }
    (repository / "envs/echo_env/discovery.json").write_text(json.dumps(declaration))
    (repository / "envs/echo_env/server.py").write_text(
        "# Authored declaration evidence\n"
    )
    revision = commit(repository)
    snapshot = build(repository)
    entry = search_catalog(snapshot, "client smoke test")[0].entry
    assert entry.data.description == declaration["description"]
    assert entry.capabilities == ["echo_message", "echo_with_length"]
    assert entry.representative_queries == declaration["representative_queries"]
    assert entry.data.interfaces[1].status == "declared"
    assert entry.data.interfaces[1].source_revision == revision
    assert (
        entry.metadata["provenance"]["agent_tools"]
        == declaration["agent_tools"]["source"]
    )


def test_license_conflicts_are_unknown_not_a_positive_claim(repository: Path) -> None:
    (repository / "envs/echo_env/README.md").write_text(
        "---\nlicense: apache-2.0\n---\n# Echo\n"
    )
    (repository / "envs/echo_env/pyproject.toml").write_text(
        '[project]\nname="echo"\ndescription="Echo messages."\nlicense="MIT"\n'
    )
    commit(repository)
    snapshot = build(repository)
    entry = next(item for item in snapshot.entries if item.data.name == "echo_env")
    assert snapshot.complete
    assert entry.data.license == "unknown"
    assert any(issue.code == "license_conflict" for issue in snapshot.issues)


@pytest.mark.parametrize(
    ("package_license", "readme_license"),
    [
        ("Custom terms A.", "Custom terms B."),
        ("Custom terms A.", "other"),
        ("other", "other"),
    ],
)
def test_custom_license_categories_do_not_establish_license_identity(
    repository: Path, package_license: str, readme_license: str
) -> None:
    (repository / "envs/echo_env/pyproject.toml").write_text(
        '[project]\nname="echo"\ndescription="Echo messages."\n'
        f"license={{text={json.dumps(package_license)}}}\n"
    )
    (repository / "envs/echo_env/README.md").write_text(
        f"---\nlicense: {json.dumps(readme_license)}\n---\n# Echo\n"
    )
    commit(repository)
    snapshot = build(repository)
    entry = next(item for item in snapshot.entries if item.data.name == "echo_env")
    assert snapshot.complete
    assert entry.data.license == "unknown"
    assert entry.data.license_url is None
    assert "license" not in entry.metadata["provenance"]
    assert any(
        issue.code == "license_conflict" and issue.path == "envs/echo_env"
        for issue in snapshot.issues
    )


@pytest.mark.parametrize(
    ("package_license", "readme_license", "expected"),
    [
        ("Custom terms A.", "Custom terms A.", "other"),
        ("Custom terms.\nSecond line.", "Custom terms.\r\nSecond line.", "other"),
        ("mit", "MIT", "MIT"),
    ],
)
def test_equivalent_license_declarations_keep_their_source_evidence(
    repository: Path, package_license: str, readme_license: str, expected: str
) -> None:
    (repository / "envs/echo_env/pyproject.toml").write_text(
        '[project]\nname="echo"\ndescription="Echo messages."\n'
        f"license={{text={json.dumps(package_license)}}}\n"
    )
    (repository / "envs/echo_env/README.md").write_text(
        f"---\nlicense: {json.dumps(readme_license)}\n---\n# Echo\n"
    )
    revision = commit(repository)
    snapshot = build(repository)
    entry = next(item for item in snapshot.entries if item.data.name == "echo_env")
    assert snapshot.complete
    assert entry.data.license == expected
    assert entry.data.license_url.endswith(f"/{revision}/envs/echo_env/pyproject.toml")
    assert not any(issue.code == "license_conflict" for issue in snapshot.issues)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_unclosed_frontmatter_is_invalid_with_each_line_ending(
    repository: Path, newline: str
) -> None:
    (repository / "envs/echo_env/README.md").write_bytes(
        newline.join(["---", "license: MIT", ""]).encode()
    )
    commit(repository)
    snapshot = build(repository)
    assert not snapshot.complete
    assert any(
        issue.code == "invalid_metadata" and issue.path == "envs/echo_env"
        for issue in snapshot.issues
    )


def test_missing_license_is_explicitly_unknown(repository: Path) -> None:
    (repository / "pyproject.toml").unlink()
    (repository / "LICENSE").unlink()
    commit(repository)
    assert {entry.data.license for entry in build(repository).entries} == {"unknown"}


@pytest.mark.parametrize("scope", ["environment", "repository"])
@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ('{file = "LICENSE"}', "unknown"),
        ('{text = "MIT"}', "MIT"),
        ('{text = "Custom license terms."}', "other"),
    ],
)
def test_license_tables_distinguish_file_pointers_from_license_text(
    repository: Path, scope: str, declaration: str, expected: str
) -> None:
    origin = (
        "envs/echo_env/pyproject.toml" if scope == "environment" else "pyproject.toml"
    )
    (repository / origin).write_text(
        '[project]\nname="fixture"\ndescription="Echo messages."\n'
        f"license={declaration}\n"
    )
    revision = commit(repository)
    snapshot = build(repository)
    entry = next(item for item in snapshot.entries if item.data.name == "echo_env")
    assert snapshot.complete
    assert entry.data.license == expected
    assert entry.metadata["provenance"]["license"] == origin
    if expected == "unknown":
        assert entry.data.license_url is None
    else:
        assert entry.data.license_url.endswith(f"/{revision}/{origin}")


def test_license_table_cannot_claim_both_a_file_and_text(repository: Path) -> None:
    (repository / "envs/echo_env/pyproject.toml").write_text(
        '[project]\nname="echo"\ndescription="Echo messages."\n'
        'license={file="LICENSE",text="MIT"}\n'
    )
    commit(repository)
    snapshot = build(repository)
    assert not snapshot.complete
    assert any(
        issue.path == "envs/echo_env"
        and issue.code == "invalid_metadata"
        and issue.severity == "error"
        for issue in snapshot.issues
    )


def test_failed_eligible_record_is_not_a_complete_inventory(
    repository: Path, tmp_path: Path
) -> None:
    (repository / "envs/echo_env/openenv.yaml").write_text("name: [not-a-name]\n")
    commit(repository)
    snapshot = build(repository)
    assert not snapshot.complete
    assert snapshot.inventory.paths == ["envs/chess_env", "envs/echo_env"]
    assert len(snapshot.entries) == 1
    assert any(
        issue.path == "envs/echo_env" and issue.severity == "error"
        for issue in snapshot.issues
    )
    path = tmp_path / "partial.json"
    write_catalog(snapshot, path)
    with pytest.raises(CatalogError, match="incomplete"):
        load_catalog(path)


def test_symlink_metadata_is_rejected_without_reading_its_target(
    repository: Path, tmp_path: Path
) -> None:
    secret = tmp_path / "private.toml"
    secret.write_text('description="must-not-be-read"\n')
    metadata = repository / "envs/echo_env/pyproject.toml"
    metadata.unlink()
    metadata.symlink_to(secret)
    commit(repository)
    snapshot = build(repository)
    assert not snapshot.complete
    assert any(issue.code == "symlink_metadata" for issue in snapshot.issues)
    assert "must-not-be-read" not in snapshot.model_dump_json()


def test_duplicate_metadata_keys_do_not_silently_override_claims(
    repository: Path,
) -> None:
    (repository / "envs/echo_env/discovery.json").write_text(
        '{"license":"MIT","license":"BSD-3-Clause"}'
    )
    commit(repository)
    snapshot = build(repository)
    assert not snapshot.complete
    assert any(issue.code == "invalid_metadata" for issue in snapshot.issues)


def test_publisher_cannot_supply_an_execution_approval_boolean(
    repository: Path,
) -> None:
    (repository / "envs/echo_env/discovery.json").write_text(
        '{"requires_explicit_trust":false}'
    )
    commit(repository)
    assert not build(repository).complete


@pytest.mark.parametrize("control", ["reset", "step", "state", "get_state"])
def test_agent_capabilities_do_not_advertise_simulation_controls(
    repository: Path, control: str
) -> None:
    (repository / "envs/echo_env/discovery.json").write_text(
        json.dumps(
            {
                "agent_tools": {
                    "protocol": "mcp",
                    "names": [control],
                    "source": "envs/echo_env/__init__.py",
                }
            }
        )
    )
    commit(repository)
    snapshot = build(repository)
    assert not snapshot.complete
    assert not any(control in entry.capabilities for entry in snapshot.entries)


def test_external_subject_keeps_known_revision_without_distribution(repository: Path):
    (repository / "envs/echo_env/discovery.json").write_text(
        '{"artifact_availability":"external"}'
    )
    revision = commit(repository)
    entry = next(
        item for item in build(repository).entries if item.data.name == "echo_env"
    )
    assert entry.data.source.revision == revision
    assert entry.data.artifact_availability == "external"
    assert entry.data.artifacts == []


def test_refresh_distinguishes_new_revisions_from_withdrawal(repository: Path) -> None:
    previous = build(repository)
    git(repository, "rm", "-qr", "envs/chess_env")
    (repository / "envs/echo_env/README.md").write_text("A corrected description.\n")
    commit(repository)
    current = build(repository)
    changes = compare_catalogs(previous, current)
    assert len(changes.superseded) == 1
    assert changes.superseded[0].previous.data.source.path == "envs/echo_env"
    assert changes.superseded[0].current.identifier != (
        changes.superseded[0].previous.identifier
    )
    assert [entry.data.source.path for entry in changes.withdrawn] == ["envs/chess_env"]
    assert (
        previous.entries[0].data.source.revision
        != current.entries[0].data.source.revision
    )


def test_refresh_reports_metadata_correction_without_changing_the_artifact(
    repository: Path,
) -> None:
    previous = build(repository)
    payload = previous.model_dump(by_alias=True, exclude_none=True, mode="json")
    payload["entries"][0]["description"] = "Corrected task description."
    payload["entries"][0]["data"]["description"] = "Corrected task description."
    payload.pop("digest")
    payload["digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
    )
    current = CatalogSnapshot.model_validate(payload)
    changes = compare_catalogs(previous, current)
    assert changes.added == []
    assert changes.withdrawn == []
    assert changes.superseded == []
    assert len(changes.corrected) == 1
    assert (
        changes.corrected[0].previous.identifier
        == changes.corrected[0].current.identifier
    )


def test_an_incomplete_refresh_cannot_withdraw_last_known_records(
    repository: Path,
) -> None:
    previous = build(repository)
    (repository / "envs/echo_env/openenv.yaml").write_text("not: [valid\n")
    commit(repository)
    with pytest.raises(CatalogError, match="incomplete"):
        compare_catalogs(previous, build(repository))


@pytest.mark.parametrize("root", ["unrelated", "."])
def test_snapshot_cannot_claim_entries_outside_its_inventory_scope(
    repository: Path, tmp_path: Path, root: str
) -> None:
    snapshot = build(repository)
    payload = snapshot.model_dump(by_alias=True, exclude_none=True, mode="json")
    payload["inventory"]["root"] = root
    payload.pop("digest")
    payload["digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
    )
    path = tmp_path / "wrong-scope.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(CatalogError, match="profile"):
        load_catalog(path)


def test_identifier_only_result_is_resolved_only_in_the_configured_snapshot(
    repository: Path,
) -> None:
    snapshot = build(repository)
    identifier = snapshot.entries[0].identifier
    assert resolve_entry(snapshot, identifier).identifier == identifier
    with pytest.raises(CatalogError, match="not found"):
        resolve_entry(snapshot, "urn:air:example.org:openenv:absent")


def test_filters_are_explicit_and_unknown_filters_fail(repository: Path) -> None:
    snapshot = build(repository)
    assert len(search_catalog(snapshot, "", filters={"license": "BSD-3-Clause"})) == 2
    assert search_catalog(snapshot, "", filters={"license": "MIT"}) == []
    with pytest.raises(CatalogError, match="Unsupported filter"):
        search_catalog(snapshot, "", filters={"provider_state": "RUNNING"})


def test_repository_credentials_are_not_reflected_in_errors(repository: Path) -> None:
    with pytest.raises(CatalogError) as error:
        build_catalog(
            repository,
            repository_uri="https://user:private-token@github.com/example/envs.git",
            publisher="example.org",
        )
    assert "private-token" not in str(error.value)
