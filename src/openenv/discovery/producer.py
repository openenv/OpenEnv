# SPDX-License-Identifier: BSD-3-Clause

"""A deterministic repository inventory projected into revision-bound ARD entries."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .errors import CatalogError
from .metadata import (
    discovery_declaration,
    DiscoveryDeclaration,
    framework_requirement,
    license_declaration,
    project_metadata,
    readme_metadata,
    yaml_mapping,
)
from .models import (
    AgentToolsInterface,
    CatalogIssue,
    CatalogSnapshot,
    DiscoveryEntry,
    EnvironmentCard,
    EnvironmentSource,
    GitArtifact,
    Inventory,
    OrchestrationInterface,
    Owner,
    PROFILE_VERSION,
    Publisher,
    RepositorySource,
)
from .repository import GitMetadataSource, MetadataError
from .serialization import catalog_digest


def _field(
    documents: list[tuple[dict[str, Any], str]], key: str
) -> tuple[object | None, str | None]:
    for document, origin in documents:
        if key in document and document[key] is not None:
            separator = "." if "#" in origin else "#"
            return document[key], f"{origin}{separator}{key}"
    return None, None


def _license(
    source: GitMetadataSource,
    path: str,
    declaration: DiscoveryDeclaration,
    project: dict[str, Any],
    readme: dict[str, Any],
    root_project: dict[str, Any],
) -> tuple[str, str | None, str | None, list[CatalogIssue]]:
    if declaration.license is not None:
        evidence = declaration.license_source or f"{path}/discovery.json"
        source.read(evidence, required=True)
        return (
            declaration.license,
            None if declaration.license == "unknown" else source.url(evidence),
            evidence,
            [],
        )
    candidates = [
        (license_declaration(project.get("license")), f"{path}/pyproject.toml"),
        (license_declaration(readme.get("license")), f"{path}/README.md"),
    ]
    known = [(value, origin) for value, origin in candidates if value is not None]
    if len({value for value, _ in known}) > 1 or (
        len(known) > 1
        and known[0][0].expression == "other"
        and known[0][0].text is None
    ):
        return (
            "unknown",
            None,
            None,
            [
                CatalogIssue(
                    path=path,
                    code="license_conflict",
                    message=(
                        "Package and README declarations do not identify one license"
                    ),
                    severity="warning",
                )
            ],
        )
    if known:
        parsed, origin = known[0]
        expression = parsed.expression
        return (
            expression,
            None if expression == "unknown" else source.url(origin),
            origin,
            [],
        )
    parsed = license_declaration(root_project.get("license"))
    if parsed is None:
        return "unknown", None, None, []
    expression = parsed.expression
    origin = "pyproject.toml"
    for filename in root_project.get("license-files", []):
        if isinstance(filename, str) and filename in source.files:
            source.read(filename, required=True)
            origin = filename
            break
    return (
        expression,
        None if expression == "unknown" else source.url(origin),
        origin,
        [],
    )


def _entry(
    source: GitMetadataSource,
    path: str,
    publisher: str,
    namespace: str,
    root_project: dict[str, Any],
) -> tuple[DiscoveryEntry, list[CatalogIssue]]:
    manifest_path = f"{path}/openenv.yaml"
    manifest_text = source.read(manifest_path, required=True)
    if manifest_text is None:
        raise MetadataError("missing_metadata", "Environment manifest is absent")
    manifest = yaml_mapping(manifest_text, manifest_path)
    project_path = f"{path}/pyproject.toml"
    project = project_metadata(source, project_path)
    readme_path = f"{path}/README.md"
    readme = readme_metadata(source, readme_path)
    declaration_path = f"{path}/discovery.json"
    declaration = discovery_declaration(source, declaration_path)
    explicit = declaration.model_dump(exclude_none=True)
    name, name_origin = _field(
        [(manifest, manifest_path), (project, project_path + "#project")], "name"
    )
    description, description_origin = _field(
        [
            (explicit, declaration_path),
            (manifest, manifest_path),
            (project, project_path + "#project"),
            (readme, readme_path),
        ],
        "description",
    )
    if not isinstance(name, str) or not name.strip():
        raise MetadataError("invalid_metadata", "Environment needs a declared name")
    if not isinstance(description, str) or not description.strip():
        raise MetadataError(
            "invalid_metadata", "Environment needs an authored task description"
        )
    license_name, license_url, license_origin, issues = _license(
        source, path, declaration, project, readme, root_project
    )
    provenance: dict[str, Any] = {
        "name": name_origin,
        "description": description_origin,
        "source": source.uri,
    }
    if license_origin is not None:
        provenance["license"] = license_origin
    environment_source = EnvironmentSource(
        provider="github",
        id=source.repository_id,
        uri=source.uri,
        path=path,
        revision=source.revision,
    )
    artifacts = []
    if declaration.artifact_availability == "resolvable":
        artifacts.append(
            GitArtifact(kind="git", uri=source.uri, path=path, revision=source.revision)
        )
    interfaces: list[OrchestrationInterface | AgentToolsInterface] = [
        OrchestrationInterface(role="orchestration", protocol="openenv")
    ]
    capabilities = []
    if declaration.agent_tools is not None:
        tools = declaration.agent_tools
        if not tools.source.startswith(path + "/"):
            raise MetadataError(
                "invalid_metadata", "Agent-tool evidence must belong to the environment"
            )
        source.read(tools.source, required=True)
        interfaces.append(
            AgentToolsInterface(
                role="agent-tools",
                protocol=tools.protocol,
                status="declared",
                source_revision=source.revision,
            )
        )
        capabilities = tools.names
        provenance["agent_tools"] = tools.source
    tags = declaration.tags if declaration.tags else readme.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise MetadataError("invalid_metadata", "Tags must be an array of strings")
    requirement = framework_requirement(project)
    if requirement is not None:
        provenance["framework_requirement"] = project_path + "#project.dependencies"
    if "spec_version" in manifest:
        provenance["manifest_spec_version"] = manifest_path + "#spec_version"
    card = EnvironmentCard(
        schema_version=PROFILE_VERSION,
        name=name,
        description=description,
        owner=Owner(authority="github.com", id=source.repository_id.split("/", 1)[0]),
        source=environment_source,
        artifact_availability=declaration.artifact_availability,
        artifacts=artifacts,
        license=license_name,
        license_url=license_url,
        interfaces=interfaces,
        manifest_spec_version=manifest.get("spec_version"),
        framework_requirement=requirement,
    )
    locator = hashlib.sha256(f"{source.uri}\n{path}".encode("utf-8")).hexdigest()
    identifier = f"urn:air:{publisher}:{namespace}:{locator}:{source.revision}"
    return (
        DiscoveryEntry(
            identifier=identifier,
            display_name=name,
            description=description,
            data=card,
            tags=sorted(set(tags)),
            capabilities=capabilities,
            representative_queries=declaration.representative_queries,
            metadata={"provenance": provenance},
        ),
        issues,
    )


def build_catalog(
    repository: str | Path,
    *,
    repository_uri: str,
    publisher: str,
    namespace: str = "openenv",
    revision: str = "HEAD",
    root: str = "envs",
) -> CatalogSnapshot:
    """
    Produce a metadata snapshot from tracked files at one source revision.

    Args:
        repository (`str` or `Path`):
            Local Git repository. Candidate Python code is never imported.
        repository_uri (`str`):
            Public GitHub HTTPS clone URI of the declared source.
        publisher (`str`):
            Explicit publication authority. This declaration does not verify ownership.
        namespace (`str`, *optional*, defaults to `"openenv"`):
            Publisher-owned discovery namespace.
        revision (`str`, *optional*, defaults to `"HEAD"`):
            Git commit or ref resolved once before enumeration.
        root (`str`, *optional*, defaults to `"envs"`):
            Tracked inventory root containing direct-child environment definitions.
    """
    try:
        TypeAdapter(Publisher).validate_python(publisher)
    except ValidationError as error:
        raise CatalogError("publisher must be an explicit DNS authority") from error
    if not re.fullmatch(r"[A-Za-z0-9._-]+", namespace):
        raise CatalogError("namespace must be a non-empty ARD name component")
    source = GitMetadataSource(Path(repository), repository_uri, revision, root)
    root_project = project_metadata(source, "pyproject.toml")
    entries = []
    issues = []
    for path in source.environments:
        try:
            entry, warnings = _entry(source, path, publisher, namespace, root_project)
        except (MetadataError, ValidationError) as error:
            code = (
                error.code if isinstance(error, MetadataError) else "invalid_metadata"
            )
            message = (
                str(error)
                if isinstance(error, MetadataError)
                else ("Metadata does not satisfy the declaration profile")
            )
            issues.append(
                CatalogIssue(path=path, code=code, message=message, severity="error")
            )
        else:
            entries.append(entry)
            issues.extend(warnings)
    snapshot = CatalogSnapshot(
        publisher=publisher,
        source=RepositorySource(
            provider="github",
            id=source.repository_id,
            uri=source.uri,
            revision=source.revision,
        ),
        inventory=Inventory(root=source.root, paths=source.environments),
        entries=entries,
        issues=issues,
        complete=not any(issue.severity == "error" for issue in issues),
        digest="sha256:" + "0" * 64,
    )
    return snapshot.model_copy(update={"digest": catalog_digest(snapshot)})
