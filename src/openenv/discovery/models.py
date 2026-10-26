# SPDX-License-Identifier: BSD-3-Clause

"""The declaration-only Environment Card and repository snapshot profile."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urlsplit

from packaging.licenses import canonicalize_license_expression
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    JsonValue,
    model_validator,
    StringConstraints,
)

PROFILE_VERSION = "0.1-draft"
ENVIRONMENT_MEDIA_TYPE = "application/vnd.openenv.environment-card+json"
ARD_CONTEXT = "https://agenticresourcediscovery.org/context/v1"
SIMULATION_CONTROLS = frozenset({"reset", "step", "state", "get_state"})
MAX_CATALOG_BYTES = 16 * 1024 * 1024

NonEmpty = Annotated[
    str, StringConstraints(min_length=1, max_length=8192, pattern=r"\S")
]
RepresentativeQueries = (
    Annotated[list[NonEmpty], Field(max_length=0)]
    | Annotated[list[NonEmpty], Field(min_length=2, max_length=5)]
)
Revision = Annotated[str, StringConstraints(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
Publisher = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=253,
        pattern=r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}$",
    ),
]


def relative_path(value: str) -> str:
    """Validate a normalized repository-relative locator without resolving it."""
    if value == ".":
        return value
    if (
        not value
        or any(
            ord(character) < 0x20
            or 0x7F <= ord(character) <= 0x9F
            or ord(character) in (0x2028, 0x2029)
            for character in value
        )
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("path must be a normalized repository-relative POSIX path")
    return value


# Positive components exclude "." and ".." without lookaround, which some
# JSON Schema regex engines do not support.
_CONTROL_CHARACTER_PATTERN = r"[\x00-\x1f\x7f-\x9f\u2028\u2029]"
_PATH_COMPONENT_PATTERN = (
    r"(?:[^./\\\x00-\x1f\x7f-\x9f\u2028\u2029]"
    r"|\.[^./\\\x00-\x1f\x7f-\x9f\u2028\u2029]"
    r"|\.\.[^./\\\x00-\x1f\x7f-\x9f\u2028\u2029]"
    r"|\.\.\.)[^/\\\x00-\x1f\x7f-\x9f\u2028\u2029]*"
)
RelativePath = Annotated[
    NonEmpty,
    AfterValidator(relative_path),
    Field(
        json_schema_extra={
            "allOf": [
                {
                    "pattern": (
                        rf"^(?:\.|{_PATH_COMPONENT_PATTERN}"
                        rf"(?:/{_PATH_COMPONENT_PATTERN})*)$"
                    ),
                },
                {"not": {"pattern": _CONTROL_CHARACTER_PATTERN}},
            ]
        }
    ),
]


def github_repository(uri: str) -> str:
    """Return the native repository identity of a credential-free GitHub URI."""
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
        or "%" in parsed.path
        or not parsed.path.endswith(".git")
    ):
        raise ValueError(
            "repository_uri must be a credential-free GitHub HTTPS Git URI"
        )
    identity = parsed.path.removeprefix("/")[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", identity):
        raise ValueError("repository_uri must identify an owner and repository")
    if any(part in (".", "..") for part in identity.split("/")):
        raise ValueError("repository_uri contains an invalid repository component")
    return identity


def metadata_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "\\" in parsed.netloc
    ):
        raise ValueError("metadata references must be credential-free HTTPS URLs")
    return value


MetadataURL = Annotated[NonEmpty, AfterValidator(metadata_url)]


class ProfileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        populate_by_name=True,
        allow_inf_nan=False,
    )


class Owner(ProfileModel):
    """A source-owner declaration, not verification or copyright attribution."""

    authority: NonEmpty
    id: NonEmpty


class RepositorySource(ProfileModel):
    provider: Literal["github"]
    id: NonEmpty
    uri: NonEmpty
    revision: Revision

    @model_validator(mode="after")
    def consistent_repository(self) -> RepositorySource:
        if github_repository(self.uri) != self.id:
            raise ValueError("source.id must match the repository URI")
        return self


class EnvironmentSource(RepositorySource):
    path: RelativePath


class GitArtifact(ProfileModel):
    kind: Literal["git"]
    uri: NonEmpty
    path: RelativePath
    revision: Revision


class OrchestrationInterface(ProfileModel):
    role: Literal["orchestration"]
    protocol: Literal["openenv"]


class AgentToolsInterface(ProfileModel):
    role: Literal["agent-tools"]
    protocol: NonEmpty
    status: Literal["declared"]
    source_revision: Revision


Interface = Annotated[
    OrchestrationInterface | AgentToolsInterface, Field(discriminator="role")
]


class EnvironmentCard(ProfileModel):
    """A revision-bound source declaration under the experimental 0.1 profile."""

    model_config = ConfigDict(
        json_schema_extra={
            "$comment": (
                "JSON Schema covers structure, relative paths and conditional "
                "presence/cardinality. Full profile validation also requires "
                "URI, SPDX and requirement parsing, source/artifact equality "
                "and agent-tool revision/protocol checks. See "
                "docs/source/guides/catalog-discovery.md#consumer-validation."
            ),
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "artifact_availability": {"const": "resolvable"}
                        },
                        "required": ["artifact_availability"],
                    },
                    "then": {
                        "properties": {"artifacts": {"minItems": 1}},
                        "required": ["artifacts"],
                    },
                    "else": {"properties": {"artifacts": {"maxItems": 0}}},
                },
                {
                    "if": {
                        "properties": {"license": {"const": "other"}},
                        "required": ["license"],
                    },
                    "then": {
                        "properties": {"license_url": {"type": "string"}},
                        "required": ["license_url"],
                    },
                },
            ],
        }
    )

    schema_version: Literal["0.1-draft"]
    name: NonEmpty
    description: NonEmpty
    owner: Owner | Literal["unknown"]
    source: EnvironmentSource
    artifact_availability: Literal["resolvable", "external", "unknown"]
    artifacts: list[GitArtifact] = Field(default_factory=list)
    license: NonEmpty
    license_url: MetadataURL | None = None
    interfaces: list[Interface] = Field(
        json_schema_extra={
            "contains": {
                "type": "object",
                "properties": {"role": {"const": "orchestration"}},
                "required": ["role"],
            },
            "minContains": 1,
            "maxContains": 1,
        }
    )
    manifest_spec_version: int | str | None = None
    framework_requirement: NonEmpty | None = None

    @field_validator("license")
    @classmethod
    def declared_license(cls, value: str) -> str:
        if value in {"other", "unknown"}:
            return value
        return str(canonicalize_license_expression(value))

    @field_validator("framework_requirement")
    @classmethod
    def declared_requirement(cls, value: str | None) -> str | None:
        if value is not None:
            requirement = Requirement(value)
            if canonicalize_name(requirement.name) != "openenv":
                raise ValueError(
                    "framework_requirement must declare the openenv package"
                )
        return value

    @model_validator(mode="after")
    def subject_and_claims(self) -> EnvironmentCard:
        if self.artifact_availability == "resolvable" and not self.artifacts:
            raise ValueError("resolvable records require an immutable artifact")
        if self.artifact_availability != "resolvable" and self.artifacts:
            raise ValueError(
                "external or unknown availability must not claim artifacts"
            )
        for artifact in self.artifacts:
            if (artifact.uri, artifact.path, artifact.revision) != (
                self.source.uri,
                self.source.path,
                self.source.revision,
            ):
                raise ValueError("artifact does not identify the card's source subject")
        if self.license == "other" and self.license_url is None:
            raise ValueError("an other license requires an authoritative reference")
        orchestration = [
            item for item in self.interfaces if item.role == "orchestration"
        ]
        if len(orchestration) != 1:
            raise ValueError("exactly one orchestration descriptor is required")
        protocols = set()
        for interface in self.interfaces:
            if isinstance(interface, AgentToolsInterface):
                if interface.source_revision != self.source.revision:
                    raise ValueError("tool declaration must match source revision")
                if interface.protocol in protocols:
                    raise ValueError("duplicate agent-tools protocol declaration")
                protocols.add(interface.protocol)
        return self


class DiscoveryEntry(ProfileModel):
    context: Literal["https://agenticresourcediscovery.org/context/v1"] = Field(
        default=ARD_CONTEXT, alias="@context"
    )
    identifier: NonEmpty
    display_name: NonEmpty = Field(alias="displayName")
    type: Literal["application/vnd.openenv.environment-card+json"] = (
        ENVIRONMENT_MEDIA_TYPE
    )
    data: EnvironmentCard
    description: NonEmpty
    tags: list[NonEmpty] = Field(default_factory=list)
    capabilities: list[NonEmpty] = Field(default_factory=list)
    representative_queries: RepresentativeQueries = Field(
        default_factory=list, alias="representativeQueries"
    )
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def revision_card(self) -> DiscoveryEntry:
        if not self.identifier.startswith("urn:air:") or not self.identifier.endswith(
            ":" + self.data.source.revision
        ):
            raise ValueError("identifier must name one immutable revision card")
        if any(name.casefold() in SIMULATION_CONTROLS for name in self.capabilities):
            raise ValueError("agent capabilities must not advertise simulation control")
        if self.capabilities and not any(
            item.role == "agent-tools" for item in self.data.interfaces
        ):
            raise ValueError(
                "capabilities require an evidenced agent-tools declaration"
            )
        return self


class Inventory(ProfileModel):
    root: RelativePath
    paths: list[RelativePath]

    @model_validator(mode="after")
    def direct_children(self) -> Inventory:
        parent = "" if self.root == "." else self.root
        if any(path == "." or path.rpartition("/")[0] != parent for path in self.paths):
            raise ValueError(
                "inventory paths must be direct children of its declared root"
            )
        return self


class CatalogIssue(ProfileModel):
    path: RelativePath
    code: NonEmpty
    message: NonEmpty
    severity: Literal["warning", "error"]


class Generator(ProfileModel):
    name: Literal["openenv-git-catalog"] = "openenv-git-catalog"
    version: Literal["1"] = "1"


class CatalogSnapshot(ProfileModel):
    schema_version: Literal["0.1-draft"] = PROFILE_VERSION
    publisher: Publisher
    source: RepositorySource
    generator: Generator = Field(default_factory=Generator)
    inventory: Inventory
    entries: list[DiscoveryEntry]
    issues: list[CatalogIssue] = Field(default_factory=list)
    complete: bool
    digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def inventory_accounting(self) -> CatalogSnapshot:
        paths = self.inventory.paths
        if len(paths) != len(set(paths)):
            raise ValueError("inventory contains duplicate environment paths")
        errors = {issue.path for issue in self.issues if issue.severity == "error"}
        found = []
        identifiers = set()
        for entry in self.entries:
            source = entry.data.source
            if (source.id, source.uri, source.revision) != (
                self.source.id,
                self.source.uri,
                self.source.revision,
            ):
                raise ValueError("entry source differs from the recorded inventory")
            if not entry.identifier.startswith(f"urn:air:{self.publisher}:"):
                raise ValueError("entry publisher differs from the snapshot publisher")
            if entry.identifier in identifiers:
                raise ValueError("duplicate revision-card identifier")
            identifiers.add(entry.identifier)
            found.append(source.path)
        if len(found) != len(set(found)):
            raise ValueError("multiple cards describe one environment path")
        if set(found) & errors or set(found) | errors != set(paths):
            raise ValueError("inventory must account for every eligible environment")
        if self.complete != (not errors):
            raise ValueError("complete must reflect eligible-record failures")
        return self


class SearchMatch(ProfileModel):
    entry: DiscoveryEntry
    score: Annotated[int, Field(ge=0, le=100)]


class SupersededEntry(ProfileModel):
    previous: DiscoveryEntry
    current: DiscoveryEntry


class CatalogChanges(ProfileModel):
    added: list[DiscoveryEntry]
    withdrawn: list[DiscoveryEntry]
    superseded: list[SupersededEntry]
    corrected: list[SupersededEntry]
