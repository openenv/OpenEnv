# SPDX-License-Identifier: BSD-3-Clause

"""Source declarations used by discovery, independent of runtime quality grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import tomli
import yaml
from packaging.licenses import canonicalize_license_expression, InvalidLicenseExpression
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from pydantic import Field, field_validator

from .errors import CatalogError
from .models import (
    NonEmpty,
    ProfileModel,
    RelativePath,
    RepresentativeQueries,
    SIMULATION_CONTROLS,
)
from .repository import GitMetadataSource, MetadataError
from .serialization import parse_json


class ToolDeclaration(ProfileModel):
    protocol: NonEmpty
    names: list[NonEmpty] = Field(min_length=1, max_length=256)
    source: RelativePath

    @field_validator("names")
    @classmethod
    def task_tools_only(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("tool declarations must not contain duplicate names")
        if any(name.casefold() in SIMULATION_CONTROLS for name in value):
            raise ValueError("simulation controls cannot be agent tools")
        return value


class DiscoveryDeclaration(ProfileModel):
    description: NonEmpty | None = None
    tags: list[NonEmpty] = Field(default_factory=list)
    representative_queries: RepresentativeQueries = Field(default_factory=list)
    license: NonEmpty | None = None
    license_source: RelativePath | None = None
    artifact_availability: Literal["resolvable", "external", "unknown"] = "resolvable"
    agent_tools: ToolDeclaration | None = None

    @field_validator("license")
    @classmethod
    def license_expression(cls, value: str | None) -> str | None:
        if value is None or value in {"unknown", "other"}:
            return value
        return str(canonicalize_license_expression(value))


class _UniqueYamlLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "Expected mapping", node.start_mark
            )
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    "Mapping keys must be unique strings",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def yaml_mapping(value: str, path: str) -> dict[str, Any]:
    try:
        result = yaml.load(value, Loader=_UniqueYamlLoader)
    except (yaml.YAMLError, RecursionError) as error:
        raise MetadataError(
            "invalid_metadata", f"Invalid YAML metadata: {path}"
        ) from error
    if not isinstance(result, dict):
        raise MetadataError("invalid_metadata", f"Metadata must be a mapping: {path}")
    return result


def project_metadata(source: GitMetadataSource, path: str) -> dict[str, Any]:
    text = source.read(path)
    if text is None:
        return {}
    try:
        parsed = tomli.loads(text)
    except tomli.TOMLDecodeError as error:
        raise MetadataError(
            "invalid_metadata", f"Invalid package metadata: {path}"
        ) from error
    project = parsed.get("project", {})
    if not isinstance(project, dict):
        raise MetadataError(
            "invalid_metadata", f"Project metadata must be a mapping: {path}"
        )
    return project


def readme_metadata(source: GitMetadataSource, path: str) -> dict[str, Any]:
    text = source.read(path)
    if text is None or not text.startswith(("---\n", "---\r")):
        return {}
    lines = text.splitlines()
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise MetadataError(
            "invalid_metadata", f"Unclosed README frontmatter: {path}"
        ) from error
    return yaml_mapping("\n".join(lines[1:end]), path)


def discovery_declaration(source: GitMetadataSource, path: str) -> DiscoveryDeclaration:
    text = source.read(path)
    if text is None:
        return DiscoveryDeclaration()
    try:
        raw = parse_json(text)
    except CatalogError as error:
        raise MetadataError(
            "invalid_metadata", f"Invalid discovery declaration: {path}"
        ) from error
    return DiscoveryDeclaration.model_validate(raw)


@dataclass(frozen=True)
class _ParsedLicense:
    expression: str
    text: str | None = None


def license_declaration(value: object) -> _ParsedLicense | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "file" in value and "text" in value:
            raise MetadataError(
                "invalid_metadata",
                "License file and text declarations are mutually exclusive",
            )
        value = value.get("text")
        if value is None:
            # A file pointer alone does not identify an SPDX or custom license.
            return _ParsedLicense("unknown")
    if not isinstance(value, str) or not value.strip():
        return _ParsedLicense("unknown")
    value = value.strip()
    if value in {"unknown", "other"}:
        return _ParsedLicense(value)
    try:
        return _ParsedLicense(str(canonicalize_license_expression(value)))
    except InvalidLicenseExpression:
        return _ParsedLicense("other", value.replace("\r\n", "\n").replace("\r", "\n"))


def framework_requirement(project: dict[str, Any]) -> str | None:
    requirements = []
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise MetadataError("invalid_metadata", "Package dependencies must be an array")
    for dependency in dependencies:
        if not isinstance(dependency, str):
            raise MetadataError(
                "invalid_metadata", "Dependency declarations must be strings"
            )
        try:
            parsed = Requirement(dependency)
        except InvalidRequirement as error:
            raise MetadataError(
                "invalid_metadata", "Invalid package dependency declaration"
            ) from error
        if canonicalize_name(parsed.name) == "openenv":
            requirements.append(dependency)
    if len(requirements) > 1:
        raise MetadataError(
            "invalid_metadata",
            "Multiple OpenEnv requirements need an explicit resolution",
        )
    return requirements[0] if requirements else None
