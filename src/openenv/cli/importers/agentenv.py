"""AgentEnv bundle source importer."""

from __future__ import annotations

import json
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from .._cli_utils import console
from .base import (
    append_dependency_files,
    collect_source_dependencies,
    DetectedEnvironment,
    iter_directories,
    prepare_import_package,
    render_importer_template,
    write_text,
)


_FRAMEWORK = "agentenv-framework"
_FRAMEWORK_EXTRA = "openenv"
_FRAMEWORK_VERSION = "0.9.1366"
_FRAMEWORK_FLOOR = f">={_FRAMEWORK_VERSION}"
_ADAPTER_SOURCE = (
    "https://github.com/scaleapi/agentenv-framework/blob/"
    f"v{_FRAMEWORK_VERSION}/src/agent_env/openenv"
)
_REQUIRES_PYTHON = ">=3.11.4"


def _is_task_file(path: Path) -> bool:
    try:
        steps = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return (
        isinstance(steps, list)
        and bool(steps)
        and all(
            isinstance(step, dict)
            and isinstance(step.get("id"), str)
            and isinstance(step.get("type"), str)
            for step in steps
        )
    )


def detect_agentenv_bundles(source: Path) -> list[DetectedEnvironment]:
    """Detect AgentEnv bundles, folders whose tasks/ holds task files, without importing source."""
    source = source.resolve()
    matches: list[DetectedEnvironment] = []

    for tasks_dir in iter_directories(source, "tasks"):
        if not any(_is_task_file(path) for path in sorted(tasks_dir.glob("*.json"))):
            continue
        bundle = tasks_dir.parent
        matches.append(
            DetectedEnvironment(
                source_type="agentenv",
                class_name=bundle.name,
                module_path=""
                if bundle == source
                else bundle.relative_to(source).as_posix(),
                file_path=bundle,
            )
        )

    return matches


def _with_framework_floor(dependencies: list[str]) -> list[str]:
    """The bundle's dependencies, with agentenv-framework's openenv extra at least the version that serves bundles."""
    for index, dependency in enumerate(dependencies):
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) == _FRAMEWORK:
            if requirement.url is not None:
                raise ValueError(
                    f"Cannot verify {_FRAMEWORK}{_FRAMEWORK_FLOOR} from a direct "
                    "reference. In the bundle's pyproject.toml or requirements.txt, "
                    "replace the framework direct reference with a version constraint "
                    f"such as {_FRAMEWORK}[{_FRAMEWORK_EXTRA}]{_FRAMEWORK_FLOOR}, "
                    "preserving any additional extras and environment marker."
                )
            requirement.extras.add(_FRAMEWORK_EXTRA)
            requirement.specifier &= SpecifierSet(_FRAMEWORK_FLOOR)
            dependencies[index] = str(requirement)
            return dependencies
    return [*dependencies, f"{_FRAMEWORK}[{_FRAMEWORK_EXTRA}]{_FRAMEWORK_FLOOR}"]


class AgentEnvImporter:
    """Importer for AgentEnv bundles, served by agentenv-framework's OpenEnv server."""

    source_type = "agentenv"

    def detect(self, source: Path) -> list[DetectedEnvironment]:
        return detect_agentenv_bundles(source)

    def generate(
        self,
        *,
        source: Path,
        destination: Path,
        env_name: str,
        detected: DetectedEnvironment,
    ) -> None:
        bundle = detected.file_path
        dependencies = _with_framework_floor(collect_source_dependencies(bundle))
        package = prepare_import_package(
            bundle, destination, env_name, vendor_dir=bundle.name
        )
        client = f"{package.class_name_prefix}Env"
        (destination / "server" / f"{env_name}_environment.py").unlink()
        (destination / "models.py").unlink()
        for name, template in (
            ("client.py", "agentenv_client.py.tpl"),
            ("__init__.py", "agentenv_init.py.tpl"),
        ):
            write_text(
                destination / name,
                render_importer_template(template, env_name=env_name, client=client),
            )
        write_text(
            destination / "server" / "__init__.py",
            f'"""{env_name} environment server: an AgentEnv bundle."""\n',
        )
        write_text(
            destination / "server" / "app.py",
            render_importer_template(
                "agentenv_app.py.tpl", bundle=json.dumps(bundle.name)
            ),
        )
        write_text(
            destination / "README.md",
            render_importer_template(
                "agentenv_README.md.tpl",
                env_name=env_name,
                bundle=bundle.name,
                framework=f"{_FRAMEWORK}[{_FRAMEWORK_EXTRA}]{_FRAMEWORK_FLOOR}",
                adapter=_ADAPTER_SOURCE,
                client=client,
            ),
        )
        console.print(
            "[yellow]Warning:[/yellow] AgentEnv requires local Docker: a Docker CLI "
            "and a running daemon accessible to the server for image builds and episode "
            "containers. The generated server/Dockerfile does not provide this "
            "runtime; openenv build / openenv push alone do not produce a working "
            "deployment. See README.md for prerequisites."
        )
        append_dependency_files(
            destination,
            env_name,
            dependencies,
            requires_python=_REQUIRES_PYTHON,
        )
