"""Open Reward Standard source importer."""

from __future__ import annotations

import ast
from pathlib import Path

from .base import (
    append_dependency_files,
    collect_source_dependencies,
    DetectedEnvironment,
    iter_python_files,
    module_path,
    prepare_import_package,
    render_importer_template,
    write_text,
)


_ORS_MODULES = {
    "ors",
    "ors.environment",
    "openreward",
    "openreward.environment",
    "openreward.environments",
    "openreward.environments.environment",
    "openrewardstandard",
    "openrewardstandard.environment",
}

_ORS_ROOT_DEPENDENCIES = {
    "openreward": "openreward",
    "openrewardstandard": "openrewardstandard",
    "ors": "ors",
}


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    return None


def _collect_environment_aliases(
    tree: ast.AST,
) -> tuple[set[str], dict[str, str]]:
    environment_aliases: set[str] = set()
    module_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in _ORS_MODULES:
                for alias in node.names:
                    if alias.name == "Environment":
                        environment_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _ORS_MODULES:
                    module_aliases[alias.asname or alias.name.split(".", 1)[0]] = (
                        alias.name
                    )

    return environment_aliases, module_aliases


def _ors_dependency_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if module in _ORS_MODULES and root in _ORS_ROOT_DEPENDENCIES:
                roots.add(root)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if alias.name in _ORS_MODULES and root in _ORS_ROOT_DEPENDENCIES:
                    roots.add(root)
    return roots


def _inherits_ors_environment(
    base: ast.AST,
    environment_aliases: set[str],
    module_aliases: dict[str, str],
) -> bool:
    dotted = _dotted_name(base)
    if dotted is None:
        return False
    if dotted in environment_aliases:
        return True

    for alias, module in module_aliases.items():
        root = module.split(".", 1)[0]
        if dotted == f"{alias}.Environment":
            return module in _ORS_MODULES or root in _ORS_ROOT_DEPENDENCIES
        if dotted == f"{alias}.environment.Environment":
            return module in _ORS_MODULES or root in _ORS_ROOT_DEPENDENCIES
        if dotted == f"{alias}.environments.Environment":
            return module in _ORS_MODULES or root in _ORS_ROOT_DEPENDENCIES
        if dotted == f"{alias}.Environment" and (
            module.endswith(".environment") or module.endswith(".environments")
        ):
            return True
    return False


def detect_ors_environments(source: Path) -> list[DetectedEnvironment]:
    """Detect ORS/OpenReward environment classes without importing source files."""
    source = source.resolve()
    matches: list[DetectedEnvironment] = []

    for file_path in iter_python_files(source):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        environment_aliases, module_aliases = _collect_environment_aliases(tree)
        if not environment_aliases and not module_aliases:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                _inherits_ors_environment(base, environment_aliases, module_aliases)
                for base in node.bases
            ):
                matches.append(
                    DetectedEnvironment(
                        source_type="ors",
                        class_name=node.name,
                        module_path=module_path(source, file_path),
                        file_path=file_path,
                    )
                )

    return matches


def detect_ors_dependencies(source: Path) -> list[str]:
    roots: set[str] = set()
    for file_path in iter_python_files(source.resolve()):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        roots.update(_ors_dependency_roots(tree))

    dependencies = []
    for root in sorted(roots):
        if (source / root).exists() or (source / f"{root}.py").exists():
            continue
        dependencies.append(_ORS_ROOT_DEPENDENCIES[root])
    return dependencies


def _wrapper_source(
    *,
    env_name: str,
    class_name_prefix: str,
    source_module: str,
    source_class: str,
    vendor_dir: str,
) -> str:
    source_import_module = f"{env_name}.vendor.{vendor_dir}"
    if source_module:
        source_import_module = f"{source_import_module}.{source_module}"
    return render_importer_template(
        "ors_environment.py.tpl",
        class_name_prefix=class_name_prefix,
        source_import_module=source_import_module,
        source_class=source_class,
        vendor_dir=vendor_dir,
    )


def _app_source(*, env_name: str, class_name_prefix: str) -> str:
    return render_importer_template(
        "ors_app.py.tpl",
        env_name=env_name,
        class_name_prefix=class_name_prefix,
    )


class ORSImporter:
    """Importer for source repos that define ORS/OpenReward environments."""

    source_type = "ors"

    def detect(self, source: Path) -> list[DetectedEnvironment]:
        return detect_ors_environments(source)

    def generate(
        self,
        *,
        source: Path,
        destination: Path,
        env_name: str,
        detected: DetectedEnvironment,
    ) -> None:
        package = prepare_import_package(source, destination, env_name)
        write_text(
            destination / "server" / f"{env_name}_environment.py",
            _wrapper_source(
                env_name=env_name,
                class_name_prefix=package.class_name_prefix,
                source_module=detected.module_path,
                source_class=detected.class_name,
                vendor_dir=package.vendor_dir,
            ),
        )
        write_text(
            destination / "server" / "app.py",
            _app_source(env_name=env_name, class_name_prefix=package.class_name_prefix),
        )
        dependencies = collect_source_dependencies(source)
        for dependency in detect_ors_dependencies(source):
            if dependency not in dependencies:
                dependencies.append(dependency)
        append_dependency_files(destination, env_name, dependencies)
