"""Prime Intellect Verifiers source importer."""

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


_VERIFIERS_MODULES = {
    "verifiers",
    "verifiers.envs.environment",
    "verifiers.v1",
}


def _imports_verifiers(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "verifiers" or alias.name.startswith("verifiers."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in _VERIFIERS_MODULES or module.startswith("verifiers."):
                return True
    return False


def detect_verifiers_environments(source: Path) -> list[DetectedEnvironment]:
    """Detect Verifiers load_environment entrypoints without importing source."""
    source = source.resolve()
    matches: list[DetectedEnvironment] = []

    for file_path in iter_python_files(source):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        if not _imports_verifiers(tree):
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "load_environment":
                    matches.append(
                        DetectedEnvironment(
                            source_type="verifiers",
                            class_name=node.name,
                            module_path=module_path(source, file_path),
                            file_path=file_path,
                        )
                    )

    return matches


def _wrapper_source(
    *,
    env_name: str,
    class_name_prefix: str,
    source_module: str,
    vendor_dir: str,
) -> str:
    source_import_module = f"{env_name}.vendor.{vendor_dir}"
    if source_module:
        source_import_module = f"{source_import_module}.{source_module}"
    return render_importer_template(
        "verifiers_environment.py.tpl",
        class_name_prefix=class_name_prefix,
        source_import_module=source_import_module,
        vendor_dir=vendor_dir,
    )


def _app_source(*, env_name: str, class_name_prefix: str) -> str:
    return render_importer_template(
        "verifiers_app.py.tpl",
        env_name=env_name,
        class_name_prefix=class_name_prefix,
    )


class VerifiersImporter:
    """Importer for Prime Intellect Verifiers environment modules."""

    source_type = "verifiers"

    def detect(self, source: Path) -> list[DetectedEnvironment]:
        return detect_verifiers_environments(source)

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
                vendor_dir=package.vendor_dir,
            ),
        )
        write_text(
            destination / "server" / "app.py",
            _app_source(env_name=env_name, class_name_prefix=package.class_name_prefix),
        )
        dependencies = collect_source_dependencies(source)
        if "verifiers>=0.1.14" not in dependencies:
            dependencies.append("verifiers>=0.1.14")
        append_dependency_files(destination, env_name, dependencies)
