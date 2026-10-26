"""Unified OpenEnv package bundling the CLI and core runtime."""

from __future__ import annotations

from importlib import import_module, metadata

__all__ = [
    "core",
    "cli",
    "AutoEnv",
    "AutoAction",
    "GenericEnvClient",
    "GenericAction",
    "SyncEnvClient",
]


def _load_package_version() -> str:
    """Resolve the installed distribution version for the OpenEnv package."""
    for distribution_name in ("openenv", "openenv-core"):
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
    return "0.0.0"


__version__ = _load_package_version()


_LAZY_MODULES = {
    "core": ".core",
    "cli": ".cli",
}

_LAZY_ATTRS = {
    "AutoEnv": (".auto", "AutoEnv"),
    "AutoAction": (".auto", "AutoAction"),
    "GenericEnvClient": (".core", "GenericEnvClient"),
    "GenericAction": (".core", "GenericAction"),
    "SyncEnvClient": (".core", "SyncEnvClient"),
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module

    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
