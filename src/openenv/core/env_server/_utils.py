"""Shared env server implementation helpers."""

from __future__ import annotations

from typing import Any


def overrides_method(method: Any, base_method: Any) -> bool:
    """Return whether a bound method differs from the base implementation."""
    return getattr(method, "__func__", method) is not base_method
