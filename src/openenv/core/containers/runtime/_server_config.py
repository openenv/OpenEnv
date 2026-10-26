# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for reading environment server configuration."""

from __future__ import annotations

import json
import re

import yaml


def parse_openenv_app_field(yaml_content: str) -> str | None:
    """Extract the top-level ``app`` value from raw ``openenv.yaml`` content."""
    try:
        data = yaml.safe_load(yaml_content) or {}
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    value = data.get("app")
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return None


def parse_dockerfile_cmd(dockerfile_content: str) -> str | None:
    """Extract the server command from the last Dockerfile ``CMD``."""
    last_cmd: str | None = None
    for line in dockerfile_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r"CMD\s+(.+)", stripped, flags=re.IGNORECASE)
        if match:
            last_cmd = match.group(1).strip()

    if last_cmd is None:
        return None

    if last_cmd.startswith("["):
        try:
            parts = json.loads(last_cmd)
            if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                return " ".join(parts)
        except (json.JSONDecodeError, TypeError):
            # Invalid exec-form JSON falls back to the raw CMD below.
            pass

    return last_cmd if last_cmd else None
