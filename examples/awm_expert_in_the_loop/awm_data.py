"""Client-side access to the AgentWorldModel-1K dataset.

The verifier-informed expert needs each task's Python verifier source. This module
reads it from the same public Hugging Face dataset the environment server uses
(`Snowflake/AgentWorldModel-1K`), but WITHOUT importing anything from the env
``server/`` package, which would break OpenEnv's client/server separation invariant.

Files are cached under ``AWM_DATA_DIR`` or ``~/.cache/openenv/awm`` (the same default
the server uses, so an existing cache is reused).
"""

from __future__ import annotations

import json
import os
import re
import threading

from huggingface_hub import hf_hub_download

HF_REPO_ID = "Snowflake/AgentWorldModel-1K"
HF_REPO_TYPE = "dataset"
_VERIFIER_CODE_FILE = "gen_verifier.pure_code.jsonl"

_lock = threading.Lock()
_verifier_index: dict[str, list[dict]] | None = None


def _cache_dir() -> str:
    return os.environ.get("AWM_DATA_DIR", os.path.expanduser("~/.cache/openenv/awm"))


def normalize_scenario_name(scenario: str) -> str:
    """Normalise a scenario name the same way the env server does."""
    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").strip()
    return s


def _ensure_file(filename: str) -> str:
    path = os.path.join(_cache_dir(), filename)
    if not os.path.exists(path):
        os.makedirs(_cache_dir(), exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=filename,
            local_dir=_cache_dir(),
        )
    return path


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _verifiers() -> dict[str, list[dict]]:
    global _verifier_index
    if _verifier_index is None:
        with _lock:
            if _verifier_index is None:
                index: dict[str, list[dict]] = {}
                for item in _load_jsonl(_ensure_file(_VERIFIER_CODE_FILE)):
                    key = normalize_scenario_name(item["scenario"])
                    index.setdefault(key, []).append(item)
                _verifier_index = index
    return _verifier_index


def get_verifier_code(scenario: str, task_idx: int) -> str | None:
    """Return the Python ``code`` verifier for a task, or `None` if absent."""
    for entry in _verifiers().get(normalize_scenario_name(scenario), []):
        if entry.get("task_idx") == task_idx:
            return entry.get("verification", {}).get("code", "")
    return None


def iter_verifier_tasks(scenario_prefixes: tuple[str, ...] | None = None):
    """Yield `(scenario, task_idx)` for every task that has a code verifier.

    Args:
        scenario_prefixes (`tuple[str, ...]`, *optional*):
            When given, only scenarios whose normalised name starts with one of
            these prefixes are yielded (for example `("workflow_automation",)`).
    """
    for scenario, entries in _verifiers().items():
        if scenario_prefixes and not scenario.startswith(scenario_prefixes):
            continue
        for entry in entries:
            task_idx = entry.get("task_idx")
            if task_idx is not None:
                yield scenario, task_idx


def list_scenarios() -> list[str]:
    """Return the sorted scenario names that have at least one code verifier."""
    return sorted(_verifiers().keys())
