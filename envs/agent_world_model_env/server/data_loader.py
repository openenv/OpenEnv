"""
Data loading and caching for Agent World Model environments.

Downloads data from HuggingFace (Snowflake/AgentWorldModel-1K) on first use
and caches it locally. Provides typed accessors for scenarios, tasks, env code,
DB schemas, sample data, and verifiers.
"""

import json
import logging
import os
import re
import threading
from typing import Any

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "Snowflake/AgentWorldModel-1K"
HF_REPO_TYPE = "dataset"

DATA_FILES = [
    "gen_scenario.jsonl",
    "gen_tasks.jsonl",
    "gen_db.jsonl",
    "gen_sample.jsonl",
    "gen_envs.jsonl",
    "gen_verifier.jsonl",
    "gen_verifier.pure_code.jsonl",
]


def _default_cache_dir() -> str:
    return os.environ.get("AWM_DATA_DIR", os.path.expanduser("~/.cache/openenv/awm"))


def normalize_scenario_name(scenario: str) -> str:
    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").strip()
    return s


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def _ensure_downloaded(cache_dir: str) -> None:
    """Download data files from HuggingFace if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)

    missing = [f for f in DATA_FILES if not os.path.exists(f"{cache_dir}/{f}")]
    if not missing:
        return

    logger.info(f"Downloading AWM data from {HF_REPO_ID} to {cache_dir} ...")

    for filename in missing:
        logger.info(f"  Downloading {filename} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=filename,
            local_dir=cache_dir,
        )

    logger.info("AWM data download complete.")


class AWMDataLoader:
    """
    Lazy-loading accessor for Agent World Model data files.
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir or _default_cache_dir()
        self._downloaded = False
        self._lock = threading.Lock()

        self._scenarios: dict[str, dict] | None = None
        self._tasks: dict[str, dict] | None = None
        self._envs: dict[str, dict] | None = None
        self._db_schemas: dict[str, dict] | None = None
        self._samples: dict[str, dict] | None = None
        self._verifiers: dict[str, dict[str, list[dict]]] | None = None

    def _ensure_data(self) -> None:
        if not self._downloaded:
            _ensure_downloaded(self._cache_dir)
            self._downloaded = True

    def _build_scenarios(self) -> dict[str, dict]:
        if self._scenarios is None:
            with self._lock:
                if self._scenarios is None:
                    self._ensure_data()
                    raw = _load_jsonl(f"{self._cache_dir}/gen_scenario.jsonl")
                    result: dict[str, dict] = {}
                    for item in raw:
                        key = normalize_scenario_name(item["name"])
                        result[key] = item
                    self._scenarios = result
        return self._scenarios

    def _build_tasks(self) -> dict[str, dict]:
        if self._tasks is None:
            with self._lock:
                if self._tasks is None:
                    self._ensure_data()
                    raw = _load_jsonl(f"{self._cache_dir}/gen_tasks.jsonl")
                    result: dict[str, dict] = {}
                    for item in raw:
                        key = normalize_scenario_name(item["scenario"])
                        result[key] = item
                    self._tasks = result
        return self._tasks

    def _build_envs(self) -> dict[str, dict]:
        if self._envs is None:
            with self._lock:
                if self._envs is None:
                    self._ensure_data()
                    raw = _load_jsonl(f"{self._cache_dir}/gen_envs.jsonl")
                    result: dict[str, dict] = {}
                    for item in raw:
                        key = normalize_scenario_name(item["scenario"])
                        result[key] = item
                    self._envs = result
        return self._envs

    def _build_db_schemas(self) -> dict[str, dict]:
        if self._db_schemas is None:
            with self._lock:
                if self._db_schemas is None:
                    self._ensure_data()
                    raw = _load_jsonl(f"{self._cache_dir}/gen_db.jsonl")
                    result: dict[str, dict] = {}
                    for item in raw:
                        key = normalize_scenario_name(item["scenario"])
                        result[key] = item
                    self._db_schemas = result
        return self._db_schemas

    def _build_samples(self) -> dict[str, dict]:
        if self._samples is None:
            with self._lock:
                if self._samples is None:
                    self._ensure_data()
                    raw = _load_jsonl(f"{self._cache_dir}/gen_sample.jsonl")
                    result: dict[str, dict] = {}
                    for item in raw:
                        key = normalize_scenario_name(item["scenario"])
                        result[key] = item
                    self._samples = result
        return self._samples

    def _build_verifiers(self, verifier_mode: str = "sql") -> dict[str, list[dict]]:
        if verifier_mode not in {"sql", "code"}:
            raise ValueError(
                f"Invalid verifier mode: {verifier_mode!r}, must be 'sql' or 'code'"
            )

        with self._lock:
            if self._verifiers is None:
                self._verifiers = {}

            if verifier_mode not in self._verifiers:
                self._ensure_data()
                if verifier_mode == "sql":
                    raw = _load_jsonl(f"{self._cache_dir}/gen_verifier.jsonl")
                elif verifier_mode == "code":
                    raw = _load_jsonl(f"{self._cache_dir}/gen_verifier.pure_code.jsonl")

                result: dict[str, list[dict]] = {}
                for item in raw:
                    key = normalize_scenario_name(item["scenario"])
                    if key not in result:
                        result[key] = []
                    result[key].append(item)
                self._verifiers[verifier_mode] = result

        return self._verifiers[verifier_mode]

    def list_scenarios(self) -> list[dict[str, Any]]:
        """Return all scenario names, descriptions, and tasks."""
        scenarios = self._build_scenarios()
        tasks = self._build_tasks()
        result = []
        for key, scenario in scenarios.items():
            task_item = tasks.get(key, {})
            task_list = task_item.get("tasks", [])
            result.append(
                {
                    "name": key,
                    "description": scenario.get("description", ""),
                    "num_tasks": len(task_list),
                    "tasks": task_list,
                }
            )
        return result

    def get_env_code(self, scenario: str) -> str:
        """Return the full_code for a scenario."""
        key = normalize_scenario_name(scenario)
        envs = self._build_envs()
        if key not in envs:
            raise ValueError(
                f"Scenario '{scenario}' (normalized: '{key}') not found in gen_envs.jsonl"
            )
        return envs[key]["full_code"]

    def get_db_schema(self, scenario: str) -> dict:
        """Return the db_schema dict for a scenario."""
        key = normalize_scenario_name(scenario)
        schemas = self._build_db_schemas()
        if key not in schemas:
            raise ValueError(f"Scenario '{scenario}' not found in gen_db.jsonl")
        return schemas[key]["db_schema"]

    def get_sample_data(self, scenario: str) -> Any:
        """Return the sample_data for a scenario."""
        key = normalize_scenario_name(scenario)
        samples = self._build_samples()
        if key not in samples:
            raise ValueError(f"Scenario '{scenario}' not found in gen_sample.jsonl")
        return samples[key]["sample_data"]

    def get_tasks(self, scenario: str) -> list[str]:
        """Return the task list for a scenario."""
        key = normalize_scenario_name(scenario)
        tasks = self._build_tasks()
        if key not in tasks:
            return []
        return tasks[key].get("tasks", [])

    def get_verifier(
        self, scenario: str, task_idx: int, verifier_mode: str = "sql"
    ) -> dict | None:
        """Return the verifier entry for a specific scenario + task_idx."""
        key = normalize_scenario_name(scenario)
        verifiers = self._build_verifiers(verifier_mode)
        entries = verifiers.get(key, [])
        for entry in entries:
            if entry.get("task_idx") == task_idx:
                return entry
        return None

    def scenario_exists(self, scenario: str) -> bool:
        key = normalize_scenario_name(scenario)
        return key in self._build_envs()
