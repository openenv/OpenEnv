"""Collect a bounded coding audit using fresh, credential-scoped HF Sandboxes.

This supplemental profile does not replace RFC 008 admission or demonstrate RL
training benefit. Inference runs in the controller; task/verifier data cross only
the explicit file boundary in ``hf``.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from copy import deepcopy
from fractions import Fraction
from pathlib import Path, PurePosixPath

from openenv.validation.runner import _DIGEST_EXCLUDED_DIRS, source_digest
from openenv.validation.runtime.json_values import dumps, loads

from .scoring import assess_coding_audit, validate_audit_config

PROFILE = "harbor-coding-audit-v1"
TARGET_MODEL = "Qwen/Qwen3.8-27B"
TARGET_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _candidate_context_paths(paths):
    message = (
        "generator.context_paths must list at most 32 unique normalized public "
        "files under environment/source/"
    )
    if not isinstance(paths, list) or len(paths) > 32:
        raise ValueError(message)
    result = []
    private = {
        "tests",
        "test",
        "solution",
        "validation",
        "verifier",
        "reference",
        "references",
        "spec",
        "specs",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "private",
    }
    for value in paths:
        if (
            not isinstance(value, str)
            or "\\" in value
            or ":" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(message)
        path = PurePosixPath(value)
        if (
            path.as_posix() != value
            or path.parts[:2] != ("environment", "source")
            or len(path.parts) < 3
            or any(
                part.startswith(".") or part.lower() in private for part in path.parts
            )
            or value in result
        ):
            raise ValueError(message)
        result.append(value)
    return result


def normalize_coding_config(config):
    """Validate operator inputs without launching Jobs, Sandboxes or inference."""
    if not isinstance(config, dict):
        raise ValueError("coding configuration must be an object")
    quality = deepcopy(config.get("validation", {}))
    config = json.loads(dumps(config))
    allowed = {
        "schema_version",
        "profile",
        "target",
        "generator",
        "execution",
        "validation",
        "generation",
        "negative_cases",
        "seeds",
        "cloud",
    }
    if (
        set(config) - allowed
        or type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("unsupported coding audit configuration")
    if config.get("profile", PROFILE) != PROFILE:
        raise ValueError("unsupported coding audit profile")
    config["profile"] = PROFILE
    config["validation"] = validate_audit_config(quality)
    _integer(config["validation"]["reference_repeats"], "reference_repeats", 2, 16)
    _integer(config["validation"]["target_attempts"], "target_attempts", 2, 128)
    for name in ("target", "generator"):
        model = config.get(name, config.get("target"))
        if not isinstance(model, dict):
            raise ValueError(f"{name} model configuration is required")
        model = dict(model)
        model_keys = {
            "model_id",
            "revision",
            "provider",
            "provider_model_id",
            "response_model_aliases",
            "max_steps",
            "max_tokens",
            "max_total_tokens",
            "max_output_bytes",
            "temperature",
            "supports_seed",
            "supports_tools",
            "top_p",
            "enable_thinking",
            "context_paths",
        }
        if name == "target" and "context_paths" in model:
            raise ValueError("target.context_paths is unsupported")
        if set(model) - model_keys:
            raise ValueError(
                f"unknown {name} model option; credentials belong in environment secrets"
            )
        if not re.fullmatch(r"[^\s/@]+/[^\s/@]+", model.get("model_id", "")):
            raise ValueError(f"{name}.model_id must be a canonical Hub repository")
        if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
            raise ValueError(f"{name}.revision must be a full source commit")
        if (
            not isinstance(model.get("provider"), str)
            or not re.fullmatch(r"[a-z0-9-]+", model["provider"])
            or model["provider"] == "auto"
        ):
            raise ValueError(f"{name}.provider must be selected explicitly")
        if "enable_thinking" in model and type(model["enable_thinking"]) is not bool:
            raise ValueError(f"{name}.enable_thinking must be a boolean")
        if "context_paths" in model:
            model["context_paths"] = _candidate_context_paths(model["context_paths"])
        model.setdefault("max_steps", 8)
        model.setdefault("max_tokens", 1024)
        model.setdefault("max_total_tokens", 16384)
        model.setdefault("supports_seed", False)
        model.setdefault("supports_tools", True)
        for key, high in (
            ("max_steps", 128),
            ("max_tokens", 32768),
            ("max_total_tokens", 262144),
        ):
            _integer(model[key], f"{name}.{key}", 1, high)
        config[name] = model
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    execution_keys = {
        "agent_image",
        "verifier_image",
        "submission_paths",
        "private_paths",
        "workdir",
        "flavor",
        "namespace",
        "timeout_seconds",
        "verifier_command",
        "reward_path",
        "reward_key",
        "max_submission_bytes",
        "cleanup_attempts",
        "cleanup_timeout_seconds",
        "max_tool_output_bytes",
        "max_script_bytes",
        "strong_isolation",
        "require_strong_isolation",
    }
    if set(execution) - execution_keys:
        raise ValueError(
            "unknown execution option; task credentials cannot be embedded"
        )
    for key in ("agent_image", "verifier_image"):
        if not _IMAGE.fullmatch(execution.get(key, "")):
            raise ValueError(f"execution.{key} must be an immutable OCI reference")
    if not execution.get("submission_paths"):
        raise ValueError(
            "execution.submission_paths must declare the artifact boundary"
        )
    generation = config.setdefault("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    if set(generation) - {"candidate_count"}:
        raise ValueError("unknown generation option")
    generation.setdefault("candidate_count", 2)
    _integer(generation["candidate_count"], "candidate_count", 1, 16)
    negatives = config.get("negative_cases", [])
    expected = config["validation"]["required_negative_cases"]
    if (
        not isinstance(negatives, list)
        or any(
            not isinstance(case, dict) or set(case) != {"id", "script"}
            for case in negatives
        )
        or [case["id"] for case in negatives] != expected
        or any(
            not isinstance(case["script"], str) or not case["script"].strip()
            for case in negatives
        )
    ):
        raise ValueError(
            "negative_cases must supply the trusted required IDs and scripts in order"
        )
    attempts = config["validation"]["target_attempts"]
    seeds = config.setdefault("seeds", list(range(attempts)))
    if (
        not isinstance(seeds, list)
        or len(seeds) != attempts
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "one distinct seed/attempt identifier per target attempt is required"
        )
    for seed in seeds:
        _integer(seed, "seed", 0, 2**32 - 1)
    cloud = config.setdefault("cloud", {})
    if not isinstance(cloud, dict):
        raise ValueError("cloud must be an object")
    if set(cloud) - {
        "results_repo",
        "namespace",
        "flavor",
        "timeout_seconds",
        "max_sandboxes",
        "max_inference_calls",
    }:
        raise ValueError("unknown cloud option")
    cloud.setdefault("max_sandboxes", 100)
    cloud.setdefault("max_inference_calls", 256)
    _integer(cloud["max_sandboxes"], "max_sandboxes", 1, 10000)
    _integer(cloud["max_inference_calls"], "max_inference_calls", 1, 10000)
    return config


def coding_budget(config, *, has_reference=True):
    """A conservative, finite launch ceiling, including leakage probes."""
    repeats = config["validation"]["reference_repeats"]
    candidates = config["generation"]["candidate_count"]
    references = candidates + int(has_reference)
    executions = (
        1 + references * repeats + len(config["negative_cases"]) + len(config["seeds"])
    )
    return {
        "sandbox_jobs_max": 2 * executions + 6,
        "inference_calls_max": candidates
        + len(config["seeds"]) * config["target"]["max_steps"],
        "generated_candidates": candidates,
        "target_attempts": len(config["seeds"]),
    }


def _credentials():
    names = ("HF_TOKEN", "HF_INFERENCE_TOKEN", "HF_SANDBOX_TOKEN")
    values = {name: os.environ.get(name) for name in names}
    if not all(values.values()):
        raise ValueError(
            "HF_TOKEN, HF_INFERENCE_TOKEN and HF_SANDBOX_TOKEN are required"
        )
    if len(set(values.values())) != len(values):
        raise ValueError(
            "artifact, inference and sandbox credentials must be distinct and separately scoped"
        )
    return values


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value) + "\n")


def live_provider_id(info, provider):
    """Read both Hub SDK provider-map representations without selecting an alias."""
    mappings = info.inference_provider_mapping
    if isinstance(mappings, dict):
        candidates = [mappings[provider]] if provider in mappings else []
    elif isinstance(mappings, list):
        candidates = [
            entry
            for entry in mappings
            if (
                entry.get("provider")
                if isinstance(entry, dict)
                else getattr(entry, "provider", None)
            )
            == provider
        ]
    else:
        candidates = []
    if len(candidates) != 1:
        raise ValueError("selected model/provider mapping is missing or ambiguous")
    entry = candidates[0]
    status = (
        entry.get("status")
        if isinstance(entry, dict)
        else getattr(entry, "status", None)
    )
    identity = (
        entry.get("provider_id")
        if isinstance(entry, dict)
        else getattr(entry, "provider_id", None)
    )
    if status != "live" or not isinstance(identity, str) or not identity:
        raise ValueError("selected model/provider is not live")
    return identity


def validate_coding_task(task_root, config):
    """Check supported native task contracts locally, before controller spending.

    Returns normalized audit configuration. Passing this check is preparation,
    not evidence that the task, image, or grader passes runtime validation.
    """
    config = normalize_coding_config(config)
    task_root = Path(task_root).resolve()
    task_config = tomllib.loads((task_root / "task.toml").read_text())
    if any((task_root / "steps").glob("*")):
        raise ValueError("multi-step Harbor tasks are not supported by this profile")
    for section in ("environment", "agent", "verifier", "solution"):
        section_config = task_config.get(section, {})
        if not isinstance(section_config, dict):
            raise ValueError("task phase configuration must be an object")
        if section_config.get("environment"):
            raise ValueError("separate phase environment contracts are unsupported")
        if section_config.get("allow_internet") is False:
            raise ValueError("non-public phase network contracts are unsupported")
        if section_config.get("allowed_hosts"):
            raise ValueError("network allowlist contracts are unsupported")
        if section_config.get("env"):
            raise ValueError(
                "task environment/credential substitutions are unsupported"
            )
        if section_config.get("network_mode", "public") != "public":
            raise ValueError("non-public phase network contracts are unsupported")
    environment = task_config.get("environment", {})
    if task_config.get("verifier", {}).get("collect"):
        raise ValueError("verifier collection commands are unsupported")
    if environment.get("workdir") not in (
        None,
        config["execution"].get("workdir", "/app"),
    ):
        raise ValueError("source and audit workdirs disagree")
    if any(
        environment.get(key)
        for key in ("gpus", "tpu", "mcp_servers", "healthcheck", "skills_dir")
    ):
        raise ValueError(
            "accelerator, healthcheck, skills and external MCP requirements are unsupported"
        )
    if any(
        (task_root / "environment" / name).exists()
        for name in (
            "compose.yaml",
            "compose.yml",
            "docker-compose.yaml",
            "docker-compose.yml",
        )
    ):
        raise ValueError("multi-service Compose tasks are unsupported")
    if (
        not (task_root / "instruction.md").is_file()
        or not (task_root / "tests/test.sh").is_file()
    ):
        raise ValueError("a native instruction and verifier are required")
    budget = coding_budget(
        config, has_reference=(task_root / "solution/solve.sh").is_file()
    )
    if (
        budget["sandbox_jobs_max"] > config["cloud"]["max_sandboxes"]
        or budget["inference_calls_max"] > config["cloud"]["max_inference_calls"]
    ):
        raise ValueError(
            "configured launch budget is too small for the requested audit"
        )
    return config


def run_coding_validation(task_root, config, output_dir, *, backend=None):
    """Generate, challenge, replay and measure a single prepared native task.

    Every result is bound to frozen task/verifier bytes and immutable requested
    image references. Provider source revisions are recorded without pretending
    they pin the provider's served weights. Outputs must be a new directory.
    """
    config = validate_coding_task(task_root, config)
    task_root, output_dir = Path(task_root).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(task_root) and not any(
        part in _DIGEST_EXCLUDED_DIRS
        for part in output_dir.relative_to(task_root).parts
    ):
        raise ValueError("audit output must be outside the digested task tree")
    secrets = [
        value
        for name in ("HF_TOKEN", "HF_INFERENCE_TOKEN", "HF_SANDBOX_TOKEN")
        if (value := os.environ.get(name))
    ]
    if any(secret in dumps(config) for secret in secrets):
        raise ValueError("credentials may not be embedded in audit metadata")
    original_digest = source_digest(task_root)
    budget = coding_budget(
        config, has_reference=(task_root / "solution/solve.sh").is_file()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence = {
        "task_digest": original_digest,
        "verifier_digest": None,
        "agent_image": config["execution"]["agent_image"],
        "verifier_image": config["execution"]["verifier_image"],
        "target": {
            "model": config["target"]["model_id"],
            "revision": config["target"]["revision"],
            "provider": config["target"]["provider"],
        },
        "references": [],
        "trials": [],
        "leakage_probes": [],
        "generation": [],
        "budget": budget,
        "configuration_digest": hashlib.sha256(dumps(config).encode()).hexdigest(),
        "requested_configuration_digest": hashlib.sha256(
            dumps(config).encode()
        ).hexdigest(),
        "limitations": [
            "Supplemental coding audit; full RFC 008 qualification is not established.",
            "Provider model source SHA does not attest served weights or deterministic sampling.",
            "Image identities are control-plane immutable references, not runtime image attestations.",
            "Sandbox RPC exposes its execution credential to the workload server; only reviewed workloads are supported.",
            "Separate verifier Jobs do not prove submitted code cannot read or tamper with its grader.",
        ],
    }

    def sanitize(value):
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {sanitize(key): sanitize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [sanitize(child) for child in value]
        return value

    def artifact(relative, data):
        if isinstance(data, str):
            data = data.encode()
        digest = hashlib.sha256(data).hexdigest()
        if any(secret.encode() in data for secret in secrets):
            evidence["leakage_probes"].append(
                {"name": "credential_in_artifact", "passed": False, "error": None}
            )
            return {"sha256": digest, "omitted": "credential detected"}
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": relative, "sha256": digest}

    def persist():
        evidence["configuration_digest"] = hashlib.sha256(
            dumps(config).encode()
        ).hexdigest()
        if any(secret in dumps(evidence) for secret in secrets) and not any(
            probe.get("name") == "credential_in_metadata"
            for probe in evidence["leakage_probes"]
        ):
            evidence["leakage_probes"].append(
                {"name": "credential_in_metadata", "passed": False, "error": None}
            )
        _write(output_dir / "evidence.json", sanitize(evidence))

    try:
        from .hf import _local_file

        def copy_artifact(source, destination):
            raw, _ = _local_file(source, Path(source).stat().st_size)
            Path(destination).write_bytes(raw)
            shutil.copystat(source, destination)

        with tempfile.TemporaryDirectory(prefix="openenv-coding-frozen-") as temporary:
            frozen = (Path(temporary) / "task").resolve()
            shutil.copytree(
                task_root,
                frozen,
                symlinks=True,
                copy_function=copy_artifact,
                ignore=shutil.ignore_patterns(*_DIGEST_EXCLUDED_DIRS),
            )
            if source_digest(frozen) != original_digest:
                raise ValueError("task changed during snapshot")
            evidence["verifier_digest"] = source_digest(frozen / "tests")
            if backend is None:
                from .hf import HFTaskBackend

                credentials = _credentials()
                # Provider preflight is read-only and precedes every paid sandbox.
                from huggingface_hub import HfApi

                api = HfApi(token=credentials["HF_INFERENCE_TOKEN"])
                for model in (config["target"], config["generator"]):
                    info = api.model_info(model["model_id"], revision=model["revision"])
                    if info.sha != model["revision"]:
                        raise ValueError("model source revision could not be verified")
                    current = api.model_info(
                        model["model_id"], expand=["inferenceProviderMapping"]
                    )
                    model.setdefault(
                        "provider_model_id",
                        live_provider_id(current, model["provider"]),
                    )
                backend = HFTaskBackend(
                    frozen, config["execution"], token=credentials["HF_SANDBOX_TOKEN"]
                )

            def trial(role, case_id, script=None, seed=None):
                if source_digest(frozen) != original_digest:
                    raise ValueError("frozen task source changed")
                number = len(evidence["trials"])
                submission = (
                    backend.solve(config["target"], seed, role="target")
                    if role == "target"
                    else backend.execute_script(script, role)
                )
                row = {
                    "role": role,
                    "case_id": case_id,
                    "task_digest": original_digest,
                    "verifier_digest": evidence["verifier_digest"],
                    "image_ref": submission.get("image_ref"),
                    "submission_digest": submission.get("submission_digest"),
                    "sandbox_id": submission.get("sandbox_id"),
                    "error": submission["error"]
                    if "error" in submission
                    else "execution error status missing",
                    "cleanup_verified": submission.get("cleanup_verified") is True,
                    "latency_seconds": submission.get("latency_seconds"),
                    "usage": submission.get("usage"),
                    "seed": seed,
                    "model_metadata": submission.get("model_metadata"),
                    "truncated": submission.get("truncated", False),
                    "termination_reason": submission.get("termination_reason"),
                }
                row["transcript"] = artifact(
                    f"trials/{number:04d}/transcript.json",
                    dumps(submission.get("transcript", [])),
                )
                for index, (name, data) in enumerate(
                    sorted(submission.get("files", {}).items())
                ):
                    item = artifact(f"trials/{number:04d}/submission/{index:04d}", data)
                    row.setdefault("submission_files", []).append(
                        {"destination": name, **item}
                    )
                if row["error"] is None and row["cleanup_verified"]:
                    grade = backend.grade(submission)
                    row["grader_transcript"] = artifact(
                        f"trials/{number:04d}/grader-transcript.json",
                        dumps(grade.get("transcript", [])),
                    )
                    row.update(
                        {
                            "grade_sandbox_id": grade.get("grade_sandbox_id"),
                            "verifier_image_ref": grade.get("verifier_image_ref"),
                            "reward": grade.get("reward"),
                            "error": grade["error"]
                            if "error" in grade
                            else "grader error status missing",
                            "cleanup_verified": grade.get("cleanup_verified") is True,
                        }
                    )
                    if grade.get("rawreward"):
                        row["raw_reward"] = artifact(
                            f"trials/{number:04d}/reward.raw", grade["rawreward"]
                        )
                        try:
                            raw_value = loads(grade["rawreward"])
                            if isinstance(raw_value, dict):
                                raw_value = raw_value[
                                    config["execution"].get("reward_key", "reward")
                                ]
                            if isinstance(raw_value, (bool, str)) or isinstance(
                                row["reward"], bool
                            ):
                                raise ValueError("reward is not numeric")
                            if Fraction(str(raw_value)) != Fraction(str(row["reward"])):
                                raise ValueError(
                                    "raw reward and parsed reward disagree"
                                )
                        except (ValueError, TypeError, KeyError, ZeroDivisionError):
                            row.update(
                                error=row["error"]
                                if row["error"] is not None
                                else "raw reward fidelity check failed",
                                reward=None,
                            )
                    elif row["error"] is None:
                        row.update(error="raw reward evidence missing", reward=None)
                if role == "target":
                    row.update(
                        {
                            "model": config["target"]["model_id"],
                            "revision": config["target"]["revision"],
                            "provider": config["target"]["provider"],
                        }
                    )
                evidence["trials"].append(row)
                persist()
                if not row["cleanup_verified"]:
                    raise ValueError("stop collection after unverified cleanup")
                return row

            trial("nop", "nop", "true")
            candidates = []
            provided = frozen / "solution/solve.sh"
            if provided.is_file():
                candidates.append(("provided", "provided", provided.read_text()))
            for index in range(config["generation"]["candidate_count"]):
                generated = backend.generate_candidate(config["generator"], index)
                script = generated.get("script", "")
                entry = {
                    key: value for key, value in generated.items() if key != "script"
                }
                entry["artifact"] = artifact(f"candidates/generated-{index}.sh", script)
                evidence["generation"].append(entry)
                if generated.get("error") or not script.strip():
                    evidence["generation_error"] = True
                    continue
                candidates.append((f"generated-{index}", "generated", script))
            for case_id, origin, script in candidates:
                records = [
                    trial("reference", case_id, script)
                    for _ in range(config["validation"]["reference_repeats"])
                ]
                evidence["references"].append(
                    {
                        "case_id": case_id,
                        "origin": origin,
                        "submission_digest": records[0].get("submission_digest"),
                        "candidate_script_digest": hashlib.sha256(
                            script.encode()
                        ).hexdigest(),
                    }
                )
            for case in config["negative_cases"]:
                trial("negative", case["id"], case["script"])
            evidence["leakage_probes"].extend(backend.leakage_probes())
            if evidence.get("generation_error"):
                evidence["leakage_probes"].append(
                    {
                        "name": "generation_complete",
                        "passed": False,
                        "error": "candidate generation incomplete",
                    }
                )
            persist()
            screening = assess_coding_audit(config["validation"], evidence)
            if all(
                check["status"] == "pass"
                for name, check in screening["checks"].items()
                if name != "target_difficulty"
            ):
                for index, seed in enumerate(config["seeds"]):
                    trial("target", f"target-{index}", seed=seed)
                # Include target transcripts in the final leakage measurement.
                names = {
                    "private_paths_absent",
                    "fresh_state_absent",
                    "canary_not_in_transcript",
                }
                evidence["leakage_probes"] = [
                    probe
                    for probe in evidence["leakage_probes"]
                    if probe.get("name") not in names
                ]
                evidence["leakage_probes"].extend(backend.leakage_probes())
            else:
                evidence["target_not_run_reason"] = (
                    "reference/verifier/reliability/leakage screening did not pass"
                )
            if (
                source_digest(task_root) != original_digest
                or source_digest(frozen) != original_digest
            ):
                raise ValueError("task source changed during audit")
    except (Exception, KeyboardInterrupt) as exc:
        # Provider exceptions may include request URLs or credentials.
        evidence["collection_error"] = type(exc).__name__
        evidence["leakage_probes"].append(
            {
                "name": "collection_complete",
                "passed": False,
                "error": type(exc).__name__,
            }
        )
    persist()
    report = assess_coding_audit(config["validation"], evidence)
    report.update(
        {
            "task_digest": original_digest,
            "verifier_digest": evidence["verifier_digest"],
            "target": evidence["target"],
            "budget": budget,
            "limitations": evidence["limitations"],
            "configuration_digest": evidence["configuration_digest"],
        }
    )
    report["checks"]["target_difficulty"]["truncated_attempts"] = sum(
        trial.get("truncated") is True
        for trial in evidence["trials"]
        if trial["role"] == "target"
    )
    report = sanitize(report)
    _write(output_dir / "config.json", sanitize(config))
    _write(output_dir / "report.json", report)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(path.read_bytes()).hexdigest()
            + "  "
            + path.relative_to(output_dir).as_posix()
            + "\n"
            for path in files
        )
    )
    return report
