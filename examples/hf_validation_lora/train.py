# SPDX-License-Identifier: BSD-3-Clause
"""Bounded online GRPO on the real ScottWuMath four-card-24 environment.

Root installs the frozen Linux/Python 3.11 training lock, creates three PRIVATE
Hub repos, and qualifies/provisions the owned source-pinned Sandbox. Run this
plain Python file (also suitable as an inline Jobs script) on one L4, timeout 1h:

    OUTPUT_REPO=... ARTIFACT_REPO=... TRACKIO_SPACE_ID=... SBX_JOB_ID=... \
        ENV_TRANSPORT=exec python train.py

HF_TOKEN must be the SAME credential used to create the Sandbox. It stays in
memory/environment only. Default exec transport calls the root-provided bridge
/opt/score_batch.py with /opt/mathenv/bin/python; that bridge uses the REAL
server's local WebSocket on port 8012. No proxy requests occur in exec mode.
ENV_TRANSPORT=direct uses GenericEnvClient through the authenticated SDK proxy.
SBX_NAMESPACE (default OUTPUT_REPO owner) and SBX_PORT (default 8012) are optional.

Objective: unchanged terminal shaped reward from indexed reset/one expression
step; group-normalized vanilla GRPO, beta=0, no auxiliary rewards. Missing tags
submit an empty expression, whose terminal reward is still issued by the server.
Before any LoRA is attached: evaluate the base model, then sample 32 TRAIN groups
of 4. Require >=2 groups with raw reward range >=1e-6 and >=25% server-valid
expressions. Otherwise persist no_grpo_learning_signal and STOP; no automatic
curriculum or reference-solution fallback exists. All 64 train tasks remain
eligible for at most 20 updates (not necessarily all visited).

Held-out test[0:16] uses greedy + seeds 7001/7002, identically before/after LoRA.
Checkpoints contain adapters, not serialized trainer arguments/credentials or
resume state. report.json, manifest.json and traces.jsonl are uploaded under a
unique run directory in ARTIFACT_REPO; adapters go to OUTPUT_REPO. The controller
owns Sandbox termination and independent qualification/provenance reporting.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import statistics
import sys
import sysconfig
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ENVIRONMENT = "FineEnvs/ScottWuMath"
ENV_REVISION = "4de8125d314fca0a3f137fa1bab04b966a3efc15"
PINS = {
    "torch": "2.8.0",
    "trl": "1.17.1",
    "transformers": "5.22.1",
    "peft": "0.21.0",
    "accelerate": "1.15.0",
    "datasets": "5.1.0",
    "trackio": "0.45.0",
    "huggingface-hub": "2.4.0",
    "openenv": "0.11.0",
    "fastmcp": "3.4.11",
}
GROUP_SIZE = 4
PILOT_GROUPS = 32
MIN_CONTRAST = 1e-6
MIN_VALID_FRACTION = (
    0.25  # Pre-registered before trials; invalid expressions keep env zero.
)
EVAL_SEEDS = (None, 7001, 7002)  # None means greedy, still reset RNG deterministically.


class NoLearningSignal(Exception):
    """An evidence-backed early stop, rather than a failed infrastructure trial."""


def require_fresh_private_trackio(api, space_id):
    """A new private dashboard must not reuse previously mounted public storage."""
    info = api.space_info(space_id)
    runtime = api.get_space_runtime(space_id)
    if (
        not info.private
        or info.sdk != "gradio"
        or runtime.volumes
        or (info.runtime is not None and info.runtime.volumes)
        or {file.rfilename for file in info.siblings} - {".gitattributes", "README.md"}
    ):
        raise RuntimeError(
            "Trackio requires a fresh private Gradio Space without mounted storage"
        )
    return {"private": True, "fresh": True, "mounted_storage": []}


def install_trackio_dashboard(api, space_id):
    # Trackio skips deployment when the required pre-created Space already exists.
    from huggingface_hub import metadata_update

    metadata_update(
        space_id,
        {"hf_oauth": True, "hf_oauth_scopes": ["write-repos"]},
        repo_type="space",
        token=api.token,
        overwrite=True,
    )
    for name, content in (
        ("requirements.txt", "trackio==0.45.0\n"),
        ("app.py", "import trackio\n\ntrackio.show()\n"),
    ):
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=name,
            repo_id=space_id,
            repo_type="space",
        )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def exception_frames(error):
    """Locate failures without retaining exception messages, paths or local values."""
    return [
        {
            "file": Path(frame.filename).name,
            "line": frame.lineno,
            "function": frame.name,
        }
        for frame in traceback.extract_tb(error.__traceback__)[-8:]
    ]


def optimizer_observation(grad_norm, skipped, counter_advanced, changed_tensors):
    """Preserve AMP overflow recovery without counting it as a weight update."""
    finite = math.isfinite(grad_norm)
    if not finite and not skipped:
        raise RuntimeError("Non-finite gradients reached an unskipped optimizer step")
    if skipped and (counter_advanced or changed_tensors):
        raise RuntimeError(
            "A skipped optimizer step changed optimizer or adapter state"
        )
    return {
        "grad_norm": grad_norm if finite else None,
        "gradients_finite": finite,
        "counter_advanced": counter_advanced,
        "step_skipped": skipped,
        "changed_tensors": changed_tensors,
    }


def counters_advanced(previous, current):
    """Inspect each parameter's counter, including one below the maximum."""
    return any(step > previous.get(parameter, 0) for parameter, step in current.items())


def effective_update(event):
    return (
        event["gradients_finite"]
        and event["grad_norm"] > 0
        and not event["step_skipped"]
        and event["counter_advanced"]
        and event["changed_tensors"] > 0
        and event["group_std"] > 0
    )


def require_c_compiler():
    compiler = (
        shutil.which(os.environ["CC"])
        if "CC" in os.environ
        else (shutil.which("gcc") or shutil.which("clang"))
    )
    if compiler is None:
        raise RuntimeError("training image lacks the C compiler required by Triton")
    return compiler


def require_python_headers():
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if not header.is_file():
        raise RuntimeError(
            "training interpreter lacks Python.h; use the pinned training image interpreter"
        )
    return str(header)


def training_precision(torch):
    # TRL's BF16->FP32 backward GEMM needs native BF16, not T4 emulation.
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )


def training_preflight(torch, dtype):
    compiler = require_c_compiler()
    header = require_python_headers()
    from trl.kernels.chunked_logprob import ChunkedLogProbFunction
    from trl.trainer.utils import selective_log_softmax

    logits = torch.randn((1, 4, 151936), device="cuda", dtype=dtype, requires_grad=True)
    index = torch.randint(0, logits.shape[-1], (1, 4), device="cuda")
    logprobs = selective_log_softmax(logits, index)
    expected = (
        logits.float().log_softmax(-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(logprobs.float(), expected, atol=0.03, rtol=0.003)
    logprobs.sum().backward()
    if not torch.isfinite(logits.grad).all():
        raise RuntimeError("training loss preflight produced non-finite gradients")
    # GRPO uses this fused head backward, not just selective_log_softmax.
    hidden = torch.randn((4, 32), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((151936, 32), device="cuda", dtype=dtype) * 0.01
    targets = index.flatten()
    logprobs = ChunkedLogProbFunction.apply(
        hidden, weight, None, targets, 1.0, 4096, None, 1.0, ("entropy",)
    )[0]
    reference_hidden = hidden.detach().float().requires_grad_()
    reference = (reference_hidden @ weight.float().t()).log_softmax(-1)
    reference = reference.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(logprobs.float(), reference, atol=0.03, rtol=0.003)
    logprobs.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(
        hidden.grad.float(), reference_hidden.grad, atol=0.003, rtol=0.03
    )
    return {
        "compiler": compiler,
        "python_header": header,
        "python_version": sys.version.split()[0],
        "cuda_runtime": torch.version.cuda,
        "precision": str(dtype),
        "logits_shape": list(logits.shape),
        "backward": "passed",
        "fused_head_backward": "passed",
    }


def extract_expression(completion):
    """Last answer outside thinking; also discard an unfinished thinking block."""
    text = re.sub(
        r"<think>.*?(?:</think>|$)", "", completion, flags=re.DOTALL | re.IGNORECASE
    )
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return answers[-1].strip() if answers else ""


def policy_prompt(observation):
    """Allowlist only the two pre-submission fields; never serialize an observation."""
    if observation.get("solution") or observation.get("solution_steps"):
        raise ValueError("Reset exposed a reference solution")
    return validate_prompt(
        [
            {"role": "system", "content": observation["system_prompt"]},
            {"role": "user", "content": observation["prompt"]},
        ]
    )


def validate_prompt(messages):
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [m.get("role") for m in messages] != ["system", "user"]
        or any(set(m) != {"role", "content"} for m in messages)
        or any(not isinstance(m["content"], str) or not m["content"] for m in messages)
    ):
        raise ValueError("Expected only public system/user messages")
    hand_identity(messages)  # Reject accidentally selected year/six-card tasks.
    return messages


def hand_identity(messages):
    text = messages[1]["content"]
    values = re.search(r"^Values: (\d+(?:, \d+)*)$", text, re.MULTILINE)
    target = re.search(r"^Target: (\d+)$", text, re.MULTILINE)
    ranks = list(map(int, values[1].split(", "))) if values else []
    if len(ranks) != 4 or any(not 1 <= v <= 13 for v in ranks) or not target:
        raise ValueError("Expected a public four-card prompt")
    if int(target[1]) != 24:
        raise ValueError("Expected target 24")
    return tuple(sorted(ranks))


def signal_gate(groups, expected_groups=8):
    """Frozen train-only smoke threshold; never tune it using test returns."""
    if len(groups) != expected_groups or any(len(g) != GROUP_SIZE for g in groups):
        raise ValueError("Pilot group count or size differs from the frozen plan")
    stats = []
    for group in groups:
        rewards = [r["reward"] for r in group]
        stats.append(
            {
                "range": max(rewards) - min(rewards),
                "std": statistics.pstdev(rewards),
                "variance": statistics.pvariance(rewards),
                "valid": sum(r["valid"] for r in group),
            }
        )
    contrasting = sum(s["range"] + 1e-12 >= MIN_CONTRAST for s in stats)
    valid_fraction = sum(s["valid"] for s in stats) / (expected_groups * GROUP_SIZE)
    return {
        "passed": contrasting >= 2 and valid_fraction >= MIN_VALID_FRACTION,
        "contrasting_groups": contrasting,
        "valid_fraction": valid_fraction,
        "minimum_raw_range": MIN_CONTRAST,
        "planned_groups": expected_groups,
        "minimum_contrasting_groups": 2,
        "minimum_valid_fraction": MIN_VALID_FRACTION,
        "groups": stats,
    }


class RemoteEnvironment:
    """One owned Sandbox, sequential complete episodes, no server imports."""

    def __init__(self, sandbox, transport, port=8012):
        if transport not in {"exec", "direct"}:
            raise ValueError("ENV_TRANSPORT must be exec or direct")
        self.sandbox, self.transport, self.port = sandbox, transport, port

    def _bridge(self, payload):
        result = self.sandbox.run(
            ["/opt/mathenv/bin/python", "/opt/score_batch.py"],
            stdin=json.dumps(payload),
            timeout=50,
        )
        if result.exit_code != 0:
            raise RuntimeError("Remote bridge failed")
        return json.loads(result.stdout)

    def _client(self):
        from openenv.core.generic_client import GenericEnvClient

        return GenericEnvClient(
            base_url=self.sandbox.proxy_url_for(self.port, "/"),
            headers=self.sandbox.proxy_headers,
            connect_timeout_s=15,
            message_timeout_s=30,
        ).sync()

    def tasks(self, split, indices):
        if self.transport == "exec":
            rows = self._bridge(
                {"operation": "tasks", "split": split, "indices": indices}
            )["tasks"]
        else:
            rows = []
            with self._client() as client:
                for index in indices:
                    result = client.reset(
                        split=split, index=index, reward_mode="shaped"
                    )
                    if result.done:
                        raise RuntimeError("Reset was terminal")
                    if result.observation["task_id"] != f"{split}-24-{index:04d}":
                        raise RuntimeError("Unexpected task identity")
                    rows.append(
                        {"index": index, "prompt": policy_prompt(result.observation)}
                    )
        if [r["index"] for r in rows] != indices:
            raise RuntimeError("Task batch alignment failed")
        # Return only public prompts + controller metadata; ignore arbitrary bridge fields.
        return [
            {
                "index": row["index"],
                "split": split,
                "episode_key": f"{split}-24-{row['index']:04d}",
                "prompt": validate_prompt(row["prompt"]),
            }
            for row in rows
        ]

    def score(self, split, indices, expressions):
        if len(indices) != len(expressions):
            raise ValueError("Score batch alignment failed")
        if self.transport == "exec":
            payload = self._bridge(
                {
                    "operation": "score",
                    "split": split,
                    "indices": indices,
                    "expressions": expressions,
                }
            )
        else:
            payload = {"rewards": [], "task_ids": [], "valid": []}
            with self._client() as client:
                for index, expression in zip(indices, expressions, strict=True):
                    client.reset(split=split, index=index, reward_mode="shaped")
                    result = client.step({"expression": expression})
                    if not result.done or result.reward is None:
                        raise RuntimeError("Expected one terminal environment reward")
                    obs = result.observation
                    payload["rewards"].append(result.reward)
                    payload["task_ids"].append(obs["task_id"])
                    payload["valid"].append(obs["result"]["valid"])
        expected_ids = [f"{split}-24-{i:04d}" for i in indices]
        if payload["task_ids"] != expected_ids:
            raise RuntimeError("Scored task identity mismatch")
        if any(len(payload[k]) != len(indices) for k in ("rewards", "valid")):
            raise RuntimeError("Scored reward/valid batch alignment failed")
        outcomes = []
        for reward, valid in zip(payload["rewards"], payload["valid"], strict=True):
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise RuntimeError("Non-numeric terminal reward")
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise RuntimeError("Invalid terminal reward")
            if not isinstance(valid, bool):
                raise RuntimeError("Missing server grader validity")
            outcomes.append(
                {"reward": float(reward), "valid": valid, "exact": reward == 1}
            )
        return outcomes


def paired_summary(traces):
    phases = {}
    for phase in ("baseline", "adapter"):
        rows = [
            r for r in traces if r["phase"] == phase and r.get("reward") is not None
        ]
        phases[phase] = {
            "planned": 16 * len(EVAL_SEEDS),
            "completed": len(rows),
            "mean_raw_reward": statistics.mean(r["reward"] for r in rows)
            if rows
            else None,
            "exact_rate": statistics.mean(r["exact"] for r in rows) if rows else None,
            "valid_rate": statistics.mean(r["valid"] for r in rows) if rows else None,
        }
    base = {
        (r["episode_key"], r["seed"]): r["reward"]
        for r in traces
        if r["phase"] == "baseline" and r.get("reward") is not None
    }
    differences = [
        r["reward"] - base[(r["episode_key"], r["seed"])]
        for r in traces
        if r["phase"] == "adapter"
        and r.get("reward") is not None
        and (r["episode_key"], r["seed"]) in base
    ]
    phases["paired"] = {
        "completed": len(differences),
        "mean_raw_reward_difference": statistics.mean(differences)
        if differences
        else None,
    }
    return phases


def main():
    started = time.monotonic()
    deadline = started + 3000  # Reserve ten minutes within the root's one-hour Job.
    run_id = "scottwu-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(os.environ.get("RUN_DIR", "/tmp")) / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "run_id": run_id,
        "objective": "raw terminal shaped reward; vanilla GRPO, group scaling, beta=0",
        "source_scope": "controller-qualified pinned-source server; no provider provenance claim",
        "environment": ENVIRONMENT,
        "environment_revision": ENV_REVISION,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "max_steps": 20,
        "pilot_planned_completions": PILOT_GROUPS * GROUP_SIZE,
        "hyperparameters": {
            "loss_type": "grpo",
            "scale_rewards": "group",
            "beta": 0.0,
            "learning_rate": 5e-5,
            "lr_scheduler_type": "constant",
            "weight_decay": 0.0,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj", "v_proj"],
            "num_generations": 4,
            "max_completion_length": 128,
            "enable_thinking": False,
        },
        "optimizer_evidence": [],
        "training_groups": [],
    }
    traces = []
    api = sandbox = trainer = model = tokenizer = trackio = None
    tracker = None
    initial = {}
    repos = {}
    safe_to_upload = False
    failed = False

    def check_time(reserve=0):
        if time.monotonic() + reserve >= deadline:
            raise TimeoutError("Soft runtime limit reached")

    def write_report():
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["evaluation"] = paired_summary(traces)
        temp = out / "report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(out / "report.json")

    def persist():
        write_report()
        if safe_to_upload:
            # Explicit allowlist: no environment dump, tokens, headers or trainer pickles.
            commit = api.upload_folder(
                repo_id=repos["ARTIFACT_REPO"],
                repo_type="dataset",
                folder_path=str(out),
                path_in_repo=run_id,
                allow_patterns=[
                    "report.json",
                    "manifest.json",
                    "traces.jsonl",
                    "train.py",
                ],
                commit_message=f"Persist {run_id} {report['status']}",
            )
            report["artifact_commit"] = commit.oid
            write_report()

    def append_trace(row):
        traces.append(row)
        with (out / "traces.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def save_adapter():
        if initial and model is not None:
            final = {
                n: p.detach().float().cpu()
                for n, p in model.named_parameters()
                if n in initial
            }
            report["changed_adapter_tensors"] = sum(
                not torch.equal(initial[n], final[n]) for n in initial
            )
            report["final_adapter_sha256"] = hashlib.sha256(
                b"".join(final[n].numpy().tobytes() for n in sorted(final))
            ).hexdigest()
            directory = out / "adapter"
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            commit = api.upload_folder(
                repo_id=repos["OUTPUT_REPO"],
                repo_type="model",
                folder_path=str(directory),
                commit_message=f"Save {run_id} adapter",
            )
            report["adapter_commit"] = commit.oid
            report["adapter_files_sha256"] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.iterdir()
                if p.is_file()
            }

    def interrupted(signum, frame):
        raise InterruptedError("Job interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    try:
        for name in (
            "OUTPUT_REPO",
            "ARTIFACT_REPO",
            "TRACKIO_SPACE_ID",
            "SBX_JOB_ID",
            "HF_TOKEN",
        ):
            if not os.environ.get(name):
                raise ValueError("Missing required runtime environment variable")
        repos = {
            k: os.environ[k]
            for k in ("OUTPUT_REPO", "ARTIFACT_REPO", "TRACKIO_SPACE_ID")
        }
        report["repos"] = repos
        report["sandbox_job_id"] = os.environ["SBX_JOB_ID"]
        report["transport"] = os.environ.get("ENV_TRANSPORT", "exec")
        report["versions"] = {name: importlib.metadata.version(name) for name in PINS}
        if any(report["versions"][k].split("+")[0] != v for k, v in PINS.items()):
            raise RuntimeError(
                "Worker dependencies differ from the frozen training lock"
            )

        import torch
        import trackio
        from datasets import Dataset
        from huggingface_hub import HfApi, Sandbox, snapshot_download
        from peft import get_peft_model, LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            set_seed,
            TrainerCallback,
        )
        from trl import GRPOConfig, GRPOTrainer

        api = HfApi(token=os.environ["HF_TOKEN"])
        for key, kind in (
            ("OUTPUT_REPO", "model"),
            ("ARTIFACT_REPO", "dataset"),
            ("TRACKIO_SPACE_ID", "space"),
        ):
            if not api.repo_info(repos[key], repo_type=kind).private:
                raise RuntimeError("All output repositories must already be private")
        report["trackio_preflight"] = require_fresh_private_trackio(
            api, repos["TRACKIO_SPACE_ID"]
        )
        safe_to_upload = True
        if globals().get("__file__") and Path(__file__).is_file():
            source = Path(__file__).read_bytes()
            (out / "train.py").write_bytes(source)
            report["script_sha256"] = hashlib.sha256(source).hexdigest()
        persist()
        report["status"] = "training_preflight"
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("This example requires exactly one CUDA GPU")
        report["gpu"] = torch.cuda.get_device_name(0)
        dtype = training_precision(torch)
        bf16 = dtype == torch.bfloat16
        report["training_preflight"] = training_preflight(torch, dtype)
        install_trackio_dashboard(api, repos["TRACKIO_SPACE_ID"])
        tracker = trackio.init(
            project="openenv-scottwu-online-lora",
            name=run_id,
            space_id=repos["TRACKIO_SPACE_ID"],
            private=True,
            config=report["hyperparameters"],
            capture_console=False,
        )
        persist()
        sandbox = Sandbox.connect(
            os.environ["SBX_JOB_ID"],
            namespace=os.environ.get(
                "SBX_NAMESPACE", repos["OUTPUT_REPO"].split("/")[0]
            ),
            token=os.environ["HF_TOKEN"],
        )
        remote = RemoteEnvironment(
            sandbox, report["transport"], int(os.environ.get("SBX_PORT", "8012"))
        )
        report["status"] = "freezing_tasks"
        train_rows = remote.tasks("train", list(range(64)))
        test_rows = remote.tasks("test", list(range(16)))
        train_hands = {hand_identity(r["prompt"]) for r in train_rows}
        test_hands = {hand_identity(r["prompt"]) for r in test_rows}
        if train_hands & test_hands:
            raise RuntimeError("Train/test arithmetic hands overlap")
        manifest = {
            "train": train_rows,
            "test": test_rows,
            "unique_train_hands": len(train_hands),
            "unique_test_hands": len(test_hands),
            "eval_seeds": EVAL_SEEDS,
            "pilot_train_indices": list(range(PILOT_GROUPS)),
            "minimum_raw_group_range": MIN_CONTRAST,
            "minimum_contrasting_groups": 2,
            "minimum_valid_fraction": MIN_VALID_FRACTION,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "max_completion_length": 128,
            "enable_thinking": False,
            "parser": "last answer outside think; missing answer submits empty expression",
        }
        report["manifest_sha256"] = digest(manifest)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        persist()
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=MODEL_REVISION, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=MODEL_REVISION,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to("cuda")

        def generate(row, seed):
            check_time(120)
            model.eval()
            set_seed(0 if seed is None else seed)
            text = tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
                model.device
            )
            settings = {"do_sample": seed is not None}
            if seed is not None:
                settings.update(temperature=0.8, top_p=1.0, top_k=0)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    **settings,
                    max_new_tokens=128,
                    max_time=60,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return tokenizer.decode(
                output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

        def score_texts(phase, rows, texts, seeds):
            check_time(60)
            if len({r["split"] for r in rows}) != 1:
                raise ValueError("Mixed score split")
            outcomes = remote.score(
                rows[0]["split"],
                [r["index"] for r in rows],
                [extract_expression(t) for t in texts],
            )
            for row, text, seed, outcome in zip(
                rows, texts, seeds, outcomes, strict=True
            ):
                append_trace(
                    {
                        "phase": phase,
                        "episode_key": row["episode_key"],
                        "seed": seed,
                        "completion": text,
                        "expression": extract_expression(text),
                        "parsed": bool(extract_expression(text)),
                        **outcome,
                    }
                )
            return outcomes

        def evaluate(phase):
            report["status"] = phase
            for row in test_rows:
                # Include index in sampling seed, then repeat these exact seeds after training.
                seeds = [
                    None if s is None else s + 100 * row["index"] for s in EVAL_SEEDS
                ]
                texts = [generate(row, seed) for seed in seeds]
                score_texts(phase, [row] * len(seeds), texts, seeds)
            persist()

        evaluate("baseline")  # The model has no adapter at this point.
        trackio.log(
            {
                "heldout/base_mean_raw_reward": report["evaluation"]["baseline"][
                    "mean_raw_reward"
                ]
            },
            step=0,
        )
        report["status"] = "pilot"
        pilot = []
        for row in train_rows[:PILOT_GROUPS]:
            seeds = [10000 + GROUP_SIZE * row["index"] + j for j in range(GROUP_SIZE)]
            texts = [generate(row, seed) for seed in seeds]
            pilot.append(score_texts("pilot", [row] * GROUP_SIZE, texts, seeds))
        report["signal_gate"] = signal_gate(pilot, expected_groups=PILOT_GROUPS)
        trackio.log(
            {
                "pilot/contrasting_groups": report["signal_gate"]["contrasting_groups"],
                "pilot/valid_fraction": report["signal_gate"]["valid_fraction"],
            },
            step=0,
        )
        if not report["signal_gate"]["passed"]:
            report["status"] = "no_grpo_learning_signal"
            persist()
            raise NoLearningSignal
        check_time(600)
        set_seed(42)
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            revision=MODEL_REVISION,
        )
        model = get_peft_model(model, lora)
        initial = {
            n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        report["trainable_parameters"] = sum(p.numel() for p in initial.values())
        report["initial_adapter_sha256"] = hashlib.sha256(
            b"".join(initial[n].numpy().tobytes() for n in sorted(initial))
        ).hexdigest()

        class Evidence(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                metrics = {
                    key: value
                    for key, value in (logs or {}).items()
                    if type(value) in (int, float) and math.isfinite(value)
                }
                if metrics:
                    trackio.log(metrics, step=state.global_step)

            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                self.before_steps = {
                    id(parameter): int(s["step"].item())
                    if hasattr(s["step"], "item")
                    else int(s["step"])
                    for parameter, s in kwargs["optimizer"].state.items()
                    if "step" in s
                }
                self.before = {
                    n: p.detach().float().cpu().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }
                gradients = [
                    p.grad.detach().float().square().sum()
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                self.grad_norm = (
                    float(torch.stack(gradients).sum().sqrt().cpu())
                    if gradients
                    else 0.0
                )

            def on_optimizer_step(self, args, state, control, **kwargs):
                optimizer = kwargs["optimizer"]
                steps = {
                    id(parameter): int(s["step"].item())
                    if hasattr(s["step"], "item")
                    else int(s["step"])
                    for parameter, s in optimizer.state.items()
                    if "step" in s
                }
                changed = 0
                for name, parameter in model.named_parameters():
                    if name in self.before:
                        value = parameter.detach().float().cpu()
                        if not torch.isfinite(value).all():
                            raise RuntimeError("Non-finite adapter weights")
                        changed += int(not torch.equal(self.before[name], value))
                report["optimizer_evidence"].append(
                    {
                        "attempt": state.global_step + 1,
                        "optimizer_step": max(steps.values(), default=0),
                        **optimizer_observation(
                            self.grad_norm,
                            bool(getattr(optimizer, "step_was_skipped", False)),
                            counters_advanced(self.before_steps, steps),
                            changed,
                        ),
                        "group_std": report["training_groups"][-1]["std"],
                    }
                )

            def on_step_end(self, args, state, control, **kwargs):
                report["global_step"] = state.global_step
                if state.global_step % 5 == 0:
                    save_adapter()
                    persist()
                if time.monotonic() + 600 >= deadline:
                    control.should_training_stop = True

        by_key = {r["episode_key"]: r for r in train_rows}

        def environment_reward(completions, episode_key, **kwargs):
            check_time(300)
            rows = [by_key[key] for key in episode_key]
            if len(rows) != len(completions) or len(rows) % GROUP_SIZE:
                raise RuntimeError("GRPO reward batch alignment failed")
            for start in range(0, len(rows), GROUP_SIZE):
                if len(set(episode_key[start : start + GROUP_SIZE])) != 1:
                    raise RuntimeError("GRPO siblings must share one task")
            texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
            outcomes = score_texts("train", rows, texts, [None] * len(rows))
            for start in range(0, len(rows), GROUP_SIZE):
                rewards = [o["reward"] for o in outcomes[start : start + GROUP_SIZE]]
                report["training_groups"].append(
                    {
                        "episode_key": episode_key[start],
                        "std": statistics.pstdev(rewards),
                        "variance": statistics.pvariance(rewards),
                    }
                )
            return [o["reward"] for o in outcomes]

        args = GRPOConfig(
            output_dir=str(out / "unused-checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            generation_batch_size=4,
            num_generations=4,
            num_iterations=1,
            max_steps=20,
            max_completion_length=128,
            learning_rate=5e-5,
            lr_scheduler_type="constant",
            warmup_steps=0,
            beta=0.0,
            loss_type="grpo",
            scale_rewards="group",
            temperature=0.8,
            top_p=1.0,
            top_k=0,
            optim="adamw_torch",
            weight_decay=0.0,
            bf16=bf16,
            fp16=not bf16,
            use_vllm=False,
            use_transformers_continuous_batching=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            chat_template_kwargs={"enable_thinking": False},
            seed=42,
            data_seed=42,
            logging_steps=1,
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            project="openenv-scottwu-online-lora",
            run_name=run_id,
            trackio_space_id=repos["TRACKIO_SPACE_ID"],
            trackio_static_space_id=False,
            hub_private_repo=True,
            push_to_hub=False,
            log_completions=False,
        )
        # Manual allowlisted uploads avoid training_args.bin and accidental public repos.
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=Dataset.from_list(train_rows),
            reward_funcs=environment_reward,
            callbacks=[Evidence()],
        )
        report["status"] = "training"
        persist()
        trainer.train()
        report["global_step"] = trainer.state.global_step
        evidence = report["optimizer_evidence"]
        effective = [e for e in evidence if effective_update(e)]
        report["effective_updates"] = len(effective)
        save_adapter()
        if not effective or not report["changed_adapter_tensors"]:
            raise RuntimeError("No verified gradient-driven adapter update")
        # Verify durable adapter bytes through an independent revision-pinned readback.
        downloaded = Path(
            snapshot_download(
                repo_id=repos["OUTPUT_REPO"],
                revision=report["adapter_commit"],
                local_dir=str(out / "readback"),
                allow_patterns=list(report["adapter_files_sha256"]),
            )
        )
        report["adapter_readback_verified"] = all(
            hashlib.sha256((downloaded / name).read_bytes()).hexdigest() == expected
            for name, expected in report["adapter_files_sha256"].items()
        )
        if not report["adapter_readback_verified"]:
            raise RuntimeError("Adapter readback hash mismatch")
        report["adapter_readback_commit"] = report["adapter_commit"]
        model.load_adapter(str(downloaded), adapter_name="default", is_trainable=False)
        model.set_adapter("default")
        reloaded = {
            n: p.detach().float().cpu()
            for n, p in model.named_parameters()
            if n in initial
        }
        reloaded_hash = hashlib.sha256(
            b"".join(reloaded[n].numpy().tobytes() for n in sorted(reloaded))
        ).hexdigest()
        report["adapter_reload_verified"] = (
            model.active_adapters == ["default"]
            and reloaded_hash == report["final_adapter_sha256"]
        )
        if not report["adapter_reload_verified"]:
            raise RuntimeError(
                "Persisted adapter reload did not match the trained weights"
            )
        report["adapter_evaluation_revision"] = report["adapter_readback_commit"]
        evaluate("adapter")
        report["status"] = "complete"
        trackio.log(
            {
                "heldout/base_mean_raw_reward": report["evaluation"]["baseline"][
                    "mean_raw_reward"
                ],
                "heldout/adapter_mean_raw_reward": report["evaluation"]["adapter"][
                    "mean_raw_reward"
                ],
                "heldout/paired_reward_difference": report["evaluation"]["paired"][
                    "mean_raw_reward_difference"
                ],
                "evidence/effective_updates": len(effective),
            }
        )
    except NoLearningSignal:
        # The signal gate recorded the early stop; finalizers still persist it.
        pass
    except Exception as error:
        failed = True
        report["failure_stage"] = report["status"]
        report["status"] = "failed"
        # Third-party exception messages may include authenticated URLs or headers.
        report["error_type"] = type(error).__name__
        report["error_frames"] = exception_frames(error)
    finally:
        # Independent finalizers preserve the original failure if one finalizer fails.
        for name, finalize in (
            (
                "adapter_save",
                lambda: save_adapter() if report["status"] != "complete" else None,
            ),
            (
                "trackio_finish",
                lambda: tracker.finish() if tracker is not None else None,
            ),
            ("sandbox_close", lambda: sandbox.close() if sandbox else None),
        ):
            try:
                finalize()
            except Exception as error:
                failed = True
                report.setdefault("finalization_errors", {})[name] = type(
                    error
                ).__name__
        if failed and report["status"] != "failed":
            report["result_status"] = report["status"]
            report["status"] = "finalization_failed"
        try:
            persist()
        except Exception as error:
            failed = True
            report.setdefault("finalization_errors", {})["artifact_upload"] = type(
                error
            ).__name__
            write_report()
            if report["status"] != "failed":
                report["result_status"] = report["status"]
                report["status"] = "finalization_failed"
            write_report()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(out / "report.json"),
                    "run_id": run_id,
                }
            ),
            flush=True,
        )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
