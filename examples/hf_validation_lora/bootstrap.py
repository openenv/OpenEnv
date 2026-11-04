"""Repeat the qualified source launch; optionally submit the bounded LoRA Job."""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import get_token, HfApi, Sandbox, Volume

IMAGE = "python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28"
TRAIN_IMAGE = "python:3.11.18-bookworm@sha256:77c90594edb5d6a43ed73de1c73b5f9d795bc94fa76d438fba4cc00036100014"
SOURCE = "FineEnvs/ScottWuMath"
REVISION = "4de8125d314fca0a3f137fa1bab04b966a3efc15"
HERE = Path(__file__).resolve().parent
TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def require_qualified(report):
    required = {
        "runtime." + name
        for name in [
            "startup",
            "reward_well_formed",
            "observation_schema",
            "state_contract",
            "episode_isolation",
        ]
    }
    measured = [row for row in report["results"] if row["check_id"] in required]
    if (
        len(measured) != len(required)
        or {row["check_id"] for row in measured} != required
        or any(row["status"] != "pass" for row in measured)
        or any(row["status"] in {"fail", "error"} for row in report["results"])
    ):
        raise RuntimeError("source-bootstrap qualification failed; training refused")


def stop_owned_job(api, job_id):
    try:
        stage = api.inspect_job(job_id=job_id).status.stage
    except Exception:
        stage = None  # An unavailable inspection must not prevent cancellation.
    if stage not in TERMINAL:
        api.cancel_job(job_id=job_id)
        for attempt in range(11):
            stage = api.inspect_job(job_id=job_id).status.stage
            if stage in TERMINAL:
                break
            if attempt < 10:
                time.sleep(1)
    if stage not in TERMINAL:
        raise RuntimeError("owned Job cancellation not confirmed")
    return stage


def finish_owned_compute(api, training, sandbox, record, output, artifact_repo):
    errors = {}
    for name, job in [("training", training), ("sandbox", sandbox)]:
        if job is None:
            continue
        try:
            record[name + "_final_stage"] = stop_owned_job(api, job.id)
        except Exception as error:
            errors[name] = type(error).__name__
    if errors:
        record["cleanup_errors"] = errors
    state = output / "state.json"
    state.write_text(json.dumps(record, indent=2))
    try:
        api.upload_file(
            path_or_fileobj=state,
            path_in_repo="reproduction/state.json",
            repo_id=artifact_repo,
            repo_type="dataset",
        )
    finally:
        if errors:
            raise RuntimeError("owned compute cleanup incomplete; see state.json")


def training_command(repo, revision):
    # Only this trusted training worker receives Hub credentials. The environment
    # Sandbox receives public source and explicit files, without the HF token.
    download = f"""import os, pathlib, urllib.request
for src,dst in [('scripts/train.py','/tmp/train.py'),('locks/training-python311.txt','/tmp/training.txt')]:
    req=urllib.request.Request('https://huggingface.co/datasets/{repo}/resolve/{revision}/'+src, headers={{'Authorization':'Bearer '+os.environ['HF_TOKEN']}})
    pathlib.Path(dst).write_bytes(urllib.request.urlopen(req,timeout=60).read())
"""
    return [
        "/bin/sh",
        "-lc",
        "python -m pip install --disable-pip-version-check uv==0.9.9 && "
        "python - <<'PY'\n" + download + "PY\n"
        "uv run --python /usr/local/bin/python --with-requirements /tmp/training.txt /tmp/train.py",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo", required=True)
    parser.add_argument("--model-repo")
    parser.add_argument("--trackio-space")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--flavor", choices=["l4x1", "t4-small"], default="l4x1")
    parser.add_argument("--output", type=Path, default=Path("hf-experiment"))
    args = parser.parse_args()
    if args.train and not (args.model_repo and args.trackio_space):
        parser.error(
            "--train requires --model-repo and a fresh private --trackio-space"
        )
    api = HfApi()
    if not api.dataset_info(args.artifact_repo).private:
        parser.error("artifact repository must already exist and be private")
    if args.train and (
        not api.model_info(args.model_repo).private
        or not api.space_info(args.trackio_space).private
    ):
        parser.error("model repository and fresh Trackio Space must be private")
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "source": SOURCE,
        "revision": REVISION,
        "image": IMAGE,
        "training_flavor": args.flavor,
    }
    sandbox = None
    training = None
    try:
        sandbox = Sandbox.create(
            image=IMAGE,
            flavor="cpu-basic",
            idle_timeout="20m",
            forward_hf_token=False,
            volumes=[
                Volume(
                    type="space",
                    source=SOURCE,
                    revision=REVISION,
                    mount_path="/snapshot",
                    read_only=True,
                )
            ],
            labels={"name": "oe-scottwu-reproduce", "experiment": "oe-fineenvs-e2e"},
        )
        record["sandbox_job_id"] = sandbox.id
        (args.output / "state.json").write_text(json.dumps(record, indent=2))
        print("Owned Sandbox:", sandbox.id, flush=True)
        sandbox.run("mkdir -p /opt/scott && cp -a /snapshot/. /opt/scott/", timeout=60)
        sandbox.files.write(
            "/tmp/math-requirements.txt", (HERE / "math-python312.txt").read_text()
        )
        for name in ["qualify.py", "score_batch.py"]:
            sandbox.files.write("/opt/" + name, (HERE / name).read_text())
        sandbox.run(
            "python -m pip install --disable-pip-version-check uv==0.9.9 && "
            "uv venv /opt/mathenv && uv pip install --python /opt/mathenv/bin/python "
            "-r /tmp/math-requirements.txt > /tmp/math-install.log 2>&1",
            timeout=300,
        )
        sandbox.run(
            "ENABLE_WEB_INTERFACE=false MAX_CONCURRENT_ENVS=4 /opt/mathenv/bin/python "
            "-m uvicorn scott_wu_math.server.app:app --host 127.0.0.1 --port 8012 "
            "> /tmp/math-server.log 2>&1",
            background=True,
        )
        sandbox.run(
            "/opt/mathenv/bin/python - <<'PY'\n"
            "import time,urllib.request\n"
            "for _ in range(60):\n"
            " try:\n"
            "  assert urllib.request.urlopen('http://127.0.0.1:8012/health',timeout=1).status==200;break\n"
            " except Exception: time.sleep(1)\n"
            "else: raise RuntimeError('server readiness deadline exceeded')\nPY",
            timeout=65,
        )
        sandbox.run(
            "/opt/mathenv/bin/python /opt/qualify.py --env math > /tmp/qualify.log",
            timeout=120,
        )
        report = json.loads(
            sandbox.files.read_text("/tmp/qualification-math/report.json")
        )
        require_qualified(report)
        sandbox.run(
            "tar -czf /tmp/qualification.tgz -C /tmp qualification-math", timeout=30
        )
        sandbox.files.download(
            "/tmp/qualification.tgz", args.output / "qualification.tgz"
        )
        api.upload_folder(
            repo_id=args.artifact_repo,
            repo_type="dataset",
            folder_path=args.output,
            path_in_repo="reproduction",
            commit_message="Retain source qualification",
        )
        if args.train:
            from huggingface_hub import CommitOperationAdd

            commit = api.create_commit(
                repo_id=args.artifact_repo,
                repo_type="dataset",
                commit_message="Pin training inputs",
                operations=[
                    CommitOperationAdd(path_in_repo=dst, path_or_fileobj=HERE / src)
                    for src, dst in [
                        ("train.py", "scripts/train.py"),
                        ("training-python311.txt", "locks/training-python311.txt"),
                    ]
                ],
            )
            training = api.run_job(
                image=TRAIN_IMAGE,
                command=training_command(args.artifact_repo, commit.oid),
                flavor=args.flavor,
                timeout="1h",
                secrets={"HF_TOKEN": get_token()},
                env={
                    "SBX_JOB_ID": sandbox.id,
                    "ENV_TRANSPORT": "exec",
                    "OUTPUT_REPO": args.model_repo,
                    "ARTIFACT_REPO": args.artifact_repo,
                    "TRACKIO_SPACE_ID": args.trackio_space,
                },
                labels={"name": "oe-scottwu-lora", "experiment": "oe-fineenvs-e2e"},
            )
            record["training_job_id"] = training.id
            print("Training Job:", training.id, flush=True)
            deadline = time.monotonic() + 3900
            while time.monotonic() < deadline:
                job = api.inspect_job(job_id=training.id)
                if job.status.stage in TERMINAL:
                    record["training_stage"] = job.status.stage
                    if job.status.stage != "COMPLETED":
                        raise RuntimeError("training Job ended at " + job.status.stage)
                    break
                time.sleep(30)
            else:
                raise TimeoutError("training Job exceeded controller deadline")
    finally:
        finish_owned_compute(
            api, training, sandbox, record, args.output, args.artifact_repo
        )


if __name__ == "__main__":
    main()
