"""Prepare operator-supplied evaluation-only native tasks without paid work.

Run ``python -m examples.hub_validation_pipeline.real_tasks --panel PANEL --output DIR``.
The panel supplies pinned source identity, task boundaries and audit controls.
``--bundle`` also calls the local, network-free ``cloud.prepare`` preparer.
Only the selected immutable HF task trees are downloaded anonymously.
"""

import argparse
import hashlib
import json
import re
import stat
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download, HfApi, RepoFile, RepoFolder
from openenv.validation.runner import source_digest

from . import cloud, sources
from .sources import MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES

_REQUIRED = {"task.toml", "instruction.md", "solution/solve.sh", "tests/test.sh"}


def _load_panel(panel_path):
    """Validate one exact byte snapshot and return its provenance."""
    path = Path(panel_path).absolute()
    data, _ = cloud._text_file(path)
    panel = cloud._loads(data)
    if (
        not isinstance(panel, dict)
        or panel.get("schema") != "openenv-native-evaluation-panel-v1"
        or panel.get("evaluation_only") is not True
    ):
        raise ValueError("Supply an evaluation-only native panel v1")
    source = panel.get("source", {})
    if (
        not isinstance(source, dict)
        or source.get("kind") != "hf"
        or not isinstance(source.get("repo_id"), str)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", source["repo_id"])
    ):
        raise ValueError("Panel source must identify an HF dataset")
    sources._pin(source.get("revision"), "hf")
    tasks = panel.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Panel requires a nonempty task list")
    names, paths = set(), set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Panel tasks must be objects")
        name = cloud._safe_name(task.get("name", ""))
        task_path = (
            PurePosixPath(".")
            if task.get("task_path") == "."
            else cloud._safe_name(task.get("task_path", ""))
        )
        if (
            len(name.parts) != 1
            or name.as_posix().casefold() in names
            or task_path.as_posix().casefold() in paths
            or name.as_posix().lower().startswith(".env")
            or any(part.lower().startswith(".env") for part in task_path.parts)
        ):
            raise ValueError("Panel requires distinct safe task names and paths")
        names.add(name.as_posix().casefold())
        paths.add(task_path.as_posix().casefold())
        config, _ = cloud._config(task.get("config", {}))
        execution = config["execution"]
        if (
            execution["agent_image"] != task.get("image")
            or execution["verifier_image"] != task.get("image")
            or execution["submission_paths"] != [task.get("submission_path")]
        ):
            raise ValueError("Panel task and execution boundaries disagree")
        if not isinstance(task.get("negative_counterexamples"), list) or not all(
            isinstance(item, dict) for item in task["negative_counterexamples"]
        ):
            raise ValueError("Panel counterexamples must be a list of objects")
        if not isinstance(task.get("annotations", {}), dict):
            raise ValueError("Panel annotations must be an object")
        task["config"] = config
    return panel, {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _select_tasks(panel, task_names):
    tasks = {task["name"]: task for task in panel["tasks"]}
    names = list(tasks if task_names is None else task_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in tasks for name in names)
    ):
        raise ValueError("Select distinct configured panel task names")
    return [tasks[name] for name in names]


def acquire_task(task, destination, *, source, api=None, download=None):
    """Acquire only one pinned native tree, with size and Git-blob checks."""
    revision = sources._pin(source["revision"], "hf")
    repo_id, task_path = source["repo_id"], task["task_path"]
    if task_path != ".":
        cloud._safe_name(task_path)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Acquisition requires a fresh task directory")
    destination = destination.absolute()
    api = api if api is not None else HfApi(token=False)
    download = download if download is not None else hf_hub_download
    info = api.dataset_info(repo_id, revision=revision)
    if info.sha != revision or info.gated or info.disabled:
        raise ValueError("Public source immutable revision could not be verified")
    files = {}
    total = 0
    for item in api.list_repo_tree(
        repo_id,
        path_in_repo=None if task_path == "." else task_path,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        token=False,
    ):
        path = cloud._safe_name(item.path)
        if not path.is_relative_to(task_path):
            raise ValueError("Source listing escaped the selected task tree")
        if any(part.lower().startswith(".env") for part in path.parts):
            raise ValueError("Environment files are not recipe inputs")
        if isinstance(item, RepoFolder):
            continue
        if not isinstance(item, RepoFile) or item.lfs:
            raise ValueError("Only native regular Git source files are supported")
        relative = path.relative_to(task_path).as_posix()
        if relative.casefold() in {key.casefold() for key in files}:
            raise ValueError("Source path collision")
        if type(item.size) is not int or not 0 <= item.size <= MAX_FILE_BYTES:
            raise ValueError("Source per-file byte cap exceeded")
        total += item.size
        files[relative] = item
        if len(files) > MAX_FILES or total > MAX_TOTAL_BYTES:
            raise ValueError("Source file count or total byte cap exceeded")
    if not _REQUIRED.issubset(files):
        raise ValueError("Native task files are missing")
    if sources.task_roots(files) != ["."]:
        raise ValueError("Selected native task root is ambiguous")
    inventory, payloads = {}, {}
    with tempfile.TemporaryDirectory(prefix="openenv-real-task-") as temporary:
        staging = Path(temporary).resolve()
        for relative, item in sorted(files.items()):
            path = Path(
                download(
                    repo_id,
                    item.path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=staging,
                    token=False,
                    force_download=True,
                )
            )
            if path.is_symlink() or path.resolve() != staging / item.path:
                raise ValueError("Download must materialize a contained regular file")
            if (
                not stat.S_ISREG(path.lstat().st_mode)
                or path.stat().st_size != item.size
            ):
                raise ValueError("Source file type or size disagrees with pinned tree")
            data, _ = cloud._text_file(path)
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != item.blob_id:
                raise ValueError("Source Git blob hash mismatch")
            inventory[relative] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob": blob,
            }
            payloads[relative] = data
        destination.mkdir(parents=True)
        for relative, data in payloads.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return inventory


def prepare(
    output_dir,
    *,
    panel_path,
    task_names=None,
    bundle=False,
    api=None,
    download=None,
):
    """Load an explicit operator panel and prepare recipes, never Jobs."""
    panel, provenance = _load_panel(panel_path)
    return _prepare(
        output_dir,
        panel=panel,
        panel_provenance=provenance,
        task_names=task_names,
        bundle=bundle,
        api=api,
        download=download,
    )


def _prepare(
    output_dir,
    *,
    panel,
    panel_provenance,
    task_names=None,
    bundle=False,
    api=None,
    download=None,
):
    tasks = _select_tasks(panel, task_names)
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in tasks:
        name, config = task["name"], task["config"]
        task_root = output / "sources" / name
        inventory = acquire_task(
            task, task_root, source=panel["source"], api=api, download=download
        )
        config_path = output / "configs" / f"{name}.json"
        config_path.parent.mkdir(exist_ok=True)
        config_bytes = cloud._json_bytes(config)
        config_path.write_bytes(config_bytes)
        native = tomllib.loads((task_root / "task.toml").read_text())
        source = {
            **panel["source"],
            "task_path": task["task_path"],
            "evaluation_panel_sha256": panel_provenance["sha256"],
            "task_file_hashes": {
                path: record["sha256"] for path, record in inventory.items()
            },
        }
        row = {
            "name": name,
            "task_id": f"hf:{source['repo_id']}@{source['revision']}:{source['task_path']}",
            "task_root": str(task_root),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source": source,
            "source_digest": source_digest(task_root),
            "files": inventory,
            "evaluation_only": True,
            "negative_counterexamples": task["negative_counterexamples"],
            "annotations": task.get("annotations", {}),
            "source_image": native.get("environment", {}).get("docker_image"),
            "execution_image": task["image"],
            "budget": cloud._config(config)[1],
        }
        if bundle:
            directory = output / "bundles" / name
            cloud.prepare(task_root, config, directory, source=source)
            plan_path = directory / "plan.json"
            row.update(
                bundle_plan=str(plan_path),
                plan_path=plan_path.relative_to(output).as_posix(),
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            )
        rows.append(row)
    manifest = {
        "schema": "openenv-real-task-recipes-v1",
        "status": "prepared",
        "evaluation_only": True,
        "evaluation_panel": panel_provenance,
        "scope": "Operator-selected tasks only; no whole-catalog qualification or admission claim.",
        "limitations": [
            "No Jobs, inference, containers, builds or pushes are performed by this recipe.",
            "Hosted model source revision does not attest provider-served weights.",
            "Native task and verifier bytes are preserved; dependency and resource reproducibility require separate qualification.",
        ],
        "tasks": rows,
    }
    (output / "manifest.json").write_bytes(cloud._json_bytes(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", dest="panel_path", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="task_names")
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Also prepare local bundles; does not submit",
    )
    args = parser.parse_args(argv)
    prepare(
        args.output,
        panel_path=args.panel_path,
        task_names=args.task_names,
        bundle=args.bundle,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output.absolute() / "manifest.json"),
                "status": "prepared",
            }
        )
    )


if __name__ == "__main__":
    main()
