# Coding split packaging tools

These tools check and package **externally supplied** Harbor-style coding tasks.
No benchmark tasks, private tests, solutions, negative implementations, task hash
inventory, or execution results are distributed here. The offline tool tests
create a disposable toy package containing a packaging marker; it is not a
benchmark task or a difficulty example.

The checker freezes each task before running controls, uses a fresh solver and
verifier directory per case, and transfers only declared Python files to the
verifier. Admission requires the baseline to have both passing and failing tests,
three successful reference runs, three successful alternative runs, and failure
of every declared negative. Missing, repeated, reordered, skipped, or inconsistent
control evidence fails admission. Every control must execute the same test
identities, with no skipped or duplicated tests. These are local processes executing trusted
input scripts, not a sandbox for hostile code.

## External input contract

Supply `--task-root` with a directory containing only the task directories selected
by your inventory. Each task uses this layout:

```text
TASK_ID/
  instruction.md
  task.toml
  spec.json
  environment/Dockerfile
  environment/source/...
  tests/test.sh
  solution/solve.sh
  solution/alternate.sh
  negatives/CASE_ID.sh
```

`spec.json` supplies `task_id` (equal to the directory name), `submission_paths`
(normalized `.py` paths below `/app/`), `negative_cases` (script basenames),
`capability_family`, `skills`, and `intended_difficulty`. Negative basenames must be
unique and must not collide with baseline, reference, or alternative control IDs. Declared submissions must
exist in the starter source. The local verifier receives `APP_DIR`, `TEST_DIR`,
`PYTHON`, and `VERIFIER_LOG_DIR`; it must exit zero, write `0` or `1` to
`$VERIFIER_LOG_DIR/reward.txt`, and emit verbose unittest output with test counts.

Supply `--inventory` with a separately reviewed JSON file:

```json
{
  "schema_version": 1,
  "revision": "your-reviewed-source-revision",
  "task_file_hashes": {
    "YOUR_TASK_ID": {
      "instruction.md": "<64 lowercase hexadecimal SHA-256 characters>",
      "... every other task file ...": "<its SHA-256>"
    }
  }
}
```

Every task file must match, with no missing or additional paths. Symlinks,
nonregular files, and generated Python bytecode are rejected. Keep the inventory
outside the task root. This tool consumes an approved inventory; it does not
regenerate one to accept changed inputs.

## Offline use

From this directory, set `TASK_ROOT` and `INVENTORY` to your private inputs:

```bash
: "${TASK_ROOT:?Set TASK_ROOT to your task directory}"
: "${INVENTORY:?Set INVENTORY to your reviewed inventory JSON}"
python3 -B split.py check --task-root "$TASK_ROOT" --inventory "$INVENTORY" \
  --python python3 --output outputs/local-controls.json
python3 -B split.py build --task-root "$TASK_ROOT" --inventory "$INVENTORY" \
  --evidence outputs/local-controls.json --output outputs/package
```

Build requires a new output directory outside the task root. It rechecks evidence
against the current source and exact inventory, copies the tasks unchanged, and
emits templates, a manifest, checksums, and `splits/synthetic_gap_v1.jsonl`.
**Outputs contain the full private payload and verifier logs.** Keep them private;
this directory's `.gitignore` allows only the six public files. An interrupted
build or preparation may leave partial output; choose a new output directory
when retrying.

The optional Docker checker takes the same external inputs:

```bash
python3 -B container_check.py --task-root "$TASK_ROOT" --inventory "$INVENTORY" \
  --output outputs/container-controls.json
python3 -B split.py build --task-root "$TASK_ROOT" --inventory "$INVENTORY" \
  --evidence outputs/local-controls.json \
  --container-evidence outputs/container-controls.json --output outputs/container-package
```

This requires Docker and the pinned Python 3.12 base image to be available. It
builds from each task's `environment/`, runs with networking disabled, and grades
transferred files in a fresh base image. Only public starter files should enter
the solver image. The checker checks `/tests` and `/solution` are absent, records
local image IDs, and removes its temporary tags. Every create/run has a unique
owned container name and removes that container and its anonymous volumes in a
`finally` block, including CLI timeout paths. Image IDs do not attest published
registry contents or establish cloud qualification.

## Optional experimental preparation

`prepare` is an opt-in integration with an **experimental, unmerged** cloud
preparer from another checkout. It is not supplied or required by this example.
Select that checkout explicitly with `--pipeline-root`. The adapter must accept
`source.task_file_hashes`, validate captured acquisition files, and expose
`_read_plan(..., expected_sha256=...)` to bind the exact parsed plan bytes; older
or incompatible adapters fail closed. Select only trusted
pipeline code, because preparation imports and executes it.

Set `PIPELINE_ROOT`, `IMAGE_LOCK`, and `RESULTS_REPO` for that integration:

```bash
: "${PIPELINE_ROOT:?Set PIPELINE_ROOT to a compatible experimental pipeline checkout}"
: "${IMAGE_LOCK:?Set IMAGE_LOCK to your private per-task registry image lock JSON}"
: "${RESULTS_REPO:?Set RESULTS_REPO to your private results repository}"
python3 -B split.py prepare --task-root "$TASK_ROOT" --inventory "$INVENTORY" \
  --pipeline-root "$PIPELINE_ROOT" --image-lock "$IMAGE_LOCK" \
  --results-repo "$RESULTS_REPO" --output outputs/plans
```

The image lock maps exactly the inventory's task IDs to real solver image names
pinned as `REGISTRY/IMAGE@sha256:DIGEST`. The generic verifier base is rejected as
a solver image. Preparation forwards the exact external file hashes and verifies
both the returned plan binding and captured bundle inventory before writing its
summary. Each recorded plan SHA is the hash passed to the reader; submission
callers can forward it as `expected_plan_sha256` to `stage_inputs` or `submit`.
Keep trusted image build receipts separately: a registry pin alone does
not prove which source was built. This command does not submit work; the external
adapter and any later provider execution are outside the offline checks here.

Templates retain the `qwen38-gap48-v1` target settings: 48 steps, 4,096 response
tokens, and 32,768 total tokens. The generator uses 1 step, 8,192 response tokens,
and 32,768 total tokens, with the target's other model settings unchanged. Tool
output is capped at 1,048,576 bytes. Templates omit the solver image and results
repository until explicitly supplied for preparation. Controls do not establish
model difficulty, trainability, provider reliability, or full RFC 008 qualification.

## Test the tooling

From the repository root, with the development tools already installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s examples/synthetic_coding_split -p test_split.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest examples/synthetic_coding_split/test_split.py -q
usort check examples/synthetic_coding_split/*.py
ruff format --check examples/synthetic_coding_split/*.py
ruff check examples/synthetic_coding_split/*.py
```

These tests use only generated toy files and mocked cloud/Docker boundaries. They
require no private tasks, credentials, Hub access, Docker daemon, or paid calls.
