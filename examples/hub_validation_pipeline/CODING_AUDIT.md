# Reproducible supplemental coding audits

The operator implementation lives in `examples/coding_audit`: `scoring.py` defines
reports/checks, `hf.py` adapts fresh HF Sandboxes, `runner.py` orchestrates attempts,
and `__main__.py` exposes the controller command. Cloud and discovery orchestration
in this directory import that example package. This workflow does not add a coding
mode to the installed `openenv validate` CLI.

`cloud.py` freezes a Harbor task, operator audit configuration, required source and
`uv.lock` into a content-addressed bundle. Submission publishes it to an existing
private results dataset, records the immutable commit, and launches an HF Job
controller. Each solution execution and verifier run uses a fresh dedicated HF
Sandbox. Docker inside the controller Job is unnecessary.

The supplemental profile is `harbor-coding-audit-v1`. Its PASS covers the measured
reference, rejection, leakage/reliability and rollout checks. Reports retain
`rfc008_complete: false` and `trainability_established: false`. OpenEnv's native
`harbor-semantic-v1` contract and the complete `full` admission requirements keep
their own meanings.

## What runs

1. Freeze/hash the task, verifier and operator configuration. Reject source drift,
   mutable image references, unsafe artifact paths and excessive workload bounds.
2. Run NOP in a clean solver Sandbox and transfer only allowlisted submission file
   bytes to a separate clean verifier Sandbox.
3. Generate bounded candidate shell solutions from solver-visible instructions.
   Execute provided/generated candidates and repeat exact submissions in fresh
   sandboxes. Clean incorrect generated candidates are excluded; execution and
   cleanup errors remain errors. Never rewrite a verifier to make a candidate pass.
4. Execute the operator's reviewed known-wrong cases and require the frozen
   verifier to reject them. The checked-in toy cases apply only to the synthetic
   demo task. A generator's claim that its own solution is wrong does not establish
   trusted negative coverage for another task.
5. Probe private-path visibility, stale solver state and a verifier-private canary
   in collected transcripts. Check source/image binding and terminal Job status
   after cleanup.
6. Only after screening passes, measure repeated target attempts. Retain successes,
   failures, infrastructure errors, rewards, variance, mixed-success groups,
   uncertainty, token usage and latency. Uniform success/failure is a measured
   difficulty result, not proof of task invalidity or universal usefulness.

A candidate passing a weak verifier does not establish intended correctness.
Demonstrating training benefit requires an independent controlled learning
experiment with held-out task families; see [the research plan](../../docs/plans/hub-environment-trainability-validation.md).

## Model and protocol

The synthetic configuration pins public source provenance to
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`, with `deepinfra` explicitly selected.
Its attempt/candidate/replay and token/step limits are declared in
[config.json](../coding_audit/config.json). Real-task protocols come entirely from the
operator panel and must be reviewed independently.

Model/provider availability is checked before task Sandbox creation. A source
commit does not force a provider to serve that revision. Evidence retains requested
and returned model identities and supplied fingerprints. Provider-specific seed
behavior is not assumed; recorded attempt seeds may be identifiers only. Frozen
inputs and the same declared protocol do not imply identical hosted generations.

## Prepare a task and configuration

Use a reviewed, self-contained Harbor task with `task.toml`, `instruction.md` and
`tests/test.sh`. A supplied `solution/solve.sh` is optional for the supplemental
audit because it can generate candidate references. This differs from native
semantic oracle admission. Multi-service, multi-step, credential-dependent or
unsupported resource/network contracts require a qualified adapter.

Solver/verifier images must already exist under immutable OCI registry digests;
HF Sandbox has no Dockerfile build-context API. Build required images on a trusted
builder. Keep private tests, reference solutions and answer keys out of solver
images and public layers. The generic pinned Python image in the toy configuration
is not a qualified image for arbitrary downloaded tasks.

Declare exact `submission_paths`, `private_paths`, verifier command, reward file/key,
reward range, trusted negatives and resource/time budgets. Submission paths name
individual files; directory/glob exports are unsupported. Only those bytes cross
into the fresh verifier. Set `cloud.results_repo` to an existing private dataset
under your control.

Create the trusted synthetic controls without executing them:

```bash
uv run --no-sync python -c \
  'from pathlib import Path; from examples.hub_validation_pipeline.demo import create_tasks; create_tasks(Path("outputs/coding-controls"))'
```

Prepare a local bundle and reviewable launch plan; this performs no paid operations
or remote writes:

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.cloud prepare \
  --task outputs/coding-controls/good \
  --config examples/coding_audit/config.json \
  --output outputs/audit-plan
```

The plan records hashes, source provenance, dependency lock and conservative
Sandbox/inference workload bounds. Review `plan.json`, the configured protocol and
applicable prices before submission. Child command/start/idle deadlines do not
replace the SDK's 24-hour remote Job maximum lifetime. An interrupted controller
may leave an active child billable until shutdown; inspect/cancel exact run-owned
Jobs and retain cleanup evidence.

To prepare discovered tasks, supply a configuration appropriate for the selected
family:

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.pipeline \
  --source both --dataset-limit 1 --tasks-per-dataset 1 \
  --coding-config inputs/reviewed-audit.json --output outputs/hub-coding-plans
```

This pipeline option selects supplemental cloud preparation; it is not an option
on `openenv validate`. The driver writes bundles and `planned.jsonl`; preparation
or a queued Job never marks a task accepted. Access/preparation failures remain
explicit exclusions. Explicit pinned dataset inputs or `--local-dataset` reproduce
the selection without relying on a later discovery ranking.

Acquisition uses ordinary HF SDK revision/local-directory semantics, corresponding
to `hf download --repo-type dataset --revision ... --local-dir ...`, with anonymous
public reads and additional selection/size/hash checks. Publication uses explicit
SDK additions and commit identities. Like `hf upload`, it does not delete remote
files unless deletion is explicitly requested.

## Operator-supplied native evaluation panels

`real_tasks.py` and `full_run.py` require `--panel PATH`. Their Python entry points
likewise require the keyword `panel_path`. There is no embedded real-task list,
reference implementation, known-wrong submission or fallback private input path.
Keep panels, private task material and populated reports/ledgers in protected
operational storage; do not include their payloads in public source or PR prose.

A panel has schema `openenv-native-evaluation-panel-v1`, `evaluation_only: true`,
a pinned HF `source` (`kind`, `repo_id`, full `revision`) and a nonempty `tasks` list.
Each task declares:

- A unique safe `name` and package-relative `task_path`.
- An immutable `image` and exact `submission_path`, matching its configuration.
- Complete audit `config`, including reviewed negative cases and required IDs.
- Operator-reviewed `negative_counterexamples` metadata.
- Optional `annotations` for task-specific caveats or provenance.

The panel's exact bytes are read once. Manifests record its path, byte length and
SHA256; each prepared bundle's source provenance includes the same panel SHA256.
Native task/verifier bytes remain unchanged and bind to the original acquired
file inventory. Remote source identities require a complete, verified acquisition
hash inventory; a revision string alone cannot authorize a bundle. Local revisions
are checked against the captured bundle bytes, including the verifier. Configuration and annotations come from the operator input.

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.real_tasks \
  --panel "$PANEL_PATH" --output outputs/native-recipes --bundle

uv run --no-sync python -m examples.hub_validation_pipeline.full_run \
  --panel "$PANEL_PATH" --hf-census "$HF_CENSUS_PATH" \
  --harbor-census "$HARBOR_CENSUS_PATH" --budget-usd "$BUDGET_USD" \
  --output outputs/native-cohort-plan
```

Set the variables to reviewed operator inputs. `full_run` freezes small census
summaries and hashes referenced inventories; census membership does not qualify
a task. Repeat `--task NAME` to select distinct names from the panel in that order;
omitting it selects all panel tasks. Empty, unknown and duplicate selections fail
before acquisition/spending. Native panels remain evaluation-only.

## Shared spending and one-shot submission

Initialize one shared ledger using your global allowance, without publishing its
contents. Set `SPEND_LEDGER` and `GLOBAL_BUDGET_USD` before this command:

```bash
uv run --no-sync python -c \
  'import os; from examples.hub_validation_pipeline import spend; spend.init(os.environ["SPEND_LEDGER"], os.environ["GLOBAL_BUDGET_USD"])'
```

Account for prior/concurrent spending and unresolved reservations in that ledger.
Use the same ledger across all cooperating launchers. For a native cohort,
`full_run(..., submit=True)` requires nonempty `spend_ledger` and `reservation_id`
before preparation; the CLI requires both flags:

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.full_run \
  --panel "$PANEL_PATH" --hf-census "$HF_CENSUS_PATH" \
  --harbor-census "$HARBOR_CENSUS_PATH" --budget-usd "$BUDGET_USD" \
  --spend-ledger "$SPEND_LEDGER" --reservation-id "$RESERVATION_ID" \
  --output outputs/native-cohort-submit --submit
```

The aggregate reservation covers configured request/token ceilings, controller
allowances and each child's full remote lifetime plus startup allowance. The
ratebook is a declared model, not a current-price guarantee or account billing cap;
unsupported model/provider/flavor changes require a reviewed ratebook. Both the
per-run cap and remaining shared allowance must cover the selected cohort.

Every budget-bearing initial plan parse and final submission checks the expected
SHA256 against the same bytes it parses. A new directory or reservation identifier
does not bypass immutable run-ID duplicate protection. Partial/uncertain receipts
and unresolved reservations remain durable for manual reconciliation; never
release them merely because a controller stopped. Jobs are never automatically
resumed or blindly retried.

`prepared_batch` provides shared reservation and bounded controller concurrency
for an already reviewed preparation manifest. Use these guarded cohort launchers
for paid work; the cloud CLI and discovery pipeline only prepare inputs.
Both preparation producers emit `plan_path` relative to that manifest's directory
and `plan_sha256` for the exact generated plan bytes. Historical manifests without
these fields require fresh preparation and review; the launcher does not infer a
new digest for an old plan.

Plans bind `runtime_versions` for uv and `huggingface_hub`. The remote bootstrap
runs the frozen dependency sync, installs the declared `huggingface_hub==1.33.0`
overlay with that environment's interpreter, then verifies the installed version
before starting the audit worker. A missing or incompatible runtime pin requires
fresh preparation; changing the package's general dependency range is unnecessary.

`cloud.submit()` requires a ledger, reservation ID and expected plan SHA256.
Each reservation binds the reviewed plan hashes to their modeled cost allocations;
allocations must fit the reserved total. Submission rechecks the active hold,
budget block and parsed plan bytes while holding the shared ledger lock through
launch. Unbound legacy reservations cannot authorize a launch.
The submitter verifies private dataset access, stages immutable inputs, claims the
run identity and persists the returned Job identifier and receipt. `prepare` never
makes model calls or launches paid Jobs.

## Credentials and execution boundary

Supply three distinct scoped credentials at runtime: `HF_TOKEN` for the private
input/result dataset, `HF_INFERENCE_TOKEN` for inference and `HF_SANDBOX_TOKEN` for
execution-only Jobs/Sandbox access. Keep their values out of configs, arguments,
bundles and Git. Submission uses HF Job secrets; they are not deliberately passed
into task environment variables or tool arguments.

The current Sandbox RPC sends its execution bearer to the workload server.
Hostile root code could replace that server, so this backend is for reviewed
workloads; separate credentials limit authority without proving hostile-code
containment. A separate verifier Sandbox also does not establish that arbitrary
submitted code cannot read its grader or forge same-user output. Strong verifier
isolation requests are refused; these limits remain in reports.

The worker retains reports, config, raw rewards, generated candidates, submission
hashes, sanitized transcripts and checksums in the private result dataset even
when an audit fails normally. Infrastructure termination can prevent final upload;
absent results remain incomplete. Keep original source and prior evidence intact.

Run the same controller locally, with task execution still in HF Sandboxes:

```bash
uv run --no-sync python -m examples.coding_audit \
  --task inputs/task --config inputs/audit.json --output outputs/audit-report.json
```

Missing evidence, malformed/ambiguous rewards, source/image drift, incorrect
provided references, accepted wrong answers, failed probes and unverified cleanup
block acceptance. Provider/Job failures never become synthetic zero-reward model
attempts. Checks retain PASS/FAIL/SKIP/ERROR and qualification limits.

## Verification

```bash
PYTHONPATH=.:src:envs uv run --no-sync python -m pytest \
  examples/hub_validation_pipeline/ -q
```

Injected SDK checks consume no cloud/inference credits and establish source-level
behavior. They do not establish Linux Docker, live-provider, billing or learning
qualification. A live audit needs separately authorized compute and scoped access.
