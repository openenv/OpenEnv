# Discover, validate and evaluate Harbor tasks

This example discovers Hugging Face datasets tagged with both `library:harbor`
and `rl-environment`, discovers Harbor Hub packages, acquires pinned task files,
and calls OpenEnv's native validator for each task. It builds an admitted corpus
before optionally running a configured model through the Harbor capture runner.
The supplemental coding audit lives in [`examples/coding_audit`](../coding_audit/);
its configuration, cloud orchestration and selection policy remain example code.

The native contract is `harbor-semantic-v1`, with schema 2 evidence from OpenEnv's
Harbor semantic validation support. See [RFC 008](../../rfcs/008-environment-auto-validation.md)
and the [pipeline plan](../../docs/plans/hub-environment-validation-pipeline.md)
for its boundaries. Neither corpus admission nor a supplemental coding-audit PASS
establishes full RFC 008 qualification or training benefit.

## Install

From the repository root, with Python 3.12+ and a Linux Docker daemon/Compose:

```bash
uv sync --python 3.13 --group dev
uv pip install 'huggingface_hub==1.33.0'
uv pip install 'harbor @ git+https://github.com/harbor-framework/harbor.git@9c7a39b991d7e2205019f6be0598cbd22d84bfc3'
uv pip check
```

Use `uv run --no-sync` below to preserve that exact Harbor source installation.
The PyPI 0.25.0 wheel lacks the listing API used here; the example checks the
installed Git revision. Its cloud adapter targets `huggingface_hub==1.33.0`.

## Run the synthetic Docker demo

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.demo \
  --output outputs/hub-pipeline-demo
```

The output directory must be new. The demo creates three trusted toy tasks and
runs native Harbor validation in Docker. The expected result is one admitted task
and two exclusions: an always-rewarding verifier and a missing reference solution.
The valid task has NOP reward 0 and oracle reward 1; its declared wrong-answer
canaries must remain at the measured floor. This command makes no model calls.
These controls are synthetic framework examples.

## Choose the qualification profile

`--profile full` is the default. It requires the complete versioned static,
runtime and semantic inventory to pass. Native Harbor coverage remains incomplete,
so missing or skipped required checks block full admission.

Select `--profile harbor-semantic-v1` explicitly for the bounded native path.
Its required results are `static.manifest`, `runtime.startup`,
`runtime.reward_well_formed`, `runtime.episode_determinism`, `semantic.oracle_max`
and `semantic.floor_gap`, plus `semantic.canary_floor` when canaries are declared.
Every required check must PASS. Any FAIL or ERROR, duplicate or missing result,
or required SKIP excludes the task. Other standard checks remain named SKIPs;
the unchanged validator report may therefore retain a WARN verdict.

The driver verifies schema 2 native evidence for fresh NOP, oracle and oracle-replay
Trials, followed by one Trial per declared canary. It checks exact task, retained
image and inspected configuration identities, raw verifier rewards, replay equality,
zero oracle exit codes and independently verified cleanup. It recomputes the
oracle maximum, floor gap and canary floors from the bound raw evidence.

This establishes bounded executable/verifier sanity and exact-submission replay.
It does not establish complete semantic coverage, hostile-code containment,
resource enforcement, network conformance or general verifier correctness.

Supply a trusted reward/resource contract with `--contract`; see
[contract.json](contract.json). Its binary reward range and canary directory are
appropriate for the demo. For another task, review the range/resources and provide
the declared canary scripts, or omit that optional declaration. The contract is an
operator input with a recorded digest. Direct CLI use reads it from
`validation/openenv.json` inside the task:

```bash
uv run --no-sync openenv validate inputs/task --level semantic --policy v2 \
  --json --output outputs/task-report.json
```

The native adapter supports self-contained single-step Linux tasks with one
Dockerfile, shared verifier execution, public networking and no GPU, task-defined
environment variables, healthchecks, external MCP/credential requirements or
task-authored Compose. Unsupported shapes remain explicit SKIPs.

## Search both hubs and validate a small selection

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.pipeline \
  --output outputs/hub-pipeline-search \
  --contract examples/hub_validation_pipeline/contract.json \
  --profile harbor-semantic-v1 \
  --source both --dataset-limit 1 --tasks-per-dataset 1
```

Use a contract appropriate for the selected tasks. HF listing exhausts pagination
before limiting the recorded candidates and checks the exact tag conjunction.
HF repositories are pinned to full commits; Harbor dataset/task packages use full
`sha256:` references. Gated or disabled sources become access failures. Acquisition
preserves native task/verifier bytes and downloads only selected trees and required
shared assets; the runnable profile requires self-contained tasks.

The example uses the HF SDK's `repo_type`, `revision`, `local_dir` and selection
arguments, corresponding to `hf download --repo-type dataset --revision ...
--local-dir ...`. It deliberately resolves mutable source aliases before execution,
uses anonymous public reads (`token=False`) and applies explicit byte/task limits.
Uploads use ordinary SDK commits with explicit additions; they do not mirror-delete
remote files. As with `hf upload`, deletion requires a separate explicit operation.

Explicit inputs replace discovery. An HF repository without `@SHA` is resolved
once to its current commit; explicit Harbor input requires an immutable digest.

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.pipeline \
  --output outputs/hub-pipeline-selected \
  --contract inputs/native-contract.json \
  --profile harbor-semantic-v1 \
  --hf-dataset 'org/repo@FULL_COMMIT_SHA' \
  --harbor-dataset 'org/dataset@sha256:FULL_PACKAGE_DIGEST' \
  --tasks-per-dataset 2
```

`--local-dataset inputs/dataset` accepts already acquired files without Hub access.
The driver copies tasks into run-owned subjects, writes the trusted contract there
and retains both original and effective digests. Failures are excluded per task.

## Evaluate a model on the admitted corpus

Add these options to a search or explicit/local-input invocation:

```bash
  --llm-url "$LLM_URL" --model "$MODEL_ID" \
  --harness opencode --model-timeout 120 --model-step-limit 32
```

Supply the inference credential through `OPENENV_LLM_API_KEY`. Validation workers
receive Docker access and a small environment whitelist; inference and Hub
credentials are not forwarded. Model execution uses the existing capture proxy and
native harness. Default forwarding is `gradio`; `--expose direct` requires sandbox
access to the host capture endpoint.

Admitted tasks are copied into a new corpus with an explicit image/resource overlay.
Before execution, a trusted Trial hook checks the selected task, source digest,
actual image and inspected container configuration. The driver rebuilds indices
and reconciles canonical task IDs; the model runner cannot enumerate excluded
siblings. Reward 0 is a graded model outcome. Model, capture or cleanup failures
remain unsuccessful/incomplete outcomes and never become fabricated zero rewards.

## Outputs and cleanup

Runs retain discovery/task/admission JSONL files, per-task JSON reports and sibling
artifact bundles, run-owned subjects, an approved corpus and `summary.json`.
Requested model runs additionally retain rollouts and native trial/capture evidence.
`harbor-evidence.json`, raw reward files and checksums bind native admission to
its task, image, contract and execution copies.

Cleanup targets exact run-owned Docker containers, networks and volumes and is
independently rechecked. Validated images are retained for model handoff; their
exact tags are recorded in native evidence. Remove only those recorded tags with
`docker image rm <recorded-tag>` when no longer needed. Sources and excluded
reports remain available; reruns use fresh output directories.

## Supplemental coding audit and operator panels

[CODING_AUDIT.md](CODING_AUDIT.md) describes generated candidate references,
operator-reviewed wrong-answer controls, fresh verifier replays, bounded leakage
probes and repeated model attempts. The local controller command is:

```bash
uv run --no-sync python -m examples.coding_audit \
  --task inputs/task --config inputs/audit.json --output outputs/coding-report.json
```

This executes tasks in fresh HF Sandboxes. The `cloud` module prepares immutable
bundles and submits HF Job controllers. `real_tasks` and `full_run` additionally
require `--panel PATH`: no real corpus, reference implementation or negative panel
is embedded in their source. Keep operator panels and populated reports/ledgers
in private operational storage.

The default model configuration records a public source revision and explicit
Inference Provider. Hosted inference does not attest an immutable served-weight
revision. Supplemental reports retain `rfc008_complete: false` and
`trainability_established: false`.

## Focused verification

```bash
PYTHONPATH=.:src:envs uv run --no-sync python -m pytest \
  examples/hub_validation_pipeline/ \
  tests/test_validation/test_harbor_validation.py \
  tests/envs/test_harbor_validated_trial_hook.py -q
```

Injected SDK tests consume no cloud/inference credits. They check source and
artifact binding, incomplete admission, raw rewards, source-mutation races,
model guards and shared-spend gates. Linux Docker execution and live provider
qualification remain separate verification steps.
