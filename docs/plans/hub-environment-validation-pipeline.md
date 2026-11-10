# Hub discovery, native validation and model execution

The runnable implementation is [the Hub validation example](../../examples/hub_validation_pipeline/README.md).
It acquires immutable Hugging Face or Harbor Hub task sources, invokes OpenEnv's
native validator, admits tasks against an explicit profile, and optionally runs a
configured model on a new corpus containing only admitted tasks. Source packages
and exclusions are retained. This document describes the reusable implementation
and its qualification boundary, without operational corpus or run records.

Native semantic validation uses the `harbor-semantic-v1` contract and schema 2
evidence introduced by [#1591](https://github.com/openenv/OpenEnv/pull/1591).
The [supplemental coding audit](../../examples/hub_validation_pipeline/CODING_AUDIT.md)
is example code in `examples/coding_audit`, with orchestration in
`examples/hub_validation_pipeline`. Neither a bounded native admission nor a
supplemental audit establishes complete RFC 008 coverage or training benefit.

## Pipeline and ownership

| Stage | Responsibility | Retained output |
| --- | --- | --- |
| Discover | Query both hubs, exhaust pagination and freeze candidate identities. | Candidate metadata and source aliases |
| Acquire | Resolve immutable identities and bound selected downloads. | Original task files, inventories and hashes |
| Enumerate | Identify task roots and preserve ordered canonical identities. | Per-task source records |
| Validate | Use OpenEnv's parser, native collector and standard graders. | JSON reports, raw evidence and checksums |
| Admit | Verify required PASS results and provenance for the selected profile. | Accepted/excluded records with reasons |
| Materialize | Copy admitted tasks with explicit image/resource overlays. | Frozen corpus and index-to-task mapping |
| Evaluate | Use the existing Harbor harness and capture runner. | Every attempted outcome and cleanup evidence |

Core native validation remains in OpenEnv. Discovery, admission policy, cloud
launching, model selection, ratebooks and research experiments remain in examples.
A sibling sandbox or outer `harbor_env` server cannot establish properties of the
actual task sandbox. Every execution copy must bind to its acquired source and
validated image/configuration.

## Acquisition and HF conventions

HF discovery uses both `library:harbor` and `rl-environment`, checks those tags in
returned metadata, and consumes listing pagination before applying a candidate
limit. Trending rank selects discovery priority; it is not a stable task identity.
HF sources require full commit revisions and Harbor packages require full `sha256:`
references. Resolve a mutable HF alias once and record the resulting commit before
acquisition. Never infer byte identity from matching dataset names or task counts.

The SDK's `snapshot_download`/`hf_hub_download` arguments correspond to
`hf download --repo-type dataset --revision ... --local-dir ...`. The example
adds anonymous public-source reads (`token=False`), explicit selected-file/task
bounds and immutable revision checks. It materializes regular files for Harbor
rather than relying on cache links. Gated, disabled, ambiguous or unsupported
sources are exclusions; they do not silently fall back to another registry.
Harbor's Git source pin is checked because a matching package version alone does
not establish that the required listing API exists.

Check Git blob/content hashes and declared file sizes, reject escaping paths,
symlinks, special files and secret-file inputs, and retain the original acquired
inventory. Parsing/acquisition never executes downloaded task scripts on the
controller host. Source revisions do not pin container bases, package downloads
or external services; those need separate image/runtime evidence.

HF publication uses explicit SDK add operations and retained immutable commit IDs.
It does not synchronize a local folder by deleting remote files. This follows the
`hf upload` convention that deletion is explicit. The cloud workflow additionally
requires an existing private result dataset and a fresh content-derived run claim;
ordinary upload success does not authorize duplicate paid execution.

## Native semantic admission

The default `full` profile requires the complete pinned static/runtime/semantic
inventory. Missing or skipped required coverage prevents full admission.
The explicitly selected `harbor-semantic-v1` profile requires:

- `static.manifest`, `runtime.startup`, `runtime.reward_well_formed` and
  `runtime.episode_determinism` to PASS.
- `semantic.oracle_max` and `semantic.floor_gap` to PASS, plus
  `semantic.canary_floor` when the operator declares canaries.
- No FAIL or ERROR anywhere in the report; no duplicate, unknown or missing
  required result, and execution of all required validation levels.
- Schema 2 native evidence for fresh NOP, oracle and oracle-replay Trials,
  followed by the declared canaries, all with the same retained image and inspected
  configuration, unchanged task identities and verified cleanup.
- Bound raw reward files and successful oracle exits. Recompute replay equality,
  the declared oracle maximum, the NOP gap and canary floors from those files.

Other standard checks remain named SKIPs and can leave the underlying report WARN.
The admission envelope records these limits beside the unchanged report; it does
not turn the report into full RFC qualification. Missing required evidence is
incomplete, not PASS, and any confirmed failure keeps FAIL precedence.

The contract lives at `validation/openenv.json` in the validation subject and
explicitly declares the reward key/range, tolerance, floor margin, resource
allocation and optional canary directory. A pipeline-supplied overlay records
both the original source digest and the effective validation digest. The current
native adapter supports self-contained, single-step Linux tasks with one
Dockerfile, shared verifier execution and public networking. Unsupported resource,
credential, multi-service and extension contracts stay named SKIPs.

```bash
uv run --no-sync openenv validate inputs/task --level semantic --policy v2 \
  --json --output outputs/task-report.json

uv run --no-sync python -m examples.hub_validation_pipeline.pipeline \
  --contract inputs/native-contract.json --profile harbor-semantic-v1 \
  --hf-dataset 'org/repo@FULL_COMMIT_SHA' --tasks-per-dataset 1 \
  --output outputs/native-selection
```

Use [the README](../../examples/hub_validation_pipeline/README.md) for installation,
Docker requirements and the synthetic demo. The generic example contract is not
a substitute for reviewing another task's reward/resource/canary declarations.

## Safe model handoff and evidence

Build a new run-owned corpus from admitted tasks only. Record its ordered mapping
from transport index to canonical task ID; filtering must not change identities
or expose excluded siblings. Materialize only the validated immutable image and
explicit resource overlay. A trusted pre-execution Trial hook checks each task
copy, source digest, actual image and inspected container configuration.

Pin the model/harness/backend and sampling, token, step and time limits. Hosted
model IDs and returned fingerprints describe available provider evidence; a
public source revision does not attest the weights served remotely. Keep inference
credentials outside validation workers and task arguments.

Record reward maps, the selected headline metric, execution status, errors,
traces, timing and cleanup. A valid reward of zero is a model outcome. Missing
rewards, failed infrastructure and partial result sets remain ungraded/incomplete.
Reconcile expected and returned task IDs. Do not replace failures with zero or
delete their source/evidence. Clean up exact run-owned containers, networks and
volumes; retained image tags are recorded for explicit later removal.

## Supplemental coding audit

`examples/coding_audit/{scoring.py,hf.py,runner.py,__main__.py}` contains the
operator-run `harbor-coding-audit-v1` workflow. It generates bounded candidate
references, repeats exact submissions against fresh verifier instances, tests
reviewed known-wrong cases, probes limited leakage/stale-state behavior and then
measures repeated target attempts. Cloud and pipeline orchestration import this
example package. The controller command is:

```bash
uv run --no-sync python -m examples.coding_audit \
  --task inputs/task --config inputs/audit.json --output outputs/audit-report.json
```

Reference solutions, verifier-private files, known-wrong submissions and their
counterexamples remain operator-controlled evaluation inputs. Real-task recipes
require `--panel PATH`; they contain no built-in benchmark selection or negative
panel. The operator panel declares a pinned HF source, task paths, immutable
execution boundaries and complete audit configuration. Its exact byte snapshot
is read once and SHA256-bound to preparation and bundle provenance. Acquisition
preserves every native task/verifier byte.

```bash
uv run --no-sync python -m examples.hub_validation_pipeline.real_tasks \
  --panel "$PANEL_PATH" --output outputs/native-recipes --bundle

uv run --no-sync python -m examples.hub_validation_pipeline.full_run \
  --panel "$PANEL_PATH" --hf-census "$HF_CENSUS_PATH" \
  --harbor-census "$HARBOR_CENSUS_PATH" --budget-usd "$BUDGET_USD" \
  --output outputs/native-plans
```

These commands prepare inputs without paid execution. Operator-selected
`--task NAME` values are validated against the supplied panel. A metadata census
is an inventory, not a deduplicated or runtime-qualified corpus. Evaluation-only
panels never become training data.

Paid native-cohort submission additionally requires a shared initialized ledger,
a reservation identifier and `--submit`. The aggregate estimate reserves the
configured request/token ceilings, controller allowance and each child's remote
maximum lifetime before the first submission. The fixed ratebook is a declared
modeling assumption; it is neither a current-price guarantee nor an account-enforced
billing cap. Every initial budget-bearing plan parse and final submission checks
the same expected plan-byte SHA256. Partial or uncertain receipts and unresolved
reservations remain durable for manual reconciliation; launchers never blindly
resume or retry.

## Qualification and research boundary

Keep source-level tests, Linux Docker evidence and live-provider qualification
separate. Framework fakes exercise binding, parsing, incomplete outcomes, budget
races and cleanup accounting; they do not establish a provider's execution or
billing behavior. Supplemental reports always retain `rfc008_complete: false`
and `trainability_established: false`.

Verifier acceptance/rejection controls measure the reviewed panel, not complete
specification correctness. Repeated attempts measure policy-dependent difficulty;
they do not demonstrate learning. Training value requires a separate controlled
experiment with training-permitted data, sealed held-out task families, matched
budgets and independent evaluation. See [the trainability research plan](hub-environment-trainability-validation.md).
