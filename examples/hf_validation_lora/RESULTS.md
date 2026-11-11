# Recorded experiment — 2026-11-04

The Hub discovery → scoped validation → online LoRA → persisted-adapter
evaluation workflow completed on Hugging Face Jobs and dedicated Sandboxes.
This is a small workflow experiment, with no exact held-out solutions and no
claim of general benchmark improvement.

## Validation

Seven real FineEnvs servers were inventoried. Two lightweight candidates were
run; source revisions and dependency locks are in [README.md](README.md).

| Environment | Unchanged CLI result | Dedicated source-bootstrap result | Training |
|---|---|---|---|
| ScottWuMath | Static FAIL: missing `validation:` block | WARN: 5 PASS, 11 SKIP | Selected |
| Wordle | Refused: no `openenv.yaml` | FAIL: reset observation omits required `tool_name`; four other measured checks pass | Excluded |

The five measured checks are startup, reward well-formedness, observation schema,
state contract and episode isolation. Researcher declarations were recorded
separately without editing either Hub source. A second clean CPU Job reproduced
the ScottWuMath bootstrap and qualification from saved scripts and locks.

These are source-bootstrap results. Published Space image provenance, resource,
host, network, oracle and full RFC admission remain unqualified. The validation
provider's `immutable_image` gate remains intact.

## Training and evaluation

[Completed T4 Job](https://huggingface.co/jobs/burtenshaw/6aebb1f20e0338d1bb80427c)
ran `scottwu-20261104T223055Z` in Python 3.11.18 with CUDA 12.8. The worker
completed in 576 seconds, excluding image/package setup. Inputs:

- Base: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
- Environment: `FineEnvs/ScottWuMath@4de8125d314fca0a3f137fa1bab04b966a3efc15`.
- Worker/lock artifact revision: `66e1bdb3e0e718532e435d7a8d30e6fdbc4355fd`.
- Training worker source at local commit `8448bf35`; later dashboard setup fixes
  are separate. The recorded run's dashboard files were installed explicitly.
- Rank-8 LoRA on `q_proj/v_proj`, 1,146,880 trainable parameters, FP16, seed 42,
  vanilla GRPO with group-normalized raw terminal rewards and beta zero.
- 64 eligible training tasks; 16 held-out task indices with 15 unique number
  multisets and no train/test multiset overlap. No reference solutions were
  passed to the policy. The actual environment graded every expression.

The 128-completion training-only pilot passed the frozen thresholds with three
contrasting groups and 35.9375% valid expressions. Training ran 20 optimizer
steps. Worker telemetry reports **four** with positive finite gradients, reward
contrast, advancing optimizer counters and changed weights. Zero-contrast steps are excluded from
that count, including changes caused by optimizer momentum. No AMP step was
reported skipped in this successful run. These per-step state observations were
not independently replayed; final adapter changes alone cannot establish the
per-step count.

| Held-out metric | Base | Reloaded LoRA |
|---|---:|---:|
| Completed trials | 48/48 | 48/48 |
| Valid expressions | 13/48 (27.1%) | 24/48 (50.0%) |
| Exact solutions | 0/48 | 0/48 |
| Mean raw reward | 0.0021267292 | 0.0023003333 |

Each of the 16 held-out hands uses greedy decoding and two fixed sampling seeds
before and after training. The mean paired reward difference is **+0.0001736042**:
two trials improved, one worsened and 45 tied. These 48 trials are three decodes
of 16 tasks, not 48 independent tasks. The small absolute reward change and zero
exact solutions do not establish robust task improvement.

All 112 adapter tensors changed; 56 LoRA B tensors are nonzero. The worker
uploaded, downloaded and hash-checked six model files, reloaded the persisted
adapter, verified its weight hash, then evaluated it. A separate CPU readback
recomputed the rollout metrics, checked the script/manifest/source pins, all six
file hashes and finite tensor values, and reproduced the final adapter hash.
Per-step gradients and optimizer states were not saved. Update counts and live
reload verification come from the pinned worker's telemetry, corroborated by
the final weights and saved rewards; they were not independently replayed.

## Friction and corrections

Feature changes stay narrow:

- HF Jobs hardware preflight is deferred until runtime validation. Static,
  skip-build and invalid-manifest reports no longer require Jobs hardware access.
- Authenticated Sandbox proxy HTTP/WebSocket redirects are refused, preventing
  credentials from being forwarded to redirect destinations.

The reproducible example also incorporates observed fixes:

- A full GPU image, matching interpreter/headers and compiler, plus the actual
  fused-head forward/backward preflight before model download and evaluation.
- Native BF16 detection. T4 emulation passed PyTorch's default support query but
  failed TRL's BF16→FP32 backward GEMM; FP16 passed the same real-GPU control.
- AMP overflow recovery and strict optimizer evidence, tracker ownership and
  final cancellation polling. Those bugs have independent CPU regressions.
- A fresh private Trackio Space and bucket, explicit app deployment and standard
  OAuth metadata. Pre-creating an empty Space alone did not deploy its app.

The HF proxy/exec routing control still reproduced an upstream transport fault.
Training therefore used the Sandbox exec API to batch local WebSocket episodes;
it made no authenticated port-proxy requests. This workaround does not close the
upstream transport bug. Missing environment manifests/schema fields and trusted
Space commit-to-image binding remain follow-up work.

Earlier attempts are retained: an eight-group pilot correctly refused training;
subsequent runs exposed compiler, interpreter-header and emulated-BF16 failures.
Those attempts produced no trained adapter. Their reports remain separate from
the successful run, rather than being counted as successful training.

## Evidence and checks

- [Private artifacts](https://huggingface.co/datasets/burtenshaw/openenv-fineenvs-e2e-20261104):
  discovery, qualification bundles, scripts/locks, rollouts, reports, diagnostics,
  reviews and cleanup records. Successful report revision:
  `1614f4432e47a6a14bc8348bdd68d1a336b0874e`.
- [Private adapter](https://huggingface.co/burtenshaw/openenv-scottwumath-qwen3-06b-lora-20261104),
  evaluated at `affa708b4618dcc9439ffa3abf84a050f5b1c703`.
- [Private dashboard](https://huggingface.co/spaces/burtenshaw/oe-scottwu-lora-20261104-r6):
  private backing bucket verified; serving Trackio with durable metric records.
- Production validation/CLI/proxy gate: **1047 passed, 8 skipped, 44 deselected**.
  Latest example gate: **19 CPU tests passed**; scoped lint/format checks pass.
  Counts overlap and are not additive.
- All **25 owned Jobs/Sandboxes are terminal**. The private dashboard is retained
  on free CPU hardware. No GitHub push, PR, merge or public message was made.

The clean CPU controller was exercised live. The successful GPU worker was
submitted through the HF Jobs connector against the already qualified Sandbox;
the single `bootstrap.py --train` command has not itself been run end to end.
Reproduction pins software and inputs; it does not promise bitwise equality
across different GPU architectures or drivers.
