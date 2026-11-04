# FineEnvs → validation → online LoRA

This experiment discovers real FineEnvs Hub environments, retains the unchanged
validator's refusals, qualifies a pinned source launch in a dedicated HF Sandbox,
and trains a small LoRA policy with rewards from that environment. Framework
specific training code stays in this example.

The [recorded experiment and measured results](RESULTS.md) include validation
outcomes, the completed LoRA run, failed attempts and remaining limitations.

## Reproduce

Use a Hub account with Jobs credits and read/write access to the three private
output repositories. These commands use current `hf` CLI names. Choose a fresh
Trackio Space for each training attempt.

```bash
hf auth login
hf repos create YOUR_ACCOUNT/fineenvs-experiment --type dataset --private
hf repos create YOUR_ACCOUNT/scottwu-lora --type model --private
hf repos create YOUR_ACCOUNT/scottwu-trackio-RUN_ID --type space --sdk gradio --private

# CPU qualification only; creates and tears down one dedicated Sandbox.
uv run --no-project --with huggingface_hub==2.4.0 \
  examples/hf_validation_lora/bootstrap.py \
  --artifact-repo YOUR_ACCOUNT/fineenvs-experiment

# Qualification, then one L4 Job capped at one hour, then explicit teardown.
uv run --no-project --with huggingface_hub==2.4.0 \
  examples/hf_validation_lora/bootstrap.py \
  --artifact-repo YOUR_ACCOUNT/fineenvs-experiment \
  --model-repo YOUR_ACCOUNT/scottwu-lora \
  --trackio-space YOUR_ACCOUNT/scottwu-trackio-RUN_ID --train
# Add --flavor t4-small for the $0.40/hour T4 alternative.
```

The L4 costs approximately $0.80/hour at the experiment's recorded pricing;
CPU-basic costs $0.01/hour, with exposed-port charges where applicable. Queue time,
package downloads, generation and upload affect elapsed time. The controller
waits for its owned Job, cancels unfinished compute on exit, and saves teardown
state. An abruptly killed controller cannot run its finalizer: use the IDs in
`hf-experiment/state.json` and `hf jobs inspect` / `hf jobs cancel` to reconcile
that run. The environment server receives no HF token. The trusted training
worker receives the credential needed for its private outputs and Sandbox exec.

## Fixed inputs and measured scope

| Input | Pin |
|---|---|
| Validator/framework | OpenEnv `986af4195a1339b9ccb8d84b82cba1cb622c3c46` |
| Environment | `FineEnvs/ScottWuMath@4de8125d314fca0a3f137fa1bab04b966a3efc15` |
| Comparison environment | `FineEnvs/wordle-openenv@a6ca4a7b3ab22052df265d358f41de34ffd351cc` |
| Base model | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| Sandbox image | `python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28` |
| Training image | `python:3.11.18-bookworm@sha256:77c90594edb5d6a43ed73de1c73b5f9d795bc94fa76d438fba4cc00036100014` |
| Environment dependencies | `math-python312.txt`, Python 3.12.13 |
| Training dependencies | `training-python311.txt`, image interpreter Python 3.11.18 |
| Bootstrap tool | `uv==0.9.9` |

Source mounting pins the public Space files. ScottWuMath is installed with the
explicit framework override above; the environment and worker have separate
dependency sets. This qualifies that source bootstrap, rather than the published
Space image. The `immutable_image` refusal in `openenv validate --space` remains.

The untouched CLI refused Wordle because it has no `openenv.yaml`. ScottWuMath
has that file but lacks its `validation:` block and fails static validation.
`qualify.py` supplies a separately identified researcher declaration, without
editing either package. It records schema, reset/step/state and same-session/fresh
episode evidence through the stack's raw collector and graders. It reports five
measured checks and eleven explicit skips, capping success at **WARN**. Resource,
host, network, oracle and full RFC admission claims remain unqualified.

The original frozen Wordle deployment (`openenv==0.4.1`) fails observation-schema
validation: its reset observation lacks required `tool_name`. It is excluded from
training. ScottWuMath passes the five measured checks. A second clean CPU Job
reproduced its setup from the saved scripts and frozen dependencies.

The SDK proxy control reproduced a separate HF transport problem: a subsequent
exec request received the inner uvicorn application's body. `ENV_TRANSPORT=exec`
therefore uses the Sandbox's exec API to batch local WebSocket episodes. No proxy
request occurs in this mode. `score_batch.py` returns public task prompts and raw
terminal rewards; it never returns reference solutions to the worker.

## Learning experiment

ScottWuMath uses its uploaded four-card “make 24” tasks: train indices `0..63`,
held-out test indices `0..15`. Their arithmetic hands do not overlap. Prompts come
only from public reset fields. The model submits one expression; the actual
environment performs its exact arithmetic/card checks and returns its shaped
terminal reward. No local reward replacement, reference-solution SFT or auxiliary
format reward is used.

The worker evaluates the unadapted model, then samples 32 training groups of
four. The frozen signal gate requires two groups with different rewards and at
least 25% server-valid expressions. Failure saves `no_grpo_learning_signal` and
stops without claiming training. Passing enables rank-8 LoRA on `q_proj/v_proj`,
up to 20 GRPO updates, four generations per group, beta zero, temperature 0.8,
128 completion tokens, seed 42, and thinking disabled.

Evaluation repeats the same 16 held-out tasks with greedy decoding and seeds
7001/7002 (offset by task index): 48 paired trials per model. Results include raw
reward, exact/valid rates, missing-trial counts, gradient/optimizer evidence and
changed adapter tensors. The persisted adapter is downloaded, hash checked and
reloaded before evaluation. A short run demonstrates the workflow; improvement
requires the measured paired results and is not assumed.

The private artifact repository contains input scripts/locks, manifests, scoped
validation bundles, rollouts, metrics and cleanup records. The private model
repository contains the adapter/tokenizer. Trackio uses a fresh private Space and
disables automatic public static snapshots.

The initial eight-group pilot found one contrasting group and stopped before
training. The second trial expands the training-only pilot to 32 groups while
keeping its two-contrast / 25% validity thresholds, model, rewards and held-out
tasks unchanged. It also fixes tracker cleanup before initialization: the worker
now owns one explicit private Trackio run across evaluation and training.

The full training image includes matching Python headers and the C compiler
required by TRL's Triton loss kernels. Before model download or evaluation, the
worker runs both selective softmax and the fused head's CUDA backward pass,
checking values and gradients against a PyTorch reference. BF16 requires native
hardware support; a T4 uses FP16 because BF16 emulation cannot run the fused
backward GEMM. Missing toolchains are reported immediately. AMP overflow skips are recorded
separately from effective updates; only finite, contrasting, gradient-driven
weight changes count as training. Failure reports retain bounded stack locations
without exception text or local values.

## Local checks

```bash
uv run --no-project --with huggingface_hub==2.4.0 \
  python -B -m unittest discover -s examples/hf_validation_lora
PYTHONPATH=src:envs uv run pytest tests/test_validation/test_hf_sandbox_provider.py \
  tests/test_core/test_env_client_headers.py -q
```
