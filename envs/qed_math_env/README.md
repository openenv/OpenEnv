---
title: QED Math Environment
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - mathematics
  - proof-evaluation
  - llm-grading
---

# QED Math Environment

Mathematical proof generation and evaluation environment for OpenEnv, ported from [QED-Nano](https://github.com/CMU-AIRe/QED-Nano). Agents receive math problems, submit proofs, and receive LLM-based rubric grading on a 0–7 scale with normalized rewards.

## Features

- **LLM-based rubric grading** (0–7 scale) via any OpenAI-compatible endpoint
- **Process-based answer verification service** (`math_verify` in worker processes)
- **Backpressure + retries + worker restart** for robust concurrent rollout operation
- **Gold-answer cache** keyed by `problem_id` and verifier normalization settings
- **Flexible dataset loading**: local JSONL/JSON, Hugging Face Hub, or built-in bootstrap problems
- **Reward shaping**: discount factor, length penalty, and optional score thresholding
- **Reasoning stripping**: configurable delimiters (e.g. `<think>...</think>`) removed before grading
- **Multi-step problems**: configurable max attempts with per-attempt feedback
- **Verifier metrics**: rollout/staging counters and health signals surfaced in observation metadata, ready for TrackIO / WandB
- **MCP tool interface**: `get_problem`, `submit_proof`, `get_grading_guidelines`

## Quick Start

### Async (default)

```python
import asyncio
from qed_math_env import QEDMathEnv

async def main():
    async with QEDMathEnv(base_url="http://localhost:8000") as env:
        # Reset to load a problem
        result = await env.reset()
        obs = result.observation
        print(f"Problem: {obs.problem[:100]}...")

        # Submit a proof
        submission = await env.submit_proof(proof="By induction on n...")
        print(f"Score: {submission.score}/7, Reward: {submission.reward:.2f}")

asyncio.run(main())
```

### Sync

```python
from qed_math_env import QEDMathEnv

with QEDMathEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    submission = env.call_tool("submit_proof", proof="By induction on n...")
```

### MCP tool-calling

```python
async with QEDMathEnv(base_url="http://localhost:8000") as env:
    await env.reset()

    # Discover tools
    tools = await env.list_tools()
    print([t.name for t in tools])
    # ['get_problem', 'submit_proof', 'get_grading_guidelines']

    # Call tools by name
    problem = await env.call_tool("get_problem")
    guidelines = await env.call_tool("get_grading_guidelines")
    result = await env.call_tool("submit_proof", proof="...")
```

## Building & Running

```bash
# Build Docker image (from project root)
docker build -t qed-math-env:latest -f envs/qed_math_env/server/Dockerfile .

# Run the server
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY qed-math-env:latest

# Or run locally with uvicorn
PYTHONPATH=src:envs uvicorn qed_math_env.server.app:app --port 8000

# Or install and run via uv
cd envs/qed_math_env
uv sync
uv run server
```

## Project Structure

```
qed_math_env/
├── __init__.py              # Module exports (QEDMathEnv, models)
├── models.py                # ProblemObservation, ProofSubmissionObservation
├── client.py                # QEDMathEnv client (MCPToolClient subclass)
├── openenv.yaml             # OpenEnv manifest with metrics declarations
├── pyproject.toml           # Dependencies
├── uv.lock                  # Locked dependencies
├── README.md
├── prompts/
│   └── evaluator_prompts/
│       ├── v0.md            # Evaluator prompt (QED-Nano v0, uses reference solution)
│       ├── v1.md            # Evaluator prompt (QED-Nano v1, default, full 0–7 range)
│       └── v2.md            # Evaluator prompt (QED-Nano v2, scores constrained to {0,1,6,7})
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI server (create_app factory)
    ├── qed_math_environment.py  # QEDMathEnvironment (MCPEnvironment)
    ├── math_verify_service.py   # Process-pool verifier service + health/metrics
    ├── mcp_server.py        # MCP tool registration
    ├── rubric.py            # MathProofRubric + GradingResult
    └── Dockerfile
```

## Configuration

The environment is configured via `QEDMathConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_path` | `None` | Dataset source: local path, Hub ID, or list of specs. `None` uses bootstrap problems. |
| `grader_model` | `"gemini-3-pro"` | Model identifier for the LLM grader (any OpenAI-compatible endpoint) |
| `prompt_name` | `"v1"` | Evaluator prompt template name (`v0`, `v1`, or `v2` in `prompts/evaluator_prompts/`). `v1` matches the QED-Nano default (full 0–7 range); `v2` constrains scores to `{0,1,6,7}` |
| `grader_temperature` | `1.0` | Sampling temperature forwarded to the grader (matches QED-Nano) |
| `grader_max_output_tokens` | `None` | Optional output-token cap for the grader. `None` uses the provider default to avoid truncating the trailing `<score>` tag |
| `custom_reward_threshold` | `False` | When `True`, collapses partial-credit scores 1–5 → 1 |
| `answer_reward_preset` | `"pure_success"` | Answer-mode reward table: `pure_success` (correct→1, else 0) or `base` (adds penalties: wrong −0.5, no_answer/unparsable −1) |
| `max_attempts` | `1` | Max proof attempts per problem (>1 for multi-step) |
| `discount_factor` | `1.0` | Exponential discount: `reward *= discount_factor ** output_length_tokens` |
| `buffer_tokens` | `0` | Length penalty zone width. `0` disables the penalty. |
| `max_tokens` | `0` | Max token limit for length penalty computation |
| `reasoning_delimiters` | `["</think>"]` | Delimiter strings to strip reasoning before grading (matches QED-Nano). Set to `None` to grade the full completion. |
| `verifier_workers` | `max(2, min(8, cpu_count//2))` | Number of process workers used for answer-mode verification |
| `verifier_queue_size` | `verifier_workers * 32` | Max in-flight verifier requests before backpressure |
| `verifier_request_timeout_seconds` | `5.0` | Per-request client-side timeout when awaiting worker response |
| `verifier_max_retries` | `1` | Retry budget for transient verifier infra failures |
| `verifier_strict` | `True` | Strict `math_verify` equivalence mode |
| `verifier_numeric_precision` | `5` | Numeric precision setting used in verifier request contract |
| `verifier_float_rounding` | `10` | Float rounding setting used in verifier request contract |

Environment variables:
- `OPENAI_API_KEY` — API key for the grader LLM
- `OPENAI_BASE_URL` — Base URL override (for non-OpenAI providers)

## Dataset Format

### Local JSONL/JSON

```json
{
  "problem": "Prove that the sum of two even integers is even.",
  "solution": "Let a=2m and b=2n. Then a+b=2(m+n), which is even.",
  "rubrics": [
    {"title": "Definitions", "points": 2, "desc": "Correctly defines even integers."},
    {"title": "Algebra", "points": 3, "desc": "Valid algebraic manipulation."},
    {"title": "Conclusion", "points": 2, "desc": "Correctly concludes evenness."}
  ],
  "dataset": "FineProofs-RL",
  "problem_id": "fp_001"
}
```

### Hugging Face Hub

```python
QEDMathConfig(dataset_path="meta-math/MetaMathQA")
# or with config
QEDMathConfig(dataset_path={"hub_id": "meta-math/MetaMathQA", "split": "train", "config": "default"})
```

### Field Aliases

The environment normalizes many dataset formats automatically:

| Canonical Field | Accepted Aliases |
|----------------|------------------|
| `problem` | `task`, `Problem` |
| `reference_solution` | `solution`, `answer`, `Solution` |
| `grading_guidelines` | `rubrics`, `schema`, `schema_0`, `Grading guidelines`, `details` |
| `problem_id` | `id` |
| `original_problem` | Used for RC-stream problems where the actor prompt differs from grading prompt |

## Observation Space

### ProblemObservation (from `reset` / `get_problem`)

| Field | Type | Description |
|-------|------|-------------|
| `problem` | `str` | Math problem statement |
| `reference_solution` | `str` | Ground-truth solution |
| `grading_guidelines` | `str` | Rubric / marking scheme |
| `problem_id` | `str` | Unique identifier |
| `problem_type` | `str` | `"proof"`, `"answer"`, or `"multi_step"` |
| `dataset_source` | `str` | Source dataset name |
| `metadata` | `dict` | Additional context (e.g. `original_problem`) |

### ProofSubmissionObservation (from `submit_proof`)

| Field | Type | Description |
|-------|------|-------------|
| `proof` | `str` | Submitted proof text |
| `score` | `int` | Raw grade (0–7) |
| `feedback` | `str` | Full grader response |
| `reward` | `float` | Shaped reward in [0, 1] |
| `done` | `bool` | Whether the episode is over |
| `is_correct` | `bool` | Whether score >= success threshold (default 7, matching QED-Nano's `score == 7`) |
| `attempt_number` | `int` | Current attempt count |
| `attempts_remaining` | `int` | Remaining attempts |
| `problem_type` | `str` | Problem type |
| `metadata` | `dict` | Contains `verifier_metrics`, `base_reward`, `shaped_reward` |

## MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_problem` | Return current problem statement and metadata | — |
| `submit_proof` | Submit a proof for LLM-based rubric grading | `proof` (str, required) |
| `get_grading_guidelines` | Return the rubric/marking scheme | — |

> **Note:** `output_length_tokens` is **not** an agent-supplied parameter. Token counts are
> injected by the training harness via the HTTP step request body (see [Reward Shaping](#reward-shaping))
> to preserve reward integrity — the agent cannot influence its own discount factor.

## Reward Shaping

The reward pipeline follows QED-Nano conventions:

1. **LLM grading**: Score 0–7 via evaluator prompt with `<score>N</score>` parsing
2. **Optional thresholding**: Collapses 1–5 → 1 (when `custom_reward_threshold=True`)
3. **Normalization**: `reward = score / 7.0`
4. **Discount factor**: `reward *= discount_factor ** output_length_tokens`
5. **Length penalty**: Linear penalty when output approaches `max_tokens`

For answer-mode problems (`evaluation_mode: "answer"`), grading is routed through the process-based verifier service: `\boxed{}` answers are extracted and verified against cached gold answers, with timeout/retry/backpressure handling for concurrent rollouts. The answer-mode reward is selected from `answer_reward_preset` keyed on the verifier status (`correct`, `wrong`, `no_answer`, `unparsable`); transient `timeout`/`internal_error` statuses stay neutral (0).

**Proof vs. answer routing:** when a dataset row does not set an explicit `problem_type`/`evaluation_mode`, the mode is auto-detected like QED-Nano's `if "schema" in problem` — a row carrying a grading rubric/schema is graded as a **proof** (LLM judge), while a row with no rubric is treated as an **answer** problem (boxed gold + `math_verify`). Set `problem_type`/`evaluation_mode` explicitly to override.

### Harness-injected token count

Steps 4 and 5 require the full generation length (including any reasoning trace that is stripped before grading). This value cannot come from the agent — it is supplied by the training harness as an out-of-band field in the HTTP step request body, mirroring the [`StateUsageTracker`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/verifiers/utils/usage_utils.py) pattern from PrimeIntellect/verifiers:

```python
# Training harness (pseudocode)
completion_tokens = llm_call.usage.completion_tokens  # from inference API

step_response = await openenv_client.step(
    action=CallToolAction(tool_name="submit_proof", arguments={"proof": proof_text}),
    output_length_tokens=completion_tokens,  # injected here, not via MCP tool
)
```

When `output_length_tokens` is absent (local testing, eval without a training loop) shaping is skipped entirely — no estimation is attempted, consistent with verifiers' behaviour of returning `None` from `StateUsageTracker.snapshot()` when no usage was recorded.

## Verifier Metrics

Every `submit_proof` call emits verifier metrics in `metadata["verifier_metrics"]`, compatible with TrackIO and WandB:

| Metric | Description |
|--------|-------------|
| `verifier/rollouts/success` | 1 if grading succeeded |
| `verifier/rollouts/failure` | 1 if grading failed |
| `verifier/failures/timeout` | Count of timeout errors |
| `verifier/failures/rate_limit` | Count of rate-limit errors |
| `verifier/failures/no_input` | 1 if proof was empty |
| `verifier/failures/no_score_tag` | 1 if LLM response had no `<score>` tag |
| `verifier/failures/all_attempts_failed` | 1 if all retries exhausted |
| `verifier/failures/num_retries` | Number of retries used |
| `verifier/runtime/latency_per_request` | Grading wall-clock time (seconds) |
| `verifier/requests/count` | Total verifier requests processed by the service |
| `verifier/requests/latency_ms` | Service-level average request latency |
| `verifier/requests/timeout_count` | Service-level timeout counter |
| `verifier/requests/error_count` | Service-level internal error counter |
| `verifier/queue/depth` | Current in-flight verifier queue depth |
| `verifier/cache/hit_rate` | Gold-answer cache hit rate |
| `verifier/workers/restart_count` | Worker-pool restart count |
| `verifier/workers/worker_restarted` | 1 if current request required worker restart |
| `verifier/workers/heartbeat_lag_ms` | Time since last verifier activity |
| `verifier/runtime/input_tokens` | Grader input tokens (real provider usage when reported, else ~chars/4 estimate) |
| `verifier/runtime/output_tokens` | Grader output tokens (real provider usage when reported, else ~chars/4 estimate) |
| `reward/base` | Pre-shaping reward |
| `reward/shaped` | Post-shaping reward |
| `reward/score_raw` | Raw integer score (0–7) |
| `reward/overlong_penalty` | Length penalty applied |
| `episode/attempt_number` | Current attempt |
| `episode/is_correct` | 1 if correct |
| `episode/problem_type` | proof / answer / multi_step |
| `episode/dataset_source` | Source dataset name |

### TrackIO Integration

```python
import trackio

run = trackio.init(project="qed-math-training")

# After each submit_proof call:
verifier_metrics = result["metadata"]["verifier_metrics"]
numeric = {k: v for k, v in verifier_metrics.items() if isinstance(v, (int, float))}
run.log(numeric, step=global_step)
```

Or with TRL's GRPOTrainer:

```python
from trl import GRPOConfig

config = GRPOConfig(
    report_to="trackio",
    trackio_space_id="your-org/qed-math-grpo",
    # ...
)
```

## Deployment

```bash
# Optional: run rollout/staging verifier validation first
PYTHONPATH=src:envs uv run python scripts/qed_math_verifier_staging_validation.py \
  --workers 4 --queue-size 128 --concurrency 64 --requests 2000 \
  --max-timeout-rate 0.05 --max-error-rate 0.02

openenv push
```