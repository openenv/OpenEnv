# RFC: Environment Services

**Status**: In Review
**Created**: 2026-10-16
**Authors**: @rycerzes
**RFC ID**: 014

Discussion: [#1382](https://github.com/huggingface/OpenEnv/issues/1382). Prototype: [`envs/posttrain_env`](../envs/posttrain_env/).

## Summary

More environments give the agent real services instead of in-env tools: post-training tasks that train models on rented GPUs, hill-climbing tasks that call an evaluator on dedicated hardware, simulators and render farms. Today each environment decides alone who holds the credentials, who pays, what happens to a job that outlives the episode and whether the agent can reach the grader.

This RFC defines **Environment Services**: a brokered layer between the agent and external services, with three rules. The agent never holds a real credential. The agent never reaches the grader. Nothing is spent without a reservation. v1 covers two service profiles, **batch job** and **evaluator**, and a **per-rollout budget**, which is enough for post-training and kernel optimisation. `envs/posttrain_env` implements v1 on HF Jobs without core changes.

## Motivation

### Problem Statement

Environments answer five questions ad hoc:

1. **Who holds the credential?** Raw keys reach the agent environment. Examples: Harbor `environment.env`, Prime sandbox `secrets`, and OpenHands exporting secrets into the agent's shell.
2. **Who pays?** Existing caps cover LLM tokens only and are checked after spending, so they overshoot (OpenHands `max_budget_per_run`, LiteLLM MCP cost, RSIGym). Nothing defines a budget per rollout, per group or per run.
3. **What happens to a job that outlives the episode?** There is no job handle, no cancel on reset and no orphan handling. `kernrl` runs its GPU evaluator inside `step`.
4. **Can the agent reach the grader?** [ROLL](https://github.com/alibaba/ROLL/blob/581046abce143e643221f8e292895c46a3b20112/roll/pipeline/agentic/env/rock/sanbox_manager.py#L1230-L1332) and [rLLM](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/eval/script_evaluator.py#L73-L94) upload tests into the agent's sandbox after the episode; in AReaL a rollout's own key can set its reward. METR measured [reward hacking in 30.4% of RE-Bench runs vs 0.7% on HCAST](https://metr.org/blog/2025-06-05-recent-reward-hacking). #1232 and #1357 raise the same concern for OpenEnv.
5. **What was called, and what did it produce?** No framework links service calls to trajectory steps.

### Goals

1. A service contract an environment declares, instead of per-env plumbing.
2. Credentials that stay with a broker; the agent and its jobs hold none.
3. A per-rollout budget that reserves before spending and settles after, published in a stable form so trainers can account for spend and plan budgets and concurrency.
4. Jobs with handles, cancellation at episode end and a defined orphan state.
5. Evaluation split into an agent-callable dev endpoint and a grader-only score.

### Non-Goals

- Hosting services. Brokers wrap existing ones (HF Jobs first).
- Profiles beyond batch job and evaluator in v1 (see [Future Work](#future-work)).
- Changing `/mcp` in this RFC. The move to MCP 2026-07-28 is a later RFC 003 amendment (decision 4).

## Design

### Architecture Overview

```
agent sandbox ──MCP tools──▶ environment / openenvd (per env)
 (no credentials)               ledger · artifacts · audit · token exchange
                                     │ broker credential, audience-bound
                                     ▼
                          shared brokers: jobs (HF Jobs) · eval
grader ◀── artifact handles + score jobs ──┘
```

Without openenvd (RFC 009), the environment server is the broker, as in `posttrain_env`. With openenvd, the daemon holds the broker credential as a backend credential and the environment only sees run tokens.

### Core Abstractions

**Service manifest.** An environment declares its services by profile:

```yaml
services:
  - {name: jobs, profile: batch_job, surface: agent, flavors: [a10g-small], max_timeout: 60m, meter: [usd, gpu_s]}
  - {name: kernel_eval, profile: evaluator, surface: {agent: dev_evaluate, grader: score},
     hardware: {class: h100, exclusive: true}}
budget: {usd: 3.0, scope: rollout}
```

`posttrain_env` keeps this in its task definitions (`server/tasks.py`); the manifest becomes a core type in step 2 of the plan below.

**Service profiles (v1).**

| Profile | Examples | Adds to the contract |
|---|---|---|
| batch job | HF Jobs; later Modal, Slurm, render farms | job handle, worst-case reservation, cancel, orphans |
| evaluator | held-out model eval, kernel timing | `dev_evaluate` vs `score`, exclusive hardware, cache |

**Budget ledger.** Money is integer micro-USD. Before a job starts the broker reserves its worst case, the flavor's per-minute price times the job's timeout, and refuses the job with a typed `budget_exhausted` error if it does not fit. When the job ends the reservation is replaced by the real cost, the per-minute price times the minutes the job ran, rounded up to whole minutes as HF Jobs bills. Budgets are per rollout and set by the orchestrator, so every rollout in a GRPO group faces the same limit. The agent sees its remaining budget as data, never as reward. The ledger is the broker's estimate; the provider's bill is authoritative, and a provider-side cap is the backstop (see Credentials).

Every observation carries a stable spend entry. It is for accounting and for planning budgets and concurrency, never a sampling weight: dividing a task's sampling weight by its cost would starve expensive, hard tasks. RFC 013's sampler never reads the environment; the trainer reads this entry from the final observation and can report one value per rollout as `costs` to `update(task, rewards, costs)`.

```python
obs.metadata["service_spend"] = {
    "version": 1,
    "settled": {"usd": 0.62, "gpu_s": 410.0, "jobs": 2, "calls": 1},  # cumulative for the episode
    "reserved": {"usd": 0.0},  # open reservations; empty once every job has settled
    "final": True,             # false while a job is live or was orphaned
}
```

Units are `usd`, `gpu_s`, `cpu_s`, `tokens`, `calls` and `jobs`, in base units. `jobs` counts agent jobs and `calls` counts dev evaluations. Grader evaluations are not charged to the agent.

**Jobs.** Tools return a handle. States are `queued → running → succeeded | failed | cancelled`, plus `cancelling` while a cancel is pending and `orphaned` when the provider does not confirm one in time. An orphaned job is settled at its full reservation, keeps `final: false` and blocks the next `reset` with an error, because it may still be running and billing. `reset`, `close`, `submit`, the step limit and the time limit cancel the episode's live jobs. Submissions take an idempotency key; a retry with the same key returns the same job and makes no second reservation.

**Evaluator.** Two endpoints. `dev_evaluate` runs on the public dev split, is rate-limited and charged to the agent, and returns at most five metrics and short feedback. `score` runs on the hidden split as a grader-only job and is the reward. Both run the environment's evaluator script as their own job, with the artifact under test mounted read-only. The evaluator prints one `POSTTRAIN_SCORE {"score": ...}` line; anything else, including a score outside [0, 1], counts as `invalid` with score 0. The evaluator takes only data from the artifact, never code or anything that shapes the evaluator's own input: `t0_arithmetic_format` loads safetensors weights into the base model's config, with the base tokenizer and chat template at a pinned revision, since a chat template could compute the answer into the prompt. Its dependencies are pinned too, so scores don't change with a new release.

**Artifacts.** An artifact is the output folder of a succeeded job, exposed as `openenv-artifact://<episode>/<job>/outputs` with each file's path, size and content hash. Graders mount the folder read-only; nothing is copied through the agent.

### Key Design Decisions

#### D1. The broker is the only credential holder

- **Chosen approach**: The broker starts jobs with `HfApi.run_uv_job`, with explicit flavor, namespace, timeout and labels and an empty `secrets` map. It never runs `hf jobs uv run`, whose script header can pull the caller's `HF_TOKEN` into the job and set the namespace; the Python API [ignores that header](https://github.com/huggingface/huggingface_hub/blob/bb2ef7579c3d2dc2eb49b16ccf15a070fe5b7023/src/huggingface_hub/hf_api.py#L12665-L12668).
- **Rationale**: A fine-grained token with only `job.write` cannot touch repos, but a job holding it can start further jobs that skip the broker's budget checks (verified on HF Jobs). So the token never enters a job.
- **Broker token**: "Start and manage Jobs" plus write access to the namespace's `jobs-artifacts` bucket. A `job.write`-only token is refused (HTTP 403) when the broker uploads the job script to that bucket.
- **Backstop**: HF tokens carry no spend cap. Billing broker jobs to a Resource Group with a Jobs spend limit (Enterprise) caps the total; over the limit, HF rejects new jobs and cancels running ones.

#### D2. Artifacts move through the HF Jobs artifacts bucket

- **Chosen approach**: Follow `run_uv_job`'s own convention. It uploads the script to the namespace's private `jobs-artifacts` bucket and mounts that job's folder read-write at `/data`. Jobs write outputs to `/data/outputs/`. Evaluator jobs mount `<folder>/outputs` read-only at `/model`.
- **Rationale**: No token inside the job, no extra repository, and HF's existing behaviour. Content hashes come from the bucket listing (Xet hashes).
- **Trade-offs**: Every agent job and evaluation creates a bucket folder; cleanup follows the bucket's retention.

#### D3. Tool names use an underscore prefix

- **Chosen approach**: `jobs_submit`, `jobs_status`, `jobs_wait`, `jobs_logs`, `jobs_cancel`, `dev_evaluate`, `submit` and `budget`.
- **Rationale**: OpenAI-compatible function names allow only `[a-zA-Z0-9_-]`, and most trainers convert MCP tools to that form, so `jobs.submit` breaks them. The reserved-name prefix rule in core (step 2) covers `jobs_`.

#### D4. Long calls block with a declared timeout

- **Chosen approach**: `submit` waits for the score job and returns the reward in the same step; `jobs_wait` waits up to a capped number of seconds. The environment raises its tool timeout for these calls and reports `score_timeout_s` at `reset` so clients can set their message deadline (the same need as #1026).
- **Rationale**: The reward stays inside the environment (RFC 004) and the episode ends with it, without a side channel.

### Decisions from the discussion

1. **Long-running services and GRPO.** A service runs live at train time only if it finishes within the trainer's staleness window; for example, TRL's experimental `AsyncGRPOTrainer` drops samples more than [`max_staleness`](https://github.com/huggingface/trl/blob/0dcd18c48bf7e63b49a23da0cb030b0a93a2f39f/trl/experimental/async_grpo/async_grpo_config.py#L126-L128) weight updates old (default 4). Slower services declare a stand-in and the real service runs only at eval. Post-training tasks are multi-hour episodes, so their near-term use is evaluation.
2. **HF Spaces.** No services on Spaces for now. A Space gives no isolation guarantee, so it must not hold a broker credential.
3. **Shared brokers are operator-run services outside the environment**, with the standing of RFC 009's backend gateways. The orchestrator's run config lists each broker's URL and expected TLS identity; that registration is the trust anchor. The daemon exchanges its broker credential for per-rollout tokens bound to one broker ([RFC 8693](https://www.rfc-editor.org/rfc/rfc8693), [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707)); brokers are MCP resource servers ([RFC 9728](https://www.rfc-editor.org/rfc/rfc9728)) that enforce a hard cap per token. This follows RSIGym, LiteLLM virtual keys and OpenMLE Sandbox.
4. **Transport targets MCP 2026-07-28**, which [removes protocol sessions](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/86f6e0c675afea9a8f233e09b6faf566b126086c/docs/specification/2026-07-28/changelog.mdx) in favour of server-minted handles passed as arguments ([SEP-2567](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2567)). That matches `openenv/session/create` and the job handles above. The upgrade adds `server/discover`, per-request protocol versions, `resultType`, the `Mcp-Method`/`Mcp-Name` headers and cacheable `tools/list`, and enables the [Tasks extension](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2663). It comes later as a backward-compatible RFC 003 amendment; v1 jobs do not need it.

### Not allowed

- Raw provider keys in the agent sandbox or in agent jobs.
- Grading in the agent's sandbox or jobs.
- Reward writes the agent can reach.
- Passing the broker's token into jobs.
- Judges reading unescaped trajectories.
- Building on MCP sampling or payment protocols; MCP has no official cost or payment extension.

## Examples

`posttrain_env` with task `t0_arithmetic_format`, driven by any agent loop:

```python
from posttrain_env import PostTrainEnv

with PostTrainEnv(base_url="http://localhost:8000", message_timeout_s=3600).sync() as env:
    result = env.reset(task_id="t0_arithmetic_format")
    print(result.observation.metadata["prompt"])
    print(env.call_tool("budget"))
    # {'remaining_usd': 3.0, 'flavors': {'t4-small': {'usd_per_hour': 0.4}, ...}, 'jobs_left': 6, ...}

    job = env.call_tool("jobs_submit", script=train_py, flavor="t4-small", timeout_minutes=30)
    status = env.call_tool("jobs_wait", job_id=job["job_id"], timeout_s=300)
    # {'state': 'succeeded', 'cost_usd': 0.04,
    #  'artifact': {'uri': 'openenv-artifact://<episode>/<job>/outputs', 'files': [...]}}

    dev = env.call_tool("dev_evaluate", job_id=job["job_id"])
    env.call_tool("jobs_wait", job_id=dev["job_id"], timeout_s=300)  # result: {'score': ...}

    final = env.call_tool("submit", job_id=job["job_id"])  # ends the episode; reward = hidden-split score
```

The final observation carries `reward`, `metadata["service_spend"]` and `metadata["episode_end"]`.

## Prototype results

`envs/posttrain_env` runs this design on HF Jobs. Its unit tests replace HF Jobs with an in-memory fake, with no network access. Two live runs on 2026-10-16:

| Task | Agent job | Dev eval | Hidden-split reward | Rollout spend |
|---|---|---|---|---|
| `t0_artifact` | `cpu-basic`, 4 s | 1.0 | 1.0 | $0.000334 |
| `t0_arithmetic_format` | SFT of SmolLM2-135M-Instruct, `t4-small`, 333 s | 0.94 | 0.95 | $0.041 (`gpu_s` 333) |

Both runs ended with no orphaned jobs and `final: true`. The score jobs were not charged to the agent. A run with a `job.write`-only broker token was refused with HTTP 403 when the script upload reached the `jobs-artifacts` bucket (D1).

## Plan

1. **This PR**: the RFC and `envs/posttrain_env`, with no core changes. Two T0 tasks: `t0_artifact` checks the plumbing for a fraction of a cent; `t0_arithmetic_format` is a real SFT task with a hidden test split.
2. **Core types**: manifest, ledger, job and artifact models outside `server/`; the reserved-name prefix rule for `jobs_`; cleanup that surfaces cancel failures (`_cleanup_session_resources` swallows errors today) and the idle reaper on when services are declared.
3. **openenvd integration** (RFC 009): credential custody, token exchange, the grader zone, and MCP Tasks after the RFC 003 amendment.

## Future Work

Profiles added when an environment needs one:

| Profile | Examples | Adds |
|---|---|---|
| model call | inference, LLM/VLM judge | token metering |
| training session | Tinker-compatible APIs | session state, checkpoints as artifacts |
| interactive simulator | Blender, Isaac, browser, VM | lease, snapshot/fork, hardware capability |
| sub-sandbox | E2B, Daytona | nested isolation, per-service network policy |
| data access | search, paper/chem DBs | licensing, contamination audit, caching |
| licensed tool | FlexLM EDA, quantum queues | non-money leases (seats, daily quotas) |
| real-world actuation | cloud lab, robot, deploy | irreversible spend, orchestrator approval |
| human in the loop | expert review | hours-to-days latency, scarcity |
| artifact registry | HF Hub push | write-scoped credentials, provenance |
| peer agent | A2A | budget passed down to the peer (#779) |

Also: audit records keyed to `(episode, step, tool_call_id)` as OpenTelemetry GenAI/MCP spans, signed checkpoints (OpenSSF model signing), and exclusive-hardware leases for timing evaluators.

## Related

- RFC 009 openenvd (#1053, #1354, #1355): the base for credential custody and the grader zone.
- RFC 013 task sampling (#1256): reports `service_spend` as cost, not as a sampling weight.
- RFC 003: the MCP 2026-07-28 amendment. RFC 004: `score` feeds rubrics. RFC 005/006: LLM proxying is the future model-call profile. RFC 008: contract checks for declared services, plus #1357's adversarial panel. RFC 012: separate-verifier precedent.
- Issues: #898, #1026, #1232, #1299, #1357, #779.
