# RFC: Environment Services

**Status**: In Review
**Created**: 2026-10-16
**Authors**: @rycerzes
**RFC ID**: 014

Discussion: [#1382](https://github.com/openenv/OpenEnv/issues/1382). Prototype: [experimental `posttrain_env` PR #1500](https://github.com/openenv/OpenEnv/pull/1500).

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
- Changing `/mcp` or the core types in this RFC.

## Design

### Architecture Overview

```
agent ──MCP tools──▶ environment server = broker (holds the HF token)
 (no credentials)       ledger · job handles · artifacts
                              │ run_uv_job, no secrets
                              ▼
                        HF Jobs: agent jobs (no network) · dev_evaluate · score
                              │
               score job ◀────┘ artifact mounted read-only; its score is the reward
```

In v1 the environment server is the broker, as in `posttrain_env`. Moving the credential to openenvd (RFC 009) is later work.

### Core Abstractions

**Service manifest.** An environment declares its services by profile:

```yaml
services:
  - {name: jobs, profile: batch_job, surface: agent, flavors: [a10g-small], max_timeout: 60m, meter: [usd, gpu_s]}
  - {name: kernel_eval, profile: evaluator, surface: {agent: dev_evaluate, grader: score},
     hardware: {class: h100, exclusive: true}}
budget: {usd: 3.0, scope: rollout}
```

`posttrain_env` keeps this in its task definitions (`server/tasks.py`); a possible core manifest is deferred to step 2, after the provider contract and second backend have been validated.

**Service profiles (v1).**

| Profile | Examples | Adds to the contract |
|---|---|---|
| batch job | HF Jobs; later Modal, Slurm, render farms | job handle, worst-case reservation, cancel, orphans |
| evaluator | held-out model eval, kernel timing | `dev_evaluate` vs `score`; exclusive hardware and caching for timing (not in the prototype) |

**Budget ledger.** Money is integer micro-USD. Before a job starts the broker reserves its worst case, the flavor's per-minute price times the job's timeout, and refuses the job with a typed `budget_exhausted` error if it does not fit. When the job ends the reservation is replaced by the real cost, the per-minute price times the minutes the job ran, rounded up to whole minutes as HF Jobs bills. Budgets are per rollout and set by the orchestrator, so every rollout in a GRPO group faces the same limit. The agent sees its remaining budget as data, never as reward. The ledger is the broker's estimate; the provider's bill is authoritative, and a provider-side cap is the backstop (D1).

Every observation carries a stable spend entry. It is for accounting, and for budgets, concurrency and rollout allocation, never a sampling weight: dividing a task's sampling weight by its cost would starve expensive, hard tasks. RFC 013's sampler never reads the environment; the trainer reads this entry from the final observation and can report one value per rollout as `costs` to `update(task, rewards, costs)`. RFC 013 takes costs for every group or for none, so an episode that ends unscored still reports its spend.

```python
obs.metadata["service_spend"] = {
    "version": 1,
    "settled": {
        "usd": 0.62,
        "gpu_s": 410.0,
        "jobs": 2,
        "calls": 1,
    },  # cumulative for the episode
    "reserved": {"usd": 0.0},  # open reservations; empty once every job has settled
    "final": True,  # false while a job is live or was orphaned
}
```

v1 reports `usd`, `gpu_s`, `jobs` and `calls`. `jobs` counts agent jobs and `calls` counts dev evaluations. Grader evaluations are not charged to the agent.

**Jobs.** Tools return a handle. States are `queued → running → succeeded | failed | cancelled`, plus `cancelling` while a cancel is pending and `orphaned` when the provider does not confirm one in time. An orphaned job is settled at its full reservation, keeps `final: false` and blocks the next `reset` with an error, because it may still be running and billing. `reset`, `close`, `submit`, the step limit and the time limit cancel the episode's live jobs. Submissions take an idempotency key; a retry with the same key returns the same job and makes no second reservation.

**Network.** Agent jobs run without network access. A job that can reach the internet can download a model already trained for the task, or the evaluation's own test data; [PostTrainBench](https://arxiv.org/abs/2603.08640) reports both. Inputs come as read-only mounts instead: the task's base model at a pinned revision, and the public, ungated Hub datasets the agent names, at their current revision, minus a per-task denylist. Packages are installed into a fresh venv from wheels only, which runs none of their code, before the agent's code runs. HF Jobs has no egress control, and its containers can't create network namespaces or firewall rules (checked on `cpu-basic` and `a10g-small`). So `posttrain_env` runs the agent's script inside a launcher that installs a seccomp filter refusing every socket except `AF_UNIX`, and io_uring and cross-process memory access, then closes inherited file descriptors and drops to a uid no other process in the job uses; systemd's `RestrictAddressFamilies=` works the same way. This limits a job to declared, recorded inputs, but doesn't make them clean: a public dataset or a PyPI wheel can still hold weights or test data. Tasks whose test data is public also need a contamination check after the run or an allowlist of permitted inputs.

**Evaluator.** Two endpoints. `dev_evaluate` runs on the public dev split, is rate-limited and charged to the agent, and returns at most five metrics and short feedback. `score` runs on the hidden split as a grader-only job and is the reward; the hidden split's seed reaches it as a secret. Both run the environment's evaluator script as their own job, with the artifact under test mounted read-only. The current prototype reads the evaluator's `POSTTRAIN_SCORE {"score": ...}` line from the logs; a checkpoint it can't use scores 0 as `invalid`. A planned follow-up, not yet implemented, will have the evaluator write `/data/result.json` and the broker read it separately from diagnostic logs. The evaluator takes only data from the artifact, never code or anything that shapes the evaluator's own input: `t0_arithmetic_format` loads safetensors weights into the base model's config, with the base tokenizer, chat template and generation settings from the base model mount, since a chat template could compute the answer into the prompt. Its dependencies are pinned too, so scores don't change with a new release. A score job that fails, times out, is cancelled or prints no valid score ends the episode unscored (`reward` is `None`, `episode_end.reason` is `score_failed`): the fault is the grader's, and RFC 013's `update` takes `None` for a rollout the environment did not score, which samplers leave out rather than count as zero. This `None`-reward integration depends on acceptance of RFC 013 (#1256) and must follow its final unscored-rollout semantics; it is not an independently accepted core contract.

**Artifacts.** An artifact is the output folder of a succeeded job, exposed as `openenv-artifact://<episode>/<job>/outputs` with each file's path, size and content hash. Graders mount the folder read-only; nothing is copied through the agent.

### Minimal batch-job provider contract (proposed)

HF Jobs is the first implementation, not the definition of the interface. A
provider must offer these operations to the broker, without exposing its
credentials to agent code:

- **Submit**: accept a workload, resource choice, deadline and declared inputs;
  support secrets for grader jobs only, for the hidden split's seed; agent jobs
  receive no secrets. Return a stable job handle and support reconciliation of
  ambiguous submissions without duplicate work or spend.
- **Inspect**: report state and execution duration/usage for settlement, including
  terminal failures and pending cancellation.
- **Cancel**: request termination and allow confirmation of a terminal state;
  an unconfirmed cancellation remains an orphan that may still bill.
- **Logs**: read bounded diagnostics for the episode's own jobs, separately from
  the authoritative evaluator result.
- **Price**: expose units, billing granularity and a conservative cost bound that
  can be reserved before execution, then settled from observed usage.
- **Artifacts**: enumerate outputs and their identity/size, and provide read-only
  access to an isolated grader without passing credentials through the agent.

Per-minute billing, `run_uv_job`, the `jobs-artifacts` bucket, Resource Group
caps and the seccomp launcher are HF implementation choices. Other providers
must preserve reservations, credential/grader separation and no-network agent
execution using their own facilities.

**Prerequisite before step 2:** sketch and validate this contract against a
second backend before proposing core types or reserved names. A local Docker
backend is proposed for no-spend end-to-end tests: container handles and inspect,
stop, logs, isolated output volumes, and an explicit local resource-price policy
would exercise these operations. Modal is another proposed check against
HF-shaped assumptions, especially network isolation. Neither backend is
implemented or adopted by this RFC. Native provider no-network support remains
a prerequisite for step 2; track HF support in
[huggingface_hub#5261](https://github.com/huggingface/huggingface_hub/issues/5261).

### Key Design Decisions

#### D1. The broker is the only credential holder

- **Chosen approach**: The broker starts jobs with `HfApi.run_uv_job`, with explicit flavor, namespace, timeout and labels. Agent jobs get an empty `secrets` map. It never runs `hf jobs uv run`, whose script header can [request secrets that resolve from the caller's environment](https://github.com/huggingface/huggingface_hub/blob/841c22b46cdc25b55a2bfb1c2ad97f418162c282/src/huggingface_hub/cli/jobs.py#L311-L323), `HF_TOKEN` included, and set the namespace; the Python API [ignores that header](https://github.com/huggingface/huggingface_hub/blob/841c22b46cdc25b55a2bfb1c2ad97f418162c282/src/huggingface_hub/hf_api.py#L12901-L12902).
- **Rationale**: A fine-grained token with only `job.write` cannot touch repos, but a job holding it can start further jobs that skip the broker's budget checks (verified on HF Jobs). So the token never enters a job.
- **Broker token**: "Start and manage Jobs" plus write access to the namespace's `jobs-artifacts` bucket. A `job.write`-only token is refused (HTTP 403) when the broker uploads the job script to that bucket.
- **Backstop**: A token has no spend limit of its own; spend limits are set on a [Resource Group](https://huggingface.co/docs/hub/en/enterprise-resource-groups) (Enterprise). Billing broker jobs to a group with a Jobs spend limit caps the total; over the limit, HF refuses new jobs and cancels running ones.

#### D2. Artifacts move through the HF Jobs artifacts bucket

- **Chosen approach**: Follow `run_uv_job`'s own convention. It uploads the script to the namespace's private `jobs-artifacts` bucket and mounts that job's folder read-write at `/data`. Jobs write outputs to `/data/outputs/`. Evaluator jobs mount `<folder>/outputs` read-only at `/model`.
- **Rationale**: No token inside the job, no extra repository, and HF's existing behaviour. Content hashes come from the bucket listing (Xet hashes).
- **Trade-offs**: Every agent job and evaluation creates a bucket folder; cleanup follows the bucket's retention, and what a job writes there is neither capped nor charged to the budget.

#### D3. Tool names use an underscore prefix

- **Chosen approach**: `jobs_submit`, `jobs_status`, `jobs_wait`, `jobs_logs`, `jobs_cancel`, `dev_evaluate`, `submit` and `budget`.
- **Rationale**: OpenAI-compatible function names allow only `[a-zA-Z0-9_-]`, and most trainers convert MCP tools to that form, so `jobs.submit` breaks them. These are prototype names only. Any reserved-name prefix rule in core is deferred until the provider contract and a second backend have been validated before step 2.

#### D4. Long calls block with a declared timeout

- **Chosen approach**: `submit` waits for the score job and returns the reward in the same step; `jobs_wait` waits up to a capped number of seconds. The environment raises its tool timeout for these calls and reports `score_timeout_s` at `reset` so clients can set their message deadline (the same need as #1026).
- **Rationale**: The reward stays inside the environment (RFC 004) and the episode ends with it, without a side channel.

### Decisions from the discussion

1. **Long-running services and GRPO.** A service runs live at train time only if it finishes within the trainer's staleness window. Asynchronous GRPO trainers drop samples generated more than a few weight updates ago, so an episode has minutes, not hours. Slower services declare a stand-in and the real service runs only at eval. Post-training tasks are multi-hour episodes, so their near-term use is evaluation.
2. **HF Spaces.** No services on Spaces for now. A Space gives no isolation guarantee, so it must not hold a broker credential.

### Not allowed

- Raw provider keys in the agent sandbox or in agent jobs.
- Network access from agent jobs.
- Grading in the agent's sandbox or jobs.
- Reward writes the agent can reach.
- Passing the broker's token into jobs.
- Building on MCP sampling or payment protocols; MCP has no official cost or payment extension.

## Examples

`posttrain_env` with task `t0_arithmetic_format`, driven by any agent loop:

```python
from openenv.core.env_server.mcp_types import CallToolAction
from posttrain_env import PostTrainEnv

with PostTrainEnv(
    base_url="http://localhost:8000", message_timeout_s=3600
).sync() as env:
    result = env.reset(task_id="t0_arithmetic_format")
    print(result.observation.metadata["prompt"])
    result = env.step(CallToolAction(tool_name="budget", arguments={}))
    print(result.observation.result["data"])

    result = env.step(
        CallToolAction(
            tool_name="jobs_submit",
            arguments={
                "script": train_py,
                "flavor": "t4-small",
                "timeout_minutes": 30,
            },
        )
    )
    assert result.observation.error is None
    job = result.observation.result["data"]
    assert "error" not in job, job
    status = "queued"
    while status in {"queued", "running"}:
        result = env.step(
            CallToolAction(
                tool_name="jobs_wait",
                arguments={
                    "job_id": job["job_id"],
                    "timeout_s": 300,
                },
            )
        )
        assert not result.done, result.observation.metadata
        status = result.observation.result["data"]["state"]
    assert status == "succeeded", result.observation.result

    dev = env.step(
        CallToolAction(
            tool_name="dev_evaluate",
            arguments={
                "job_id": job["job_id"],
            },
        )
    )
    dev_job = dev.observation.result["data"]
    assert "error" not in dev_job, dev_job
    status = "queued"
    while status in {"queued", "running"}:
        result = env.step(
            CallToolAction(
                tool_name="jobs_wait",
                arguments={
                    "job_id": dev_job["job_id"],
                    "timeout_s": 300,
                },
            )
        )
        assert not result.done, result.observation.metadata
        status = result.observation.result["data"]["state"]
    assert status == "succeeded", result.observation.result
    print(result.observation.result["data"]["result"])

    final = env.step(
        CallToolAction(
            tool_name="submit",
            arguments={
                "job_id": job["job_id"],
            },
        )
    )
    print(final.done, final.reward, final.observation.metadata["service_spend"])
```

The final step result carries `reward`, and its observation carries
`metadata["service_spend"]` and `metadata["episode_end"]`. Direct `call_tool`
uses `/mcp`, bypasses episode step/time limits and does not return this final
step observation; core enforcement is tracked in [#1507](https://github.com/openenv/OpenEnv/issues/1507).
`service_spend["final"]` means all jobs have settled, not that the episode ended.
`submit` blocks for up to reset metadata's `score_timeout_s` (2700 s for this
task), so allow additional cancellation/transport time in client deadlines.

## Prototype results

`envs/posttrain_env` runs this design on HF Jobs. Its unit tests replace HF Jobs with an in-memory fake, with no network access. Live runs:

| Date | Task | Driver | Agent jobs' network | Dev eval | Reward | Rollout spend |
|---|---|---|---|---|---|---|
| 2026-10-16 | `t0_artifact` | script | open | 1.0 | 1.0 | $0.000334 |
| 2026-10-16 | `t0_arithmetic_format` | script: SFT on `t4-small`, 333 s | open | 0.94 | 0.95 | $0.041 (`gpu_s` 333) |
| 2026-10-26 | `t0_arithmetic_format` | Qwen3.8 through HF Inference Providers, 25 and 29 turns | open | 1.0 | 1.0, 1.0 | $0.21, $0.34 |
| 2026-10-27 | `t0_artifact` | script | none | | 1.0 | $0.000167 |
| 2026-10-27 | `t0_arithmetic_format` | script: SFT on `t4-small`, 333 s | none | 0.94 | 0.93 | $0.041 (`gpu_s` 333) |
| 2026-10-27 | `t0_arithmetic_format` | Qwen3.8 through HF Inference Providers, 23 turns | none | 1.0 | 1.0 | $0.12 |

Spend is the agent's HF Jobs spend; LLM tokens are not included. Every run ended with no orphaned jobs and `final: true`, and score jobs were not charged to the agent. A run with a `job.write`-only broker token was refused with HTTP 403 when the script upload reached the `jobs-artifacts` bucket (D1).

- **No network.** In the 27 Oct runs the agent's script could not open a socket and ran as an unprivileged uid. CUDA training, DataLoader workers and writing `/data/outputs/` worked as before. Qwen3.8 loaded the base model from `/base_model` from its first script on; its three failed jobs were its own bugs and one CUDA allocation error, none from the network policy. Loading the base model from its mount instead of the Hub also cut the score job from about 20 minutes to 140 s.
- **T0 measures plumbing, not training skill.** Qwen3.8 trained on all 10,000 `(a, b)` pairs, so a hidden split of an input space that small hides nothing. Real tasks need inputs that cannot be enumerated, or test data the agent cannot reach.
- **A failed score job is not a zero.** One Qwen3.8 run scored 1.0 on dev, and its score job printed 1.0 and then hit its timeout; the episode got reward 0. Such episodes now end unscored (`score_failed`), and the evaluator installs CPU-only torch.

## Plan

1. **Separate design and prototype PRs**: this PR contains only RFC 014 and its index entry; [#1500](https://github.com/openenv/OpenEnv/pull/1500) contains experimental `posttrain_env`, with no core changes and no network in agent jobs. Two T0 tasks that check the plumbing: `t0_artifact` for a fraction of a cent, and `t0_arithmetic_format`, a small SFT task with a hidden test split.
2. **Proposed core types, after prerequisites**: first validate the provider contract with a second backend and obtain native no-network support. Only then propose manifest, ledger, job and artifact models outside `server/`, any reserved-name prefix rule, cleanup that surfaces cancel failures and an idle reaper for declared services, in a separate PR. None of these core changes or names is adopted here.
3. **openenvd integration** (RFC 009): credential custody and the grader zone.

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

Also: shared brokers outside the environment with per-rollout tokens ([RFC 8693](https://www.rfc-editor.org/rfc/rfc8693) token exchange), the MCP 2026-07-28 transport through an RFC 003 amendment, audit records keyed to `(episode, step, tool_call_id)` as OpenTelemetry GenAI/MCP spans, signed checkpoints (OpenSSF model signing), and exclusive-hardware leases for timing evaluators.

## Related

- RFC 009 openenvd (#1053, #1354, #1355): the base for credential custody and the grader zone.
- RFC 013 task sampling (#1256): `service_spend` is one source of the trainer's `costs`, never a sampling weight; the proposed treatment of an unscored episode's `None` reward depends on RFC 013 acceptance and its final semantics.
- RFC 004: `score` feeds rubrics. RFC 005/006: LLM proxying is the future model-call profile. RFC 008: contract checks for declared services, plus #1357's adversarial panel. RFC 012: separate-verifier precedent.
- Issues: #898, #1026, #1232, #1299, #1357, #779.
