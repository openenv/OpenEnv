<!-- openenv-source: posttrain_env -->
# Post-training Environment (experimental)

`posttrain_env` is an MCP environment where the agent post-trains a model on
[HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). It is the
experimental, eval-first prototype for [RFC 014: Environment Services
(PR #1508)](https://github.com/openenv/OpenEnv/pull/1508); discussion in
[#1382](https://github.com/openenv/OpenEnv/issues/1382).

The agent never holds a credential. It submits training scripts through MCP
tools, and the server's broker starts them on HF Jobs with its own token. Each
job's worst-case cost is reserved against a per-rollout budget before it starts
and settled at the real cost when it ends. `dev_evaluate` scores a checkpoint
on a public dev split. `submit` scores it on a hidden split in a grader-only
job, and that score is the reward.

## Tools

| Tool | What it does |
|---|---|
| `budget()` | Remaining budget, allowed hardware with its price, jobs and dev evaluations left |
| `jobs_submit(script, flavor, timeout_minutes, dependencies=None, datasets=None, idempotency_key=None)` | Run a Python script as an HF Job, like `hf jobs uv run`, without network access. Reserves `price/min × timeout` first |
| `jobs_status(job_id)` | State, cost once settled, artifact files of a succeeded job, result of a finished evaluation |
| `jobs_wait(job_id, timeout_s=60)` | Wait up to 300 s for a job to finish |
| `jobs_logs(job_id, tail=100)` | Last lines of one of your own jobs' logs |
| `jobs_cancel(job_id)` | Cancel a job; it settles for the minutes it ran |
| `dev_evaluate(job_id)` | Score a succeeded job's artifact on the dev split, as a new job charged to your budget |
| `submit(job_id)` | Final answer: cancels your other jobs, scores the artifact on the hidden split, ends the episode |

Errors come back as data, with a type the agent can act on: `budget_exhausted`,
`flavor_not_allowed`, `timeout_not_allowed`, `invalid_script`,
`invalid_dependency`, `dataset_not_allowed`, `limit_reached`, `not_ready`,
`unknown_job`, `not_allowed`, `episode_over` or `hf_error`.

## How a job runs

Agent scripts run with `HfApi.run_uv_job`, following its usual behaviour: the
script is uploaded to the namespace's private `jobs-artifacts` bucket, and the
job's folder there is mounted read-write at `/data`. Anything the job writes
under `/data/outputs/` is its artifact:

```
openenv-artifact://<episode>/<job_id>/outputs
```

Jobs are started with an explicit flavor, namespace, timeout and labels, and
with no secrets. The broker never passes its own token into a job: a job holding
a "Start and manage Jobs" token could start further jobs that skip the budget.
It also never shells out to `hf jobs uv run`, whose script header can pull the
caller's `HF_TOKEN` into the job.

### No network in agent jobs

A job that can reach the internet can download a model already trained for the
task, or the evaluation's own test data. So agent jobs get their inputs as
read-only mounts, which the broker checks and the episode records, and run
without network access:

- The task's base model is mounted at `/base_model`, at a pinned revision.
- Each dataset in `datasets` is mounted at `/datasets/<repo_id>`, at its current
  revision. It must be public and ungated, and not on the task's denylist. The
  broker looks it up without its token, so the token can't open private repos.
- Packages from `dependencies` and the script's PEP 723 header are installed
  into a fresh venv before the script runs, from PyPI wheels only. Installing a
  wheel runs none of its code, so no package code runs while the job still has
  network access. URLs and git dependencies are refused. Other settings in the
  script's header are ignored.

Mounts and wheels narrow what a job can fetch, but a public dataset or a PyPI
wheel can still hold weights or test data. Tasks whose test data is public also
need a contamination check after the run.

HF Jobs has no egress control, and its containers can't create network
namespaces or firewall rules. So the broker sends
[`server/job_launcher.py`](https://github.com/openenv/OpenEnv/blob/main/envs/posttrain_env/server/job_launcher.py) as the job's script, with the
agent's script and packages embedded. Once it has installed the packages, the
launcher installs a seccomp filter that refuses every socket except `AF_UNIX`,
and io_uring and cross-process memory access; systemd's
`RestrictAddressFamilies=` works the same way. It then closes inherited file
descriptors, switches to a uid no other process in the job uses, and runs the
agent's script with `HF_HUB_OFFLINE=1`. The filter can't be removed and covers every child
process. If any step fails, the script doesn't run. Training on a single GPU,
DataLoader workers and writing `/data/outputs/` work under the filter;
`torch.distributed` over localhost TCP does not.

### Evaluations

Evaluations run the task's evaluator script as their own job with that folder
mounted read-only at `/model`, and the task's base model at `/base_model`. The
evaluator prints one `POSTTRAIN_SCORE {...}` line. A missing line, or a score
outside [0, 1], scores 0 as `invalid`. Evaluators take only weights from the
artifact, loaded as safetensors. The tokenizer, chat template, config and
generation settings come from the base model mount, because a chat template is
a program that sees the question and could write the answer into the prompt. Each evaluator pins its
dependencies in its PEP 723 header, with an `exclude-newer` date for the rest.

## Budget and spend

Money is tracked in micro-USD from the prices `list_jobs_hardware` reports.
A job settles at its price per minute times its running minutes, rounded up to
whole minutes. Grader evaluations are not charged to the agent.

Every observation carries the episode's spend, for accounting and for budgets,
concurrency and rollout allocation. It is not a sampling weight (RFC 013), and
an episode that ends unscored still reports it:

```python
obs.metadata["service_spend"]
# {"version": 1,
#  "settled": {"usd": 0.000334, "gpu_s": 0.0, "jobs": 1, "calls": 1},
#  "reserved": {"usd": 0.0},
#  "final": True}
```

`jobs` counts agent jobs and `calls` counts dev evaluations. `final` is false
while a job is still running or a cancellation was never confirmed.

## Episode end

Through `step(CallToolAction(...))`, an episode ends when the agent calls
`submit`, after `max_steps` tool calls (default 100) or after the task's time limit.
The step and time limits give reward 0.

Direct `/mcp` calls, including `env.call_tool`, bypass the step and time limits
and do not return the final step observation with reward and `service_spend`.

If the score job itself fails, times out or is cancelled, the episode ends
unscored: `reward` is `None` and `episode_end.reason` is `score_failed`, since
the fault is the grader's. Trainers should drop such a rollout, not count it as
zero, subject to acceptance of RFC 013 (#1256) and its final unscored-rollout
semantics. Ending the episode cancels every live job. `reset` cancels the previous
episode's jobs first and raises if HF Jobs does not confirm a cancellation,
since that job may still be running and billing.

## Using it with a trainer

This environment is eval-first today. Post-training episodes can take hours,
beyond the staleness window of asynchronous GRPO trainers. Use it for evaluation;
training integration needs a service that fits the trainer's staleness window or
a stand-in for the slow service.

`submit` blocks for up to `score_timeout_s`, reported in reset metadata (2700 s
for `t0_arithmetic_format`). Size the client's tool/message timeout above that
value, with room for cancellation and transport overhead; the examples use
`message_timeout_s=3600`.

Read `final.reward` and `final.observation.metadata["service_spend"]` from the
final `step` result. For RFC 013 integration, forward one settled amount per
rollout, for example `service_spend["settled"]["usd"]`, as `costs` in the trainer's
chosen unit. Spend is accounting, never a sampling weight. `final: true` means
all jobs have settled, even mid-episode; it does not mean the episode is done.
Use the step result's `done` for that. An orphan keeps spend `final: false` and
may still bill; retain that accounting even when reward is `None`.

## Tasks

| Task | What the agent does | Hardware | Budget |
|---|---|---|---|
| `t0_artifact` | Write the episode's nonce to `/data/outputs/answer.txt`. Checks the plumbing | `cpu-basic` | $0.05 |
| `t0_arithmetic_format` | Fine-tune `HuggingFaceTB/SmolLM2-135M-Instruct` to answer two-digit addition as `#### {a+b}`. Reward is exact-match accuracy on a hidden test split | `cpu-upgrade`, `t4-small`, `a10g-small` | $3.00 |

## Running the server

```bash
export HF_TOKEN=...   # the broker's token, see below
cd envs/posttrain_env
uv run --project . server
```

The local entry point binds `127.0.0.1:8000`. The Docker command explicitly
binds `0.0.0.0:8000` so a private host can reach the published port. This
experimental env is excluded from automatic Spaces releases via
`SKIP_HF_DEPLOYMENT`; do not publish it to Spaces or add it to HF collections.

Configuration, all read by the server only:

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Broker token. A fine-grained token with "Start and manage Jobs" plus write access to `<namespace>/jobs-artifacts`. A Jobs-only token is refused when the script is uploaded |
| `POSTTRAIN_NAMESPACE` | Namespace jobs run and bill in. Defaults to the token's owner |
| `POSTTRAIN_RESOURCE_GROUP_ID` | Resource Group to bill. Its Jobs spend limit caps total spend even if the ledger is wrong |
| `POSTTRAIN_TEST_SEED` | Seed of the hidden test split. Set it for any run whose scores you compare: if unset it is random per process (the server logs a warning), so scores are only comparable within one server process. Keep it secret from the agent |

Run the server where its token is not shared with the agent, for example a
container you control. Do not deploy it as a public Space: a Space gives no
isolation guarantee for the credential it holds.

## Client

```python
from openenv.core.env_server.mcp_types import CallToolAction
from posttrain_env import PostTrainEnv

with PostTrainEnv(
    base_url="http://localhost:8000", message_timeout_s=3600
).sync() as env:
    result = env.reset(task_id="t0_artifact")
    print(result.observation.metadata["prompt"])
    result = env.step(
        CallToolAction(
            tool_name="jobs_submit",
            arguments={
                "script": open("train.py").read(),
                "flavor": "cpu-basic",
                "timeout_minutes": 5,
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
    final = env.step(
        CallToolAction(
            tool_name="submit",
            arguments={
                "job_id": job["job_id"],
            },
        )
    )
    print(final.done, final.reward)
    print(final.observation.metadata["service_spend"])
```

[`examples/posttrain_env_agent.py`](https://github.com/openenv/OpenEnv/blob/main/examples/posttrain_env_agent.py)
plays a whole episode with an LLM through any OpenAI-compatible endpoint,
Hugging Face Inference Providers by default. The agent's API key is its own;
it never needs the broker's token.

## Known gaps

- The server has no authentication of its own: anyone who can reach it can
  start episodes and spend its budget. Keep it on localhost or a private network.
- Job folders stay in the `jobs-artifacts` bucket, and the size of what a job
  writes there is not capped or charged to the budget.
- `max_steps` and the time limit are checked in `step`; a client calling the
  MCP endpoint directly bypasses them, though not the budget. Core enforcement
  is tracked in [#1507](https://github.com/openenv/OpenEnv/issues/1507).
- `torch.distributed` over localhost TCP is blocked in agent jobs, so tasks use
  single-GPU flavors.

## Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_posttrain_environment.py -v
```

The tests replace HF Jobs with an in-memory fake, so they make no network calls
and spend nothing.
