# Post-training Environment

`posttrain_env` is an MCP environment where the agent post-trains a model on
[HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). It is the
prototype for [RFC 014: Environment Services](../../rfcs/014-environment-services.md).

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
| `jobs_submit(script, flavor, timeout_minutes, dependencies=None, idempotency_key=None)` | Run a Python script as an HF Job, like `hf jobs uv run`. Reserves `price/min × timeout` first |
| `jobs_status(job_id)` | State, cost once settled, artifact files of a succeeded job, result of a finished evaluation |
| `jobs_wait(job_id, timeout_s=60)` | Wait up to 300 s for a job to finish |
| `jobs_logs(job_id, tail=100)` | Last lines of one of your own jobs' logs |
| `jobs_cancel(job_id)` | Cancel a job; it settles for the minutes it ran |
| `dev_evaluate(job_id)` | Score a succeeded job's artifact on the dev split, as a new job charged to your budget |
| `submit(job_id)` | Final answer: cancels your other jobs, scores the artifact on the hidden split, ends the episode |

Errors come back as data, with a type the agent can act on: `budget_exhausted`,
`flavor_not_allowed`, `timeout_not_allowed`, `limit_reached`, `not_ready`,
`unknown_job`, `not_allowed`, `episode_over` or `hf_error`.

## How a job runs

Agent scripts run with `HfApi.run_uv_job`, following its usual behaviour: the
script is uploaded to the namespace's private `jobs-artifacts` bucket, and the
job's folder there is mounted read-write at `/data`. Anything the job writes
under `/data/outputs/` is its artifact:

```
openenv-artifact://<episode>/<job_id>/outputs
```

Evaluations run the task's evaluator script as their own job with that folder
mounted read-only at `/model`. The evaluator prints one `POSTTRAIN_SCORE {...}`
line. A missing line, or a score outside [0, 1], scores 0 as `invalid`.

Jobs are started with an explicit flavor, namespace, timeout and labels, and
with no secrets. The broker never passes its own token into a job: a job holding
a "Start and manage Jobs" token could start further jobs that skip the budget.
It also never shells out to `hf jobs uv run`, whose script header can pull the
caller's `HF_TOKEN` into the job.

## Budget and spend

Money is tracked in micro-USD from the prices `list_jobs_hardware` reports.
A job settles at its price per minute times its running minutes, rounded up to
whole minutes. Grader evaluations are not charged to the agent.

Every observation carries the episode's spend, so a trainer can pass it to a
cost-aware sampler (RFC 013) as `costs`:

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

An episode ends when the agent calls `submit`, after `max_steps` tool calls
(default 100) or after the task's time limit. The last two give reward 0.
Ending the episode cancels every live job. `reset` cancels the previous
episode's jobs first and raises if HF Jobs does not confirm a cancellation,
since that job may still be running and billing.

`submit` waits for the score job, so it can take several minutes. `reset`
reports `score_timeout_s`; set the client's `message_timeout_s` above it.

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

Configuration, all read by the server only:

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Broker token. A fine-grained token with "Start and manage Jobs" plus write access to `<namespace>/jobs-artifacts`. A Jobs-only token is refused when the script is uploaded |
| `POSTTRAIN_NAMESPACE` | Namespace jobs run and bill in. Defaults to the token's owner |
| `POSTTRAIN_RESOURCE_GROUP_ID` | Resource Group to bill. Its Jobs spend limit caps total spend even if the ledger is wrong |
| `POSTTRAIN_TEST_SEED` | Seed of the hidden test split. Random per process if unset, so scores are only comparable within one server process |

Run the server where its token is not shared with the agent, for example a
container you control. Do not deploy it as a public Space: a Space gives no
isolation guarantee for the credential it holds.

## Client

```python
from posttrain_env import PostTrainEnv

with PostTrainEnv(base_url="http://localhost:8000", message_timeout_s=3600).sync() as env:
    result = env.reset(task_id="t0_artifact")
    print(result.observation.metadata["prompt"])
    job = env.call_tool(
        "jobs_submit",
        script=open("train.py").read(),
        flavor="cpu-basic",
        timeout_minutes=5,
    )
    env.call_tool("jobs_wait", job_id=job["job_id"], timeout_s=300)
    print(env.call_tool("submit", job_id=job["job_id"]))
```

## Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_posttrain_environment.py -v
```

The tests replace HF Jobs with an in-memory fake, so they make no network calls
and spend nothing.
