# LLM Capacity Reservation

An OpenEnv MCP environment for sequential capacity reservation using real, publicly
released BurstGPT conversation-service workload logs. Before seeing the next day's
requests, an agent commits 24 hourly token reservations for each of two deployments.
The environment then reveals demand and charges reservation and overflow costs.

The environment replays recorded demand and applies a fixed reservation and
shortfall cost model.

## Task and interaction

One episode contains 14 decision days. The observation provides the previous 14
realized days, the next relative log day, cost-model coefficients, and the most
recent outcome. Compare `cumulative_modeled_cost` only across completed episodes
on the same start. Summed reward treats a day that is never committed as zero
savings.

Each action is a dictionary with exactly `ChatGPT` and `GPT-4`,
each containing 24 finite nonnegative hourly token reservations in `[0, 1000000]`.
Reservations are measured in tokens per hour; the upper bound limits the action space.

Agent-facing MCP tools:

- `observe_demand_history()`: inspect revealed history and costs; does not advance time.
- `reserve_capacity(reservations)`: irreversibly commit all 48 hourly quantities,
  advance one day, and return the realized demand, cost and next observation.

The training orchestrator owns `reset(seed=...)`, `step`, and `state`; these are
not agent-facing MCP tools. Reading history pays zero reward. A successful
reservation emits its day's reward exactly once. Invalid actions do not change
costs or advance the decision day. Calls after the final decision cannot submit
additional reservations.

## Cost and reward

For model `m` and hour `h`, let `r` be reserved tokens and `d` realized total tokens:

```text
hourly_modeled_cost = unit_cost[m] * (r + 4 * max(d - r, 0)) / 1000
unit_cost = {ChatGPT: 0.002, GPT-4: 0.0375}
daily_cost = sum(hourly_modeled_cost)
zero_reservation_cost = sum(unit_cost[m] * 4 * d / 1000)
daily_reward = (zero_reservation_cost - daily_cost) / 1000
```

Minimize total modeled episode cost, or equivalently maximize summed reward.
The coefficients are fixed modeling assumptions for this capacity-reservation
task. Reward is continuous. The factor 1000 is a fixed numerical scale that
preserves the optimal policy.

The per-decision reward range is `[-0.948, 1.422]`. The data loader checks
that hourly demand is at most 500,000 tokens. Reserving zero earns exactly zero;
reserving useful capacity can earn positive savings, while excess reservation can
produce negative reward. The lower bound occurs when demand is zero and all
48 reservations are 1,000,000 tokens. The upper bound occurs when demand and
reservations are both 500,000 tokens in every hour. Reset, invalid actions and
observation-only calls earn zero and do not count as completed decisions.
For a fixed completed episode, the zero-reservation cost is policy-independent,
so maximizing total reward is equivalent to minimizing total modeled cost.

The 0.75 empirical quantile is a useful baseline because the cost has a 4x
shortfall multiplier. Drift and limited observation history create forecasting
uncertainty as workload levels change.

## Data, split and provenance

Data: Yuxin Wang et al., *BurstGPT: A Real-World Workload Dataset to Optimize LLM
Serving Systems*, KDD 2025, DOI `10.1145/3711896.3737413`, arXiv `2401.17644`.
Upstream: https://github.com/HPMLL/BurstGPT, release `v2.0`.

We use hourly sums of `Total tokens` from `Conversation log` records, for the
`ChatGPT` and `GPT-4` labels. There are 231 original relative days: 0–120 and
225–334. Days 121–224 are absent in the upstream release. Episodes never cross
this gap. Missing hours within a logged day have zero demand.
Day 77 has zero ChatGPT demand in every hour; test days 101–106 have
7–20 zero-demand hours per deployment. These may reflect upstream logging gaps
rather than absent traffic. They remain scored as zero demand, so costs partly
reflect forecasting these gaps; the trace does not establish real service outages.

All 231 days from the two selected conversation-service streams are retained.
Results describe these two streams and do not represent all LLM workloads.
No request text, completions, or user identifiers are distributed.

- Train decision-day segments: 14–89 and 239–304.
- Test decision-day segments: 90–120 and 305–334.
- The split is chronological within each recorded segment, not across segments:
  the second train segment comes after the first test segment.
- Start points require 14 preceding observed days and 14 consecutive target days.
- There are 116 train starts and 35 test starts. Adjacent episodes overlap; they
  must not be treated as independent experimental replicates.
- Test episodes may use earlier revealed history, including earlier days of the
  test time segment; this models online deployment with expanding observations.

Set `CAPACITY_SPLIT=train` (default) or `test` when starting the server. Seeded
reset chooses uniformly from eligible starts. The split is orchestrator-owned.
The public trace is shipped on the server side. Agents interact through the MCP
tools; the server filesystem contains the full trace and is reserved for the
environment operator. No live data API or provider account is required.

To independently reconstruct the aggregate, download `BurstGPT_1.csv`,
`BurstGPT_2.csv`, and `BurstGPT_3.csv` from upstream release v2.0 into a raw directory
(gzip versions are also accepted), then run:

```bash
python scripts/rebuild_aggregates.py --raw-dir /path/to/raw --output rebuilt-hourly.csv
sha256sum rebuilt-hourly.csv
```

Expected SHA-256:
`503ba7712c204aea5295427d6cef49fd0f662338fb62598453dfbe2b35877f77`.
The reconstruction script aggregates all three original release assets using the
selection and zero-filling rules above.

Data license: CC-BY-4.0, preserved in `server/data/LICENSE-BurstGPT.txt`.
Modification details and aggregate hashes are in `server/data/PROVENANCE.json`.
Environment code license: BSD-3-Clause. See `THIRD_PARTY_NOTICES.md`.

## Install and run

The environment supports Python 3.11 and later and requires OpenEnv 0.11.0 or later.
From `envs/llm_capacity_env`, install the frozen dependency set and run:

```bash
uv sync --python 3.11 --frozen --all-groups --all-extras
uv run --frozen python examples/evaluate.py baseline-results.json
uv run --frozen uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In another shell in the same directory:

```bash
uv run --frozen python examples/client_demo.py http://localhost:8000
uv run --frozen openenv validate . --level static --skip-build
uv run --frozen openenv validate --url http://localhost:8000
```

The integration tests live in the OpenEnv repository. From its root, install
its development dependencies and run:

```bash
uv sync --all-extras
PYTHONPATH=src:envs uv run pytest -q \
  tests/envs/test_llm_capacity_mcp.py \
  tests/envs/test_llm_capacity_replay.py \
  tests/envs/test_llm_capacity_trajectory.py
```

`uv.lock` supplies the same frozen runtime dependency set for the container
and the repository dependency check. Development extras are not installed in the
runtime image.

Docker:

```bash
docker build -f server/Dockerfile -t llm-capacity-env:0.1.0 .
docker run --rm -p 8000:8000 -e CAPACITY_SPLIT=test llm-capacity-env:0.1.0
```

`/health`, `/schema`, and `/openapi.json` are provided by OpenEnv. The debug UI is
available at `/web/` when `ENABLE_WEB_INTERFACE=true` (enabled in the Docker image).
MCP and WebSocket clients use OpenEnv's standard endpoints. The server supports
independent concurrent sessions through an environment factory.

## Baselines

`examples/evaluate.py` evaluates zero reservation, previous-day demand, a fixed
initial-history 0.75 quantile, and an adaptive rolling 0.75 quantile on all eligible
starts. It also reports a hindsight diagnostic that reserves the realized demand;
this diagnostic reads future data and is not an agent policy. Compare completed
episodes on the same starts using mean modeled cost (lower is better) or summed
savings reward (higher is better). Overlapping episodes are not independent samples.

## Optional trajectory recording

Recording is disabled by default. To write one JSONL file per server episode,
set `CAPACITY_TRACE_DIR=/path/to/trajectories` before starting the server, or pass
`trace_dir` when constructing `CapacityEnvironment`. The operator is responsible
for retention and disk limits when recording is enabled.

To generate and independently replay complete heuristic-policy examples:

```bash
uv run --frozen python examples/record_trace.py outputs/example-traces
```

Generated traces and validation logs are not included in the environment package.

