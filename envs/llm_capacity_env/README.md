---
title: LLM Capacity Reservation
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
base_path: /web
license: bsd-3-clause
tags:
  - openenv
  - reinforcement-learning
  - resource-allocation
  - llm-serving
---

# LLM Capacity Reservation

An OpenEnv MCP environment for sequential capacity reservation using real, publicly
released BurstGPT conversation-service workload logs. Before seeing the next day's
requests, an agent commits 24 hourly token reservations for each of two deployments.
The environment then reveals demand and charges reservation and overflow costs.

This is a **trace-replay environment with modeled economics**. It is not a live
Azure service, hardware simulator, or measurement of current cloud pricing.

## Task and interaction

One episode contains 14 decision days. The observation provides the previous 14
realized days, the next relative log day, published cost coefficients, and the most
recent outcome. Only completed 14-decision episodes are comparable; an unfinished prefix is not a successful zero-cost episode. The evaluation harness must require completion.

Each action is a dictionary with exactly `ChatGPT` and `GPT-4`,
each containing 24 finite nonnegative hourly token reservations in `[0, 1000000]`.
The upper bound is a numerical action guard, not a claim about GPU throughput.

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
daily_reward = -sum(hourly_modeled_cost) / 1000
```

Minimize total modeled episode cost, or equivalently maximize summed reward.
The coefficients are explicit accounting assumptions retained from our existing
capacity-reservation case. They are not real customer invoices, contemporary
provider prices, or service-level guarantees. There is **no binary acceptance
threshold**. The factor 1000 in reward is a fixed numerical scale; it does not
change the optimal policy.

The declared per-step range is `[-2, 0]`. The data loader checks that hourly demand
is at most 500,000 tokens. Cost is convex in each reservation; at either legal
endpoint it is at most `unit_cost * 2,000,000 / 1000`. Summing 24 hours and the two
coefficients gives at most 1,896 modeled cost units, hence reward at least -1.896.
Zero reward is also used for reset and observation calls; it is not a claim that
zero operating cost is attainable on a nonempty demand day.

The 0.75 empirical quantile is a useful baseline because the cost has a 4x
shortfall multiplier. Drift and limited observation history create forecasting
uncertainty. We do not claim this environment already defeats frontier models.

## Data, split and provenance

Data: Yuxin Wang et al., *BurstGPT: A Real-World Workload Dataset to Optimize LLM
Serving Systems*, KDD 2025, DOI `10.1145/3711896.3737413`, arXiv `2401.17644`.
Upstream: https://github.com/HPMLL/BurstGPT, release `v2.0`.

We use hourly sums of `Total tokens` from `Conversation log` records, for the
`ChatGPT` and `GPT-4` labels. There are 231 original relative days: 0–120 and
225–334. Days 121–224 are absent in the upstream release. Episodes never cross
this gap. Missing hours within a logged day have zero demand.

All 231 selected-stream days are retained, with no score-based episode filtering.
This is an adaptation of our S11 capacity case, which previously selected these
streams for studying workload drift. It is not an unbiased sample of all LLM use.
No request text, completions, or user identifiers are distributed.

- Train decision-day segments: 14–89 and 239–304.
- Test decision-day segments: 90–120 and 305–334.
- Start points require 14 preceding observed days and 14 consecutive target days.
- There are 116 train starts and 35 test starts. Adjacent episodes overlap; they
  must not be treated as independent experimental replicates.
- Test episodes may use earlier revealed history, including earlier days of the
  test time segment; this models online deployment with expanding observations.

Set `CAPACITY_SPLIT=train` (default) or `test` when starting the server. Seeded
reset chooses uniformly from eligible starts. The split is orchestrator-owned.
The complete trace is public and shipped on the **server side**; these are not
secret, contamination-proof holdouts. Give remote agents the tools, not server
filesystem access. No live data API, secrets, or provider account is required.

To independently reconstruct the aggregate, download `BurstGPT_1.csv`,
`BurstGPT_2.csv`, and `BurstGPT_3.csv` from upstream release v2.0 into a raw directory
(gzip versions are also accepted), then run:

```bash
python scripts/rebuild_aggregates.py --raw-dir /path/to/raw --output rebuilt-hourly.csv
sha256sum rebuilt-hourly.csv
```

Expected SHA-256:
`de9db57e9dd8aa1c96f33789e45eaf5fe0b09726160e9554645d2cb8ca58c130`.
This reconstruction has been run on all three original release assets; no demand
values were modified to obtain the baseline results.

Data license: CC-BY-4.0, preserved in `server/data/LICENSE-BurstGPT.txt`.
Modification details and aggregate hashes are in `server/data/PROVENANCE.json`.
Environment code license: BSD-3-Clause. See `THIRD_PARTY_NOTICES.md`.

## Install and run

The environment supports Python 3.11 and later and pins OpenEnv 0.11.0.
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

`requirements.lock` records the separately validated Python 3.13 dependency
snapshot used for the container and standalone runtime checks. `uv.lock` is
the environment lockfile used by the repository's frozen-dependency check.

Docker:

```bash
docker build -t llm-capacity-env:0.1.0 .
docker run --rm -p 8000:8000 -e CAPACITY_SPLIT=test llm-capacity-env:0.1.0
```

`/health`, `/schema`, and `/openapi.json` are provided by OpenEnv. The debug UI is
available at `/web/` when `ENABLE_WEB_INTERFACE=true` (enabled in the Docker image).
MCP and WebSocket clients use OpenEnv's standard endpoints. The server supports
independent concurrent sessions through an environment factory.

## Baselines and validation scope

`examples/evaluate.py` evaluates zero reservation, previous-day demand, a fixed
initial-history 0.75 quantile, and an adaptive rolling 0.75 quantile over **every**
eligible start. It also computes a perfect-information cost lower bound offline.
That diagnostic reads future demand and is explicitly not a causal policy or
agent tool. Baselines are descriptive controls, not a required pass score.

Full example baseline trajectories are included under `outputs/example-traces/`.
Each records observations, MCP actions, rewards, terminal flags and next
observations for 14 decisions. Exact replay was checked independently. Reproduce:

```bash
python examples/record_trace.py outputs/traces
```

These are deterministic heuristic-policy traces, not LLM reasoning traces or
evidence of frontier-agent difficulty. Recording here is explicit through the
example runner. The API server additionally records each reset and MCP transition
(action, observation, reward, terminal flag and step count) as a separate JSONL
file under `outputs/trajectories/`. Set `CAPACITY_TRACE_DIR` to choose another
server-owned output directory. Trace files are not exposed through agent tools
and generated session files are excluded from the release package. Direct
Python use records only when `trace_dir` or `CAPACITY_TRACE_DIR` is set.
This instrumentation is not a claim of full hub telemetry certification.

Tests cover cost arithmetic, data hashes, gap handling, split boundaries, causal
observations, invalid actions, deterministic replay, terminal behavior, isolation,
MCP tool declarations, and reward accounting. A compact validation record is included in `validation-evidence/`; full raw
cluster/container logs remain in the accompanying submission preparation.
The documented static/API validation does not constitute full semantic-validator
certification, upstream approval, or a frontier-model difficulty result.

## Publish

After validating, the contributor can publish to their own Hugging Face Space:

```bash
hf auth login
openenv push --repo-id YOUR_HF_USERNAME/llm_capacity_env
```

Publishing a new personal Space does not automatically add it to an official
OpenEnv collection or merge it into the GitHub monorepo. Ask maintainers about
listing/review using the prepared contribution description and validation record.

## Contributor reference-policy study

A separate 2026-11-02 study generated three independent policies from each of
`claude-opus-5-5` and `gemini-4-argon` gateway aliases, with three train-feedback
submissions per repeat. All six policies froze before the 35-start test; all
210 test trajectories (2,940 decisions) were exactly replayed. Mean test cost
over the three programs was 1173.798785 for Opus and 1180.904867 for Gemini,
compared with the causal rolling baseline at 1240.313545. Costs are modeled
accounting units and lower is better. This was contributor-side program synthesis,
not an official hosted evaluation or LLM weight training. The evidence archive
`reference-model-evidence-20261102.tar.gz` is supplied separately for review,
including requests, responses, policies, feedback, hashes and replay instructions.

A separate 72-setting-per-deployment causal policy parameter fit selected the
original baseline and showed no test improvement. These results demonstrate
policy differences and measured baseline improvement, without establishing a
hard frontier-model benchmark, ordered model capability tiers, an optimal causal
ceiling or successful RL training. Test episodes overlap; they are not independent
statistical samples.

## Additional difficulty and trainability evidence (2026-11-02)

A completed extension trains a 32-hidden-unit neural quantile policy with Adam
on the exact environment cost. Across three seeds per initialization, cold-start
cost drops 45.20%; training a residual on
the existing rolling baseline drops cost 3.93%
(conditional 95% interval [0.94%, 6.84%]).
These statistics aggregate 61 unique test dates and use seven-day time blocks,
not independent-sample inference over overlapping episodes. This demonstrates
held-out improvement within these public recorded segments, including improvement
from a useful baseline rather than only a weak initialization.

There are 24 fitting configurations (two arms, three seeds, two learning rates,
two regularizers), 80 epochs each, 82 fit days and 60 validation days. There are 2,480 minibatch updates per configuration (59,520 total);
training-label collection covers 116 overlapping 14-decision episodes, 1,624
replay decisions and 142 unique label days. Only
validation selects configuration/checkpoint; all six selections are frozen before
testing all 35 starts. Test records were used in earlier studies; they are not
claimed to be previously unseen by all authors or contamination-proof. Training
uses labels revealed after legal training actions. It is supervised neural policy
fitting, not LLM weight fine-tuning or policy-gradient RL.

A new controlled program-synthesis comparison uses Haiku 4.5 and Opus 5.5 gateway
aliases, three independent programs each, three train-feedback submissions each,
the same prompt/train starts/sandbox and 32,768 output tokens with 8,192 thinking
tokens per response. All six programs freeze before test. Opus/Haiku three-program mean 14-day costs were 1169.929902/1274.433480. The separate unique-day analysis gives 6.47% cost reduction for Opus, conditional 95% interval [0.79%, 12.79%].
The full interval and per-program results, including unfavorable results, are
retained. API aliases/returned IDs are checked, not independently vendor-attested.

The new evidence contains 280 initial/trained test episodes, 12 fixed-policy
repeat episodes and 210 matched-model test episodes. All **502 episodes / 7,028
decisions** pass independent exact replay, source/data/hash and API-budget checks;
frozen neural weights also reproduce actions. Fixed-policy repeats have zero
observed cost range, which checks deterministic replay rather than distributional
robustness. Learning curves, all 192 checkpoints, model requests/responses,
feedback and selection records are archived in `quality-evidence-20261102.tar.gz`.

The earlier small-grid fitting negative result remains valid for that narrower
policy family. These results establish measured learnability and characterize
policy/model differences on the recorded data; they do not establish a universal
frontier-hardness level, causal optimum, or transfer from training to general LLM
capability. The hindsight bound reads future demand and is only a diagnostic.
The reward/data/environment implementation is unchanged by this extension.
