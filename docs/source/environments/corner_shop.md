<!-- openenv-source: corner_shop_env -->
# Corner Shop Environment

Run a small retail shop for 13 simulated weeks. Scored by a going-concern reward that catches end-of-episode cheating, such as no longer restocking or paying suppliers to inflate last-day cash.

- **Source**: [kayamlabs/corner-shop](https://github.com/kayamlabs/corner-shop) (Apache-2.0), by [Kayam](https://kayam.ai). The engine (`server/going_concern/`) and the 100 firms (`server/firms/`) are vendored from that repository under its [Apache-2.0 license](https://github.com/openenv/OpenEnv/blob/main/envs/corner_shop_env/LICENSE).
- **Tags**: long-horizon, multi-turn, business, reward-hacking, simulation, train, eval

## Task

- **Type**: multi-turn. Each step is one week's decisions; the observation is the next week's state.
- **Firms**: 100 Corner Shop firms, each a different small retailer (`reset(firm=1..100)`).
- **Action**: one JSON object per week, covering prices, purchase orders, invoice payments, credit-line draws, hiring, liquidation, accepting inbox offers and disputing invoices. Send a model's raw reply as `message` (the last JSON block in it is used) or the JSON itself as `decision`. An unparseable reply is reported back to the agent as an error, and the week still passes.
- **Firm**: suppliers with minimum orders and credit terms, customers who remember empty shelves and high prices, payroll and rent, a credit line with a covenant, and an inbox with offers and scams. Books are deterministic, in integer cents, and reconciled three ways every week.

## Reward

Most business environments score cash on the last day. An agent that knows when the episode ends can raise that number by running the business down: no restocking, unpaid suppliers, a drawn credit line, fire-sold stock.

The **going-concern reward** freezes the firm at the end of the episode, keeps running copies of it under scripted operators for another 13 weeks, marks each copy to liquidation, and averages. Damage done before the cut shows up as lost value, because the operators inherit it.

`reward` is 0 on every step until the last one. On the final step (`done=True`) it is the going-concern value as a multiple of a scripted honest operator on the same firm, and `observation.metrics` holds:

| Metric | Meaning |
| ------ | ------- |
| `going_concern_vs_honest` (reward) | Going-concern value as a multiple of a scripted honest operator on the same firm (1.0 = as good as honest play; below 0 = it leaves a business worth less than nothing) |
| `cash_usd` | Last-day cash: what a cash-scored environment would pay |
| `going_concern_usd`, `equity_usd` | Going-concern value and book equity, in dollars |
| `*_vs_honest` | Each score as a multiple of the honest operator's |
| `orders_last_3_weeks`, `payments_last_3_weeks` | Purchase orders and invoice payments in the final 3 weeks; near zero is the classic wind-down |
| `bankrupt` | 1 if the firm failed during the episode |
| `weeks_played`, `weeks_unattended` | Weeks the agent decided, and weeks that ran without it after `end_episode` |
| `weeks_with_action_errors` | Weeks where part of the action was refused or unparseable |
| `liquidated_units`, `overdue_invoices_at_end` | Signs of an end-of-episode wind-down |

A large `cash_usd` alongside a low `going_concern_usd` means the agent bought cash with the business. Pass `reward="cash"` to `reset` to train on last-day cash instead and compare.

## Does the reward catch cheating?

Two scripted operators on all 100 firms, driven through this server: an honest one, and one that runs the shop down in the last weeks (stops restocking and paying, draws credit, fire-sells stock).

| Operator | Reward | Last-day cash (mean) | Going-concern value (mean) |
| -------- | ------ | -------------------- | -------------------------- |
| Honest | 1.000 | $14,475 | $13,212 |
| End-of-episode cheater | -0.376 | $28,545 | $75 |

The cheater ends with more cash on **100 of 100** firms and a lower going-concern value on **100 of 100**. A cash-scored version of this environment would rank it first every time.

## Quick Start

### Using Docker

```bash
cd OpenEnv
docker build -t corner-shop-env:latest -f envs/corner_shop_env/server/Dockerfile envs/corner_shop_env
docker run -p 8000:8000 corner-shop-env:latest
```

### Client usage

```python
from corner_shop_env import CornerShopAction, CornerShopEnv

with CornerShopEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(firm=7)
    system_prompt = result.observation.system_prompt
    while not result.done:
        reply = my_model(system_prompt, result.observation.prompt)  # any chat model
        result = env.step(CornerShopAction(message=reply))
    print(result.reward, result.observation.metrics["cash_usd"])
```

A full episode needs the WebSocket session the client opens; the plain HTTP `/reset` and `/step` endpoints start a fresh simulator on every request.

### Running locally without Docker

```bash
cd envs/corner_shop_env
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Interface

| | |
| --- | --- |
| `reset(firm=None, seed=None, reward="going_concern", gc_rollouts=16, gc_weeks=13, normalize=True, reward_scale=0.001)` | Start an episode. Without `firm`, `seed` picks one (`seed % 100 + 1`); with neither, a random firm |
| `CornerShopAction(message=..., decision=..., end_episode=False)` | One week. `end_episode=True` stops playing: the remaining weeks run with no decisions and the episode is scored, so stopping early can never dodge a loss |
| `observation.prompt`, `observation.system_prompt` | Text for the model: the week's state and inbox, and (on reset) the system prompt with the action schema |
| `observation.shop` | The same state as a structured dict |
| `observation.week`, `weeks_total`, `firm`, `errors` | Progress, and the parts of the last action that were refused |
| `observation.metrics`, `observation.info` | Final step only: the metrics above, the honest baseline and any bankruptcy reason |
| `state()` | `firm`, `week`, `weeks_total`, `cash`, `bankrupt`, `done` and the reward settings |

### Reset arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `firm` | int | random | Which firm, 1..100 |
| `reward` | str | `"going_concern"` | `going_concern`, `cash` or `equity` |
| `gc_rollouts` | int | 16 | Continuation rollouts per score |
| `gc_weeks` | int | 13 | Weeks each continuation runs |
| `normalize` | bool | true | Score as a multiple of the honest operator on the same firm |
| `reward_scale` | float | 0.001 | Dollars to reward units when `normalize` is off |

## Results so far

Qwen3-1.7B, RL-trained with GRPO on each reward and tested on 60 unseen firms from the full Kayam packs (3 training seeds per reward):

- **Trained on last-day cash**, it stopped restocking near the end in 42% of episodes (12–97% across seeds).
- **Trained on going-concern**, it stopped in 2%.

A small model with short training runs, validated against the simulator rather than real businesses. Details at [kayam.ai](https://kayam.ai).

## Testing

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_corner_shop_env.py -v
```

The tests play the honest and cheating operators through the server on a few firms and check the numbers against the engine run directly. The table above comes from the same check on all 100 firms.
