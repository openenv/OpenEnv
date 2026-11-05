---
title: Fraud Triage Environment Server
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - fraud-detection
  - fintech
  - risk-decisioning
---

# Fraud Triage Environment

A sequential transaction risk-decisioning environment for RL post-training,
built for the OpenEnv specification.

## Overview

Existing OpenEnv finance environments cover stock trading ([`finrl_env`](../finrl_env))
and financial document QA ([`finqa_env`](../finqa_env)), but nothing covers
**transaction-level risk decisioning** -- the core loop of a payments fraud
or risk team. This environment fills that gap.

An agent is streamed synthetic financial transactions one at a time and must
choose one of four actions for each:

- **APPROVE** — let the transaction go through with no flag
- **FLAG** — allow it, but tag for downstream/offline review
- **ESCALATE** — hold for a human analyst before settling
- **BLOCK** — reject the transaction outright

Each transaction carries engineered risk features modeled on the kind of
signals a production anomaly-detection pipeline computes in real time:
amount, deviation from the account's historical spending (z-score),
transaction velocity over 1h/24h windows, whether the merchant or device is
new to the account, cross-border flag, time of day, and account age.

## Reward design

Reward is intentionally **asymmetric**, matching how real payment-risk teams
weigh these outcomes:

| Outcome | Reward |
|---|---|
| Correctly flag/escalate/block a fraudulent transaction (true positive) | `+1.0` |
| Correctly approve a legitimate transaction (true negative) | `+0.1` |
| Incorrectly flag/escalate/block a legitimate transaction (false positive) | `-0.5` |
| Incorrectly approve a fraudulent transaction (false negative) | `-3.0` |
| Escalation (in addition to the above) | additional `-0.1` |

This is a simplified classification task. Missing fraud is penalized more
heavily than flagging a legitimate transaction. FLAG and BLOCK have the same
payoff; ESCALATE adds a fixed cost without providing extra evidence, so it is
always less rewarding than FLAG or BLOCK. The environment does not model a
human investigation or later transaction outcomes.

Fraudulent and legitimate transactions are drawn from overlapping (not
perfectly separable) feature distributions, so the task rewards genuine
pattern-learning over simple thresholding.

## Quick Start

### Using Docker

```bash
cd OpenEnv
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t fraud-triage-env:latest -f envs/fraud_triage_env/server/Dockerfile envs/fraud_triage_env
docker run -p 8000:8000 fraud-triage-env:latest
```

### Client usage

```python
from envs.fraud_triage_env import FraudTriageEnv, FraudTriageAction, TriageDecision

with FraudTriageEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    obs = result.observation
    print(f"amount={obs.amount}, z-score={obs.amount_zscore}, new_device={obs.is_new_device}")

    # A simple heuristic policy, for illustration:
    if obs.amount_zscore > 2.5 or (obs.is_new_device and obs.is_new_merchant):
        action = FraudTriageAction(decision=TriageDecision.ESCALATE)
    else:
        action = FraudTriageAction(decision=TriageDecision.APPROVE)

    result = env.step(action)
    print(f"reward={result.reward}, done={result.done}")
```

### Running locally without Docker

```bash
cd envs/fraud_triage_env
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Configuration

The environment accepts two constructor parameters (set in `server/app.py`
or via environment-specific wiring if you fork this):

- `episode_length` (default `200`): number of transactions per episode
- `fraud_rate` (default `0.12`): fraction of transactions that are fraudulent

## Testing

```bash
PYTHONPATH=src:envs pytest tests/envs/test_fraud_triage_env.py -v
```
