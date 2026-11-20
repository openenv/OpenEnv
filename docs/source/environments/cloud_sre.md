<!-- openenv-source: cloud_sre_env -->
# Cloud SRE & FinOps Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant environment for training and evaluating AI agents on **Cloud Site Reliability Engineering (SRE)** and **Financial Operations (FinOps)** tasks.

The agent manages a simulated cloud infrastructure: diagnosing outages, terminating idle resources, scaling services, and optimizing costs — all without causing collateral damage to production workloads.

## Features

| Feature | Description |
|---|---|
| **3 Difficulty Tiers** | Easy → Medium → Hard tasks covering cost optimization, scaling, and incident response |
| **Deterministic Grading** | Fine-grained scoring breakdowns for agent debugging |
| **Seeded Procedural Gen** | Reproducible yet varied infrastructure layouts for RL training |
| **Chaos Injection** | Random cost spikes, CPU anomalies, and spurious alerts |
| **Zero Dependencies** | No external API keys or cloud accounts needed |

## Reward and grading decision

`step()` returns the per-action shaping reward in the observation. It gives
immediate feedback for useful and harmful operations, including penalties for
waiting or damaging production resources. The cumulative sum is available in
`state.cumulative_reward`.

`grade()` is a separate deterministic episode evaluator. Its task score and
breakdown are not added to the step reward or returned by the environment
protocol. Training therefore uses the shaping reward; use `grade()` for
evaluation and task-level reporting. Keeping these signals separate avoids
mixing a delayed episode score into the per-action reward scale. A training
pipeline that needs the task score must currently run in-process with the
environment and call `env.grade()` after the episode. `CloudSREEnv` does not
expose this evaluator over WebSocket; remote clients receive shaping rewards
only.

## Tasks

### 1. Phantom Volume Cleanup (Easy)
- **Goal:** Find and terminate unattached EBS volumes wasting money
- **Trap:** Do NOT touch running EC2 instances or in-use volumes
- **Scoring:** +1/N per orphan terminated, −0.5 per active resource destroyed

### 2. Latency Spike Remediation (Medium)
- **Goal:** Scale an under-provisioned RDS to fix high API latency
- **Trap:** Stay within the budget limit
- **Scoring:** 40% RDS scaled + 30% latency resolved + 30% under budget

### 3. Noisy Neighbor Incident (Hard)
- **Goal:** Investigate a rogue test instance, terminate it, reboot crashed production
- **Trap:** Don't terminate production infrastructure
- **Scoring:** 20% inspect + 30% terminate rogue + 30% reboot backend + 20% alerts resolved

## Quick Start

### From Docker
```bash
# Build base image
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build Cloud SRE environment
docker build -t cloud-sre-env:latest -f envs/cloud_sre_env/server/Dockerfile .
```

### Using the Client
```python
from envs.cloud_sre_env import SREAction, CloudSREEnv

# Start the container and use the synchronous client facade.
with CloudSREEnv.from_docker_image("cloud-sre-env:latest").sync() as client:
    result = client.reset()
    result = client.step(SREAction(command="inspect", resource_id="ec2-web-001"))
    result = client.step(SREAction(command="terminate", resource_id="ebs-orphan-001"))
    state = client.state()
    print(f"Step: {state.current_step}, Reward: {state.cumulative_reward}")
```

### Direct Python Usage (no Docker)
Run from the repository checkout with `PYTHONPATH=src:envs`.

```python
from cloud_sre_env.server.cloud_sre_environment import CloudSREEnvironment
from cloud_sre_env.models import SREAction

env = CloudSREEnvironment()

# Reset with seeded procedural generation
obs = env.reset(task_id="phantom_volume_cleanup", seed=42)
print(f"Resources: {len(obs.resources)}, Alerts: {len(obs.alerts)}")

# Agent loop
for step in range(15):
    action = SREAction(command="terminate", resource_id="ebs-orphan-001")
    obs = env.step(action)
    if env.state.done:
        break

# Grade the agent
score, breakdown = env.grade()
print(f"Score: {score}, Breakdown: {breakdown}")
```

## Action Space

| Command | Description | Parameters |
|---------|-------------|------------|
| `terminate` | Remove a resource permanently | `resource_id` |
| `scale` | Change instance size | `resource_id`, `params.target_size` |
| `reboot` | Restart a stopped/running instance | `resource_id` |
| `inspect` | View detailed resource info (no side-effects) | `resource_id` |
| `wait` | Do nothing for this step | — |

## Observation Space

```json
{
  "resources": [{"id": "ec2-web-001", "type": "ec2_instance", "status": "running", ...}],
  "alerts": [{"alert_id": "alert-cost-001", "severity": "warning", "message": "..."}],
  "total_hourly_cost": 4.52,
  "system_uptime": 78.0,
  "budget_limit": 12.00,
  "task_description": "Your cloud account has unattached EBS volumes..."
}
```

## Project Structure

```
cloud_sre_env/
├── __init__.py                          # Exports SREAction, SREObservation, SREState, CloudSREEnv
├── models.py                            # Pydantic wire models (Action, Observation, State)
├── client.py                            # EnvClient subclass for HTTP/Docker communication
├── openenv.yaml                         # Environment manifest
├── README.md                            # This file
└── server/
    ├── __init__.py
    ├── cloud_sre_environment.py         # Core environment logic + 3 tasks + grading
    ├── app.py                           # FastAPI application
    ├── Dockerfile                       # Container image
    └── requirements.txt                 # Server dependencies
```

## Testing

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_cloud_sre_environment.py -v
```

## License

BSD-3-Clause — same as OpenEnv.
