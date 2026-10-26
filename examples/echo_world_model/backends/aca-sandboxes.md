# ECHO rollouts in ACA Sandboxes

> Illustrative. `ACASandboxProvider` ships in huggingface/OpenEnv#793 and is
> not part of this PR. The snippet below is the wiring for when that provider lands;
> the CPU demos in this folder run in-process and need none of it.

ECHO needs rollouts, and rollouts need somewhere to run the agent's commands. An
**Azure Container Apps Sandbox** spins up an isolated container, runs the command,
returns the observation, and runs the verifier for reward, with default-deny egress
and governance. With the `ACASandboxProvider` from
[huggingface/OpenEnv#793](https://github.com/huggingface/OpenEnv/pull/793) as the
execution backend:

## The wiring

The OpenEnv terminal Environment (the real version of
[`mini_terminal_env.py`](../mini_terminal_env.py)) is hosted inside the sandbox. The
trainer rolls the policy out against it and tags each token by role for ECHO:

```python
from openenv.core.containers.runtime.aca_provider import ACASandboxProvider

# 1. start the terminal env inside a governed sandbox
provider = ACASandboxProvider(
    anonymous_port=True,
    egress_policy=ACASandboxProvider.deny_all_egress(allow=["<model-endpoint>"]),
    cmd="python -m my_terminal_env.server --host 0.0.0.0 --port 8000",
)
base_url = provider.start_container("disk:incident-triage")
provider.wait_for_ready(base_url)

# 2. roll the policy out, recording per-token role masks (trajectory.py)
async with GenericEnvClient(base_url=base_url) as client:
    obs = await client.reset()
    for _ in range(max_turns):
        action = policy.act(obs)                 # action tokens
        result = await client.step(action)       # runs in the sandbox
        record(action_role="action", obs=result.observation, role="env_output")
        if result.done:
            break

# 3. tokenize with masks and apply the ECHO loss (echo_loss.py) in your trainer
```

For the full GPU run, point your trainer (SkyRL / Tinker / Foundry Fine-Tuning) at a
pool of these sandboxes for parallel rollouts, one `ACASandboxProvider` per rollout
worker. The CPU demo in this folder skips the sandbox (the env is a tiny in-process
deterministic terminal) so it stays creds-free and reproducible; this file is the
bridge to the real, governed runtime.
