# ECHO on Foundry Fine-Tuning (Microsoft-specific)

Foundry Fine-Tuning exposes a Tinker-style training loop: you own the data and the
loop, the service owns GPU topology and sharding. The step shape is the same as
[Tinker](./tinker.md):

```python
for step in range(num_steps):
    batch = dataset.get_next_batch()                              # you own batching / curriculum
    sampling_client = training_client.save_weights_and_get_sampling_client()
    rollouts = rollout(sampling_client, batch)                    # act in the env
    advantages = group_relative(rollouts)                         # GRPO
    training_client.forward_backward(data, loss_fn=...)           # ECHO plugs in here
    training_client.optim_step(...)
```

ECHO is the same one-line change as the [Tinker](./tinker.md) adapter: in the data you
hand to `forward_backward`, put a constant positive advantage `lambda` on the
`env_output` tokens (this example's `obs_mask`) alongside the GRPO advantage on the
action tokens. No new loss function.

> This path is Microsoft-specific (needs a Foundry project endpoint and an API key).
> It is included so the same ECHO config travels across backends. For an open,
> reproducible run use [SkyRL](./skyrl.md).
