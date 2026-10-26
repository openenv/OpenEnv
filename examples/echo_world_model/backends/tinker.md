# ECHO on Tinker

[Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) exposes the training
primitives while its service owns the GPUs. This runnable example sends the
existing terminal rollouts to a remote LoRA trainer and measures held-out
environment-token cross-entropy before and after training.

## Run it

From `examples/echo_world_model`:

```bash
export TINKER_API_KEY="..."
uv run backends/tinker_echo_demo.py --steps 15
```

The inline dependency pins `tinker==0.22.7`. Tinker requires an account, API
key, and available credits. The default is a rank-16 LoRA on
`Qwen/Qwen3.5-4B`; use `--model` to select another supported public model.

## How the role masks map to Tinker

For each role-tagged rollout, the script tokenizes every segment and creates a
causal-language-model `Datum`:

```python
datum = tinker.types.Datum(
    model_input=tinker.types.ModelInput.from_ints(token_ids[:-1]),
    loss_fn_inputs={
        "target_tokens": token_ids[1:],
        "weights": normalized_obs_mask[1:],
    },
)
```

The shift matters: input position `t` predicts target token `t + 1`, so the loss
weight comes from the **target** token's role. Only `env_output` has non-zero
weight; action, context, and warning tokens remain zero-weight conditioning
context. Batch-wide normalization makes Tinker's sum-reduced cross-entropy equal
the mean env-token CE used by the local demo.

Each step follows Tinker's documented training path:

```python
fwdbwd = training_client.forward_backward(data, "cross_entropy")
optim = training_client.optim_step(tinker.types.AdamParams(learning_rate=2e-4))
result = fwdbwd.result()
optim.result()
```

`TrainingClient.forward(..., "cross_entropy")` evaluates held-out data before
and after the loop without accumulating gradients.

## Scope

This is the existing **verifier-free** ECHO objective (`use_rl=False`), not the
full `L_GRPO + lambda * L_env` hybrid. It isolates the OpenEnv integration seam:
preserving token roles and putting cross-entropy weight only on environment
outputs.
