# ECHO on SkyRL (the reference implementation)

ECHO's reference code is **`github.com/microsoft/echo-rl`**, built on
**SkyRL** (`NovaSky-AI/SkyRL`, FSDP + Ray + vLLM), which runs terminal tasks in
an isolated container backend (spins up task containers, runs commands, returns
observations, runs the verifier for reward).

The point of this folder is that an OpenEnv rollout — once it carries the
[per-token role masks](../trajectory.py) — drops straight into a SkyRL run.

## The masks map 1:1

| OpenEnv `trajectory.py` | SkyRL / echo-rl |
|---|---|
| `action_mask` | `completion_masks` (the GRPO target) |
| `obs_mask` (`env_only`) | `completion_observation_masks` ∧ `completion_env_output_masks` |
| `warning_mask` | `completion_warning_masks` (excluded from the env loss) |

## The config is the same knobs

```yaml
# echo-rl config — the *entire* vanilla→ECHO diff is one line:
world_model_coeff: 0.05        # λ. 0.0 = vanilla GRPO. (matches echo_loss.world_model_coeff)
world_loss_target: env_only    # train on real terminal output, exclude harness warnings
# rollout filters — only let clean rollouts contribute the env loss:
wm_filter_min_valid_tool_call_pct: 0.5
wm_filter_parse_clean_pct: 0.5
wm_filter_correct_pct: 0.0
```

Reported results (Qwen3-8B/14B terminal agents): **~2.3× faster** to equal
score; TerminalBench-2.0 pass@1 ~**doubles** (8B 2.7→5.2; 14B 5.2→10.8);
recovers **50–104%** of expert-SFT gain with no teacher; verifier-free env-only
improves OOD (+3.8/+5.2/+10.0 pp).

> arXiv `2605.24517` — *"Terminal Agents Learn World Models for Free."*
