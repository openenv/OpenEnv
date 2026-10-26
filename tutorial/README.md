# OpenEnv Walkthrough

![OpenEnv Walkthrough](./images/banner.png)

A hands-on guide to building, deploying, and scaling RL environments with [OpenEnv](https://github.com/huggingface/OpenEnv).

## Guide Structure

| Section | Description |
|---------|-------------|
| [1. Environments](./01-environments.md) | OpenEnv fundamentals, architecture, and OpenSpiel integration |
| [2. Deployment](./02-deployment.md) | Local dev, Docker, and HF Spaces deployment |
| [3. Scaling](./03-scaling.md) | WebSocket scaling and infrastructure benchmarks |
| [4. Training](./04-training.md) | GRPO training with TRL on Wordle |

## File Structure

```
tutorial/
├── 01-environments.md          # OpenEnv fundamentals and architecture
├── 02-deployment.md            # Local, Docker, and HF Spaces deployment
├── 03-scaling.md               # WebSocket scaling and benchmarks
├── 04-training.md              # GRPO training tutorial
└── examples/
    ├── OpenEnv_Tutorial.ipynb  # Interactive Colab notebook
    ├── wordle.py               # Wordle environment example
    ├── wordle_prompt.txt       # System prompt for Wordle
    ├── repl_with_llm.py        # REPL with LLM example
    └── unsloth_2048.ipynb      # Unsloth 2048 environment example
```

## Resources

| Resource | Link |
|----------|------|
| OpenEnv Repository | [github.com/huggingface/OpenEnv](https://github.com/huggingface/OpenEnv) |
| Environment Hub | [huggingface.co/collections/openenv](https://huggingface.co/collections/openenv/environment-hub) |
| TRL Integration Docs | [huggingface.co/docs/trl/openenv](https://huggingface.co/docs/trl/openenv) |
| Scaling Experiments | [github.com/burtenshaw/openenv-scaling](https://github.com/burtenshaw/openenv-scaling) |