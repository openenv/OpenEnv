# Learn

Choose a learning path from the sidebar:

- **Basics:** start with [Hello World](openenv-tutorial), build [Your First Environment](../guides/first-environment) and [deploy it](../getting_started/environment-builder).
- **Training:** [the ways to train and the supported frameworks](../guides/training), then train a [reasoning model](end-to-end-walkthrough), play [Wordle](wordle-grpo) or [2048](rl-training-2048) with GRPO, or [collect rollouts for SFT](sft-warmup). Each framework's own examples (ART, Miles, Oumi, SkyRL, torchforge, TRL, Unsloth, …) are listed in [Integrations](../guides/training#integrations).
- **Harnesses:** [pick a path](harnesses): train with the trainer's own loop on [BrowserGym](browsergym-harness) (white-box), train real agents through [Harbor](harbor-harness) (black-box), or [evaluate Claude Code inside an environment](claude-code-harness).
- **Evals:** follow [Evaluating with Environments](evaluation-inspect).

For [MCP environments](mcp-environment) and [rubrics](rubrics), see Concepts in the sidebar.

For end-to-end projects built on OpenEnv environments, from designing an environment to training against it at scale, see [FineEnvs](https://github.com/adithya-s-k/FineEnvs) and its environments, datasets and models on [the Hub](https://huggingface.co/FineEnvs).

## New to OpenEnv? Start Here

The Getting Started Series walks you from zero to deploying your own environment in five short parts. No GPU required.

| Part | What it covers | Notebook |
|------|---------------|----------|
| 1 — Introduction & Quick Start | What OpenEnv is, why it exists, and your first environment in under 10 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/openenv_introduction_quickstart.ipynb) |
| 2 — Using Environments | Connect to environments, create policies, run evaluations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/openenv_using_environments.ipynb) |
| 3 — Building Environments | Create a custom environment from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/openenv_building_environments.ipynb) |
| [4 — Deploying an Environment](../getting_started/environment-builder) | Package with Docker and deploy to Hugging Face | — |
| [5 — Contributing Environments](../getting_started/contributing-envs) | Publish, fork, and share environments on the Hub | — |

## Topic Tutorials

Already familiar with the basics? These tutorials cover specific workflows in depth.

| Tutorial | What it covers | GPU | Notebook |
|----------|---------------|-----|----------|
| [Hello World](openenv-tutorial) | Install OpenEnv, run an environment on a local server, and build one from scratch: models, environment, server app and client. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) |
| [Train a Reasoning Model](end-to-end-walkthrough) | The full pipeline: connect to `reasoning_gym`, wire it into TRL via `environment_factory`, fine-tune with GRPO, and push the checkpoint to the Hub. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/end_to_end_walkthrough.ipynb) |
| [MCP Environments](mcp-environment) | Consume and build MCP-backed environments: list and call tools through `step()`, register Python functions as tools with FastMCP. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/mcp_environment.ipynb) |
| [Rubrics](rubrics) | Compose reward functions from reusable pieces using `Gate`, `WeightedSum`, `LLMJudge`, and `TrajectoryRubric`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/rubrics.ipynb) |
| [Play Wordle with GRPO](wordle-grpo) | Train an agent to play Wordle using GRPO via TRL's `environment_factory`. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_wordle/grpo_wordle.ipynb) |
| [Play 2048 with GRPO](rl-training-2048) | Train a language model to play 2048 on the OpenSpiel environment with TRL's `GRPOTrainer` and `environment_factory`. Unsloth's own [2048 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) trains `gpt-oss-20b` to write a 2048 strategy. | Yes | — |
| [Evaluating with Environments](evaluation-inspect) | Wrap an OpenEnv environment in an Inspect AI `Task`, run it via `InspectAIHarness`, and get a structured `EvalResult`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/evaluation_inspect.ipynb) |
| [White-Box: Train a Web Agent on BrowserGym](browsergym-harness) | Train a vision-language model on BrowserGym web tasks with TRL's `GRPOTrainer` and `environment_factory`, so TRL runs the multi-turn tool loop. | Yes | — |
| [Black-Box: Train Real Agents with Harbor](harbor-harness) | Run real agents (OpenCode, Claude Code, Codex, …) on Harbor tasks with their token ids and logprobs captured, and train the model behind them with TRL's `AsyncGRPOTrainer`. | Yes | — |
| [Evaluate Claude Code in an Environment](claude-code-harness) | Run Claude Code inside an OpenEnv environment with `HarnessEnvironment` (RFC 005), inject the environment's tools over MCP, and evaluate it on τ²-bench against a simulated customer. Also serves it in production mode. | No | — |
| [Collect Rollouts for SFT](sft-warmup) | Run a teacher model to collect reward-labeled rollouts, filter them, and fine-tune a student with TRL's `SFTTrainer` as a warm-start for GRPO. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/sft_warmup.ipynb) |
