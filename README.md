<img width="1920" height="1080" alt="OpenEnv-banner" src="https://github.com/user-attachments/assets/380f1b86-fc36-44f8-a77b-067773b5012d" />

# OpenEnv: Agentic Execution Environments

An end-to-end framework for creating, deploying and using isolated execution environments for agentic RL, with a simple Gymnasium-style API.

<p align="center">
    <a href="https://pypi.org/project/openenv/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openenv?color=blue"/></a>
    <a href="https://github.com/huggingface/OpenEnv/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"/></a>
    <a href="https://huggingface.co/docs/openenv"><img alt="Docs" src="https://img.shields.io/badge/Docs-Explore-blue?logo=readthedocs&logoColor=white"/></a>
    <a href="https://huggingface.co/openenv"><img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-OpenEnv-yellow"/></a>
    <a href="https://discord.gg/YsTYBh6PD9"><img alt="Discord" src="https://img.shields.io/badge/Discord-OpenEnv-7289da?style=flat&logo=discord&logoColor=white"/></a>
    <a href="https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

## What you get

- **One API for every environment.** `reset()`, `step()` and `state()`, sync or async, over a WebSocket.
- **Isolated, deployable environments.** Each environment is a Docker image that runs locally, on a cloud sandbox, or as a [Hugging Face Space](https://huggingface.co/docs/openenv/getting_started/contributing-envs).
- **40+ ready-to-use environments**, from games and coding sandboxes to browsers, finance and simulators, in the [environment catalog](https://huggingface.co/docs/openenv/environments).
- **Train with your framework.** TRL, Unsloth, SkyRL, ART, Oumi, torchforge, Miles and more (see [Integrations](#integrations)).
- **Train real coding agents.** The [Harbor integration](https://huggingface.co/docs/openenv/environments/harbor) runs Claude Code, Codex, OpenCode, mini-swe-agent and other harnesses on Harbor tasks, and captures the exact tokens for RL.
- **Serve tools to agents.** MCP environments expose their tools to agents over `/mcp` in [production mode](https://huggingface.co/docs/openenv/guides/simulation-vs-production), while training keeps using `step()` and rewards.
- **Rewards and evals built in.** Compose rewards with [rubrics](https://huggingface.co/docs/openenv/guides/rewards) and evaluate with [Inspect AI](https://huggingface.co/docs/openenv/tutorials/evaluation-inspect).

## Quick Start

Install the OpenEnv package:

```bash
pip install openenv
```

Install an environment client (e.g., Echo):

```bash
pip install git+https://huggingface.co/spaces/openenv/echo_env
```

Then use the environment:

```python
import asyncio
from echo_env import CallToolAction, EchoEnv

async def main():
    # Connect to a running Space (async context manager)
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        # Reset the environment
        result = await client.reset()
        print(result.observation.metadata["message"])  # "Echo environment ready!"

        # Send messages
        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, World!"},
            )
        )
        print(result.observation.result["data"])  # "Hello, World!"
        print(result.reward)

asyncio.run(main())
```

**Synchronous usage** is also supported via the `.sync()` wrapper:

```python
from echo_env import CallToolAction, EchoEnv

# Use .sync() for synchronous context manager
with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello, World!"},
        )
    )
    print(result.observation.result["data"])
```

For a detailed quick start, check out the [docs page](https://huggingface.co/docs/openenv/getting-started).

## Train an agent

Any training framework that can call an environment can train on it. [Training with OpenEnv](https://huggingface.co/docs/openenv/guides/training) maps the ways to train and the frameworks that support each one.

- **Environments as tools (white-box).** The trainer runs the multi-turn tool loop and the environment supplies the tools and the reward. Each framework under [Integrations](#integrations) has its own example. With TRL, `GRPOTrainer` takes an `environment_factory`: start with the [Wordle GRPO tutorial](https://huggingface.co/docs/openenv/tutorials/wordle-grpo) or [TRL's OpenEnv guide](https://huggingface.co/docs/trl/openenv).
- **Real agent harnesses (loop-owning).** The agent runs its own loop, and OpenEnv's [Harbor integration](https://huggingface.co/docs/openenv/environments/harbor) captures every model call as a framework-neutral `TrainingTrace`. TRL's `AsyncGRPOTrainer` trains on those captures today: see [`examples/async_grpo_harbor`](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor) and [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl).

## Build your own environment

```bash
openenv init my_env       # scaffold an environment
openenv validate my_env --level static --skip-build   # quick check against the OpenEnv contract
openenv push my_env       # deploy it to Hugging Face Spaces
```

See [Your First Environment](https://huggingface.co/docs/openenv/guides/first-environment) and [Deploying an Environment](https://huggingface.co/docs/openenv/getting_started/environment-builder). `openenv import` wraps an existing environment from ORS/OpenReward or Verifiers.

## Environments

A few to start with:

| Environment | What it is |
|---|---|
| [Echo](https://huggingface.co/docs/openenv/environments/echo) | Minimal MCP environment, for learning the API and testing a deployment |
| [Coding](https://huggingface.co/docs/openenv/environments/coding) | Sandboxed Python execution with stdout, stderr and exit codes |
| [TextArena (Wordle and more)](https://huggingface.co/docs/openenv/environments/textarena) | Text games for multi-turn RL |
| [OpenSpiel](https://huggingface.co/docs/openenv/environments/openspiel) | Board and card games from DeepMind's OpenSpiel |
| [BrowserGym](https://huggingface.co/docs/openenv/environments/browsergym) | Web navigation tasks (MiniWoB++, WebArena, ...) |
| [Harbor](https://huggingface.co/docs/openenv/environments/harbor) | Harbor task datasets through coding-agent harnesses, with token capture for training |

Browse all of them in the [environment catalog](https://huggingface.co/docs/openenv/environments), or on the [OpenEnv Hub organization](https://huggingface.co/openenv).

## Integrations

OpenEnv works with a growing ecosystem of RL frameworks and platforms. If your project supports OpenEnv, open a PR to add it here.

| Framework | Example |
|---|---|
| ART | [ART integration](https://art.openpipe.ai/integrations/openenv-integration) |
| Lightning AI | [Templates](https://lightning.ai/templates?section=featured&query=openenv) |
| Miles | [Terminal-Bench-2 GRPO](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |
| Oumi | [GRPO notebook](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20OpenEnv%20GRPO%20with%20trl.ipynb) |
| SkyRL | [SkyRL example](https://skyrl.readthedocs.io/en/latest/examples/openenv.html) |
| torchforge | [GRPO BlackJack](https://github.com/huggingface/OpenEnv/tree/main/examples/grpo_blackjack) |
| TorchTitan-RL (TitanRL) | [OpenEnv bridge and Muse Glimmer 30B chess recipe](examples/titanrl_openenv/) for GRPO/DAPO, with a CPU demo and tests |
| TRL | [OpenEnv guide](https://huggingface.co/docs/trl/openenv) (GRPO with `environment_factory`, and harness training) |
| Unsloth | [2048 with gpt-oss](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) |

## Learn more

- [Documentation](https://huggingface.co/docs/openenv): concepts, guides and tutorials
- [Core Concepts](https://huggingface.co/docs/openenv/guides/concepts) and the [CLI reference](https://huggingface.co/docs/openenv/reference/cli)
- [Tutorials](https://huggingface.co/docs/openenv/tutorials/index), and the [Zero to Hero tutorial](https://github.com/huggingface/OpenEnv/tree/main/tutorial) from our GPU Mode lecture
- [FineEnvs](https://github.com/adithya-s-k/FineEnvs): end-to-end projects that build, deploy and train against OpenEnv environments
- [RFCs](https://github.com/huggingface/OpenEnv/tree/main/rfcs), the proposals behind major changes
- [Contributing](https://github.com/huggingface/OpenEnv/blob/main/CONTRIBUTING.md): development setup, tests and the PR process

> [!NOTE]
> OpenEnv is in early development, so APIs may still change. Bug fixes are welcome. For larger changes, open or claim an issue first so the change can be discussed.

## Community Support & Acknowledgments

OpenEnv is governed by a technical committee that coordinates project direction, major technical decisions, RFCs, and release planning through the public issue tracker, pull requests, and RFC process. Current committee members: Meta-PyTorch, Reflection, Unsloth, Modal, Prime Intellect, Nvidia, Mercor, Fleet AI, Microsoft, Hugging Face, RadixArk, and Nebius.

The project is also supported by a broader community of organizations. If you would like to add your project or organization here, please open a pull request for maintainer review.

Supporters include: [Meta-PyTorch](https://github.com/meta-pytorch), [Hugging Face](https://huggingface.co), [Scaler AI Labs](https://scalerailabs.com), [Patronus AI](https://patronus.ai), [Surge AI](https://surgehq.ai), [LastMile AI](https://www.lastmileai.dev), [Unsloth](https://unsloth.ai), [Reflection](https://reflection.ai), [vLLM](https://vllm.ai), [SkyRL](https://skyrl.readthedocs.io) (UC-Berkeley), [Lightning AI](https://lightning.ai), [Axolotl AI](https://github.com/axolotl-ai-cloud/axolotl), [Stanford Scaling Intelligence Lab](https://scalingintelligence.stanford.edu/), [Mithril](https://mithril.ai), [OpenMined](https://openmined.org/), [Fleet AI](https://fleetai.com), [Halluminate](https://halluminate.ai/), [Turing](https://www.turing.com/), [Scale AI](https://scale.com/), [Scorecard](https://www.scorecard.io/), [Snorkel AI](https://snorkel.ai/), [SGLang](https://github.com/sgl-project/sglang), [Miles](https://github.com/radixark/miles), [Nebius](https://nebius.com)

And we'd also like to acknowledge the team at Farama Foundation as the OpenEnv API was heavily inspired by the work you all have done on Gymnasium. Cheers!

## License

BSD 3-Clause License (see [LICENSE](./LICENSE) file)
