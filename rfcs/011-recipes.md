# RFC: GRPO Harness Training Recipes

OpenEnv has a standard interface for agentic environments, and RFC 005 describes how external agentic harnesses can run inside an OpenEnv-managed execution boundary. What is still missing is a standard recipe for the workflow that motivated much of this design: training a model with GRPO while a harness remains in the interaction loop and an OpenEnv environment remains the source of task state, tools, observations, and reward.

Today, each project that wants this shape has to rediscover the same contract:

- how the trainer presents a prompt or task to the harness
- how the harness interacts with the OpenEnv environment during a rollout
- how environment observations and harness events become the trajectory seen by GRPO
- where reward is computed and how the scalar reward is returned to the trainer
- how episode boundaries, reset semantics, tool access, logging, and failure handling work
- what evidence is required before a workflow can be considered reproducible

Without shared recipe expectations, examples tend to struggle on framework-specific tradeoffs. That makes them useful as demos but weak as OpenEnv design artifacts. The project needs an RFC-backed **family of framework-specific recipes** that all share the portable OpenEnv shape of "GRPO with a harness in the loop" without binding OpenEnv to any one trainer, model serving stack, or orchestration backend.

## Goal

The goal of this RFC is to bring together all other RFCs into a unified examples for training frameworks. In doing so, we will:

- share common recipes with the community to train on OpenEnv environments
- share common challenges and solutions with framework authors
- share reliable recipe templates for model authors releasing new models

## Proposed Solution

Define canonical **GRPO Harness Training Recipes** for OpenEnv.

These recipes are framework-neutral workflow contracts, not new OpenEnv core APIs. They describe the roles, boundaries, required artifacts, rollout semantics, reward path, and validation expectations for training a model with:

- a GRPO-style optimizer or trainer
- a model policy being updated by that trainer in the harness
- an agentic harness that owns the model interaction loop for each turn
- an OpenEnv environment that owns task state, tool surfaces, observations, reward, reset, and reproducibility

Each training framework should have its own concrete recipe that implements this shared contract in the idioms of that framework. A TRL recipe, an Unsloth recipe, a Miles recipe, a SkyRL recipe, and internal lab recipes can differ in launch mechanics, model serving, adapter configuration, distributed execution, checkpointing, and logging.

### What The Recipe Must Specify

A GRPO harness training recipe should specify the following, in prose or another representation chosen by the eventual implementation:

- the training framework that owns the concrete recipe, such as TRL, Unsloth, Miles, SkyRL, or another GRPO-capable stack
- the target OpenEnv environment and task distribution
- the harness used in the loop and the boundary where it connects to the environment
- the policy model family and the model interface expected by the harness or trainer
- the GRPO grouping semantics, including what constitutes one prompt, one rollout, and one reward-bearing sample
- the episode lifecycle, including reset, start context, turn loop, termination, and cleanup
- the reward path from environment computation to trainer input
- the trajectory record required for debugging, replay, and training audit
- the runtime assumptions: hardware class, parallelism, time budget, network access, secrets, and model serving requirements
- the smoke-validation bar for a cheap run
- the full-run evidence needed before claiming a recipe is validated
- the expected outputs, including trained model artifacts, logs, metrics, trajectories, and any published model or dataset references

This RFC intentionally does not prescribe a serialization schema. The first implementation may choose its own representation, but it should preserve the contract above.

### Non-goals

- Do not define a universal training-framework API.
- Do not require OpenEnv to own GRPO implementation details.
- Do not prescribe any specific serialization format in this RFC.
- Do not collapse all training frameworks into OpenEnv-maintained code.
- Do not require every OpenEnv environment to support harness-based GRPO training.
- Do not make the harness responsible for environment reward computation.
- Do not expose `reset`, `step`, or `state` as agent-callable tools.
- Do not require a single trajectory storage format before the rollout schema RFCs settle.
- Do not claim scientific superiority for a recipe solely because it is validated as runnable.

## Open Questions

1. Which harness should be the first reference implementation?
2. Which OpenEnv environment is simple enough for smoke validation but representative enough to exercise harness-mediated tool use?
