# OpenEnv: Charter

This document sets out what OpenEnv is, the problem it solves, who it serves, how we grow adoption across the ecosystem, and how the Technical Committee governs the project.

The scope now spans the full agentic-RL loop, not environments alone (see below). Treat the recommendations as a starting proposal to be discussed; it will evolve as the project and the ecosystem do.

***This document is the committee's shared playbook and will evolve with the project and community.***

**Last updated:** June 10, 2026

* Context
* Goals
* The Who (and Why)
* Governance

## Context

OpenEnv launched in October 2025 as a joint project between Hugging Face and Meta (Meta-PyTorch), [introduced](https://www.youtube.com/watch?v=EflwqcbRYYs) at the PyTorch Conference as an open space for building, sharing, and running agentic environments, with an interface inspired by Gymnasium. In June 2026 the repository moved to huggingface/OpenEnv and stewardship broadened from the two founders to a multi-company Technical Committee: Meta-PyTorch, Reflection, Nvidia, Microsoft, Prime Intellect, Mercor, Fleet AI, Modal, Unsloth, BenchFlow, and Hugging Face.

### Community and adoption

* **Hackathons:** an in-person [hackathon](https://cerebralvalley.ai/e/openenv-hackathon-sf/hackathon/gallery) with 200 participants in San Francisco (March 2026) brought together data and environment partners including Mercor, Fleet AI, Snorkel AI, Scale AI, Patronus AI, and Halluminate, followed by the world’s largest AI [hackathon](https://www.linkedin.com/posts/scaler-school-of-technology_we-just-wrapped-the-grand-finale-of-indias-activity-7454776681455697920-3pVc?utm_source=share&utm_medium=member_desktop&rcm=ACoAABNg2VgBFZcy-AYn3DBx34p5fz0S0bI-JWg) in India (April 2026) with Scaler AI Labs, where ~2,000 on-site attendees contributed hundreds of environments to the hub.

* **AgentBeats challenge with UCB:** [workshop](https://www.youtube.com/watch?v=1jU05MlENOI&t=660s) with UCB on OpenEnv, sponsored by Meta-PyTorch, Hugging Face, and Unsloth.

* **PostTrain Arena (OpenEnv Arena):** an open competition launched with Hugging Face, BenchFlow, and Nebius where the community submits RL training environments evaluated on a pinned open model against held-out benchmarks.

* **Lectures and tutorials:** a GPU MODE [lecture](https://www.youtube.com/watch?v=jMSCJZAEYR8) (Reinforcement Learning, Agents & OpenEnv) and a [bootcamp](https://www.youtube.com/watch?v=kkCNMz0Ptd8) introduced the project to ML systems engineers and the wider community.

* **Discord:** with 21,000+ members on [Discord](https://discord.gg/yMzKaYjU), OpenEnv has a vibrant and growing community of developers.

* **Supporters:** dozens of organizations across the ecosystem supported and adopted, including the PyTorch Foundation, vLLM, SkyRL (UCB), Lightning AI, Axolotl AI, Stanford Scaling Intelligence Lab, Mithril, OpenMined, Scaler AI Labs, Scale AI, Patronus AI, Surge AI, Halluminate, Turing, Scorecard, and Snorkel AI.

### Current status

OpenEnv is in active development, and the scope is expanding from environments toward the full agentic-RL loop. Where it stands today:

| Dimension | Where it stands |
| :---- | :---- |
| **License** | BSD-3-Clause; contributions accepted under the same terms. |
| **Repository** | Hosted at huggingface/OpenEnv, with 2,300+ GitHub stars |
| **Environments on the Hub** | 4,200+ environments published to the Hugging Face OpenEnv Hub, spanning coding REPLs, browser control, and games such as Wordle and Sudoku. |
| **In flight** | [RFC 005](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md) (agentic harnesses)<br>[RFC 006](https://github.com/huggingface/OpenEnv/blob/main/rfcs/006-agentic-rl-harness-interception.md) (agentic RL through harness interception)<br>[RFC 007](https://github.com/huggingface/OpenEnv/pull/727) (environment datasets)<br>[RFC 008](https://github.com/huggingface/OpenEnv/issues/778) (environment auto-validation) |

The project is now pivoting from making environments easy to publish and share toward making them validated and trainable, which is what RFCs 005-008 addresses.

## Goals

***The primary goal of OpenEnv is to grow the open source agentic RL community by addressing the challenges of its inherent diversity.***

Frontier labs train a model and its harness to work like hand in glove; trainers expect environments in their own format; eval suites ship their own runners; and the agentic harnesses that do the work each have their own way of running. The result is that an environment, a harness, or a trajectory built for one stack rarely works with another, and progress gets trapped in silos. OpenEnv exists to fix this.
In recent releases, OpenEnv has become an interoperability layer for RL environments. Its job is to standardize how environments are published, deployed, and consumed by agents.

***Our long-term vision is for OpenEnv to become a library to interface between harness, environment, and trainer, which works on any model.***

Getting there means staying a thin, neutral set of contracts that every stack plugs into, so work written once runs anywhere, and being the protocol others build on rather than a competitor to them. We have articulated this vision in more detail on our announcement [blogpost](https://huggingface.co/blog/openenv-agentic-rl).

At the highest level, OpenEnv has two goals:

* **Interoperability as the product.** Success is measured by how much of the agentic-RL loop speaks OpenEnv (trainers, harnesses, evaluators, reward libraries, and environments), not by features or surface area it captures from the ecosystem.

* **Network effects across the loop.** Every environment, harness, or trainer that adopts the standard makes it more valuable to the whole ecosystem.

In short, “Built for OpenEnv” becomes the default expectation for new environments so that researchers and beyond reach for OpenEnv first, because it is where the environments and agents are.

## The Who (and Why)

| Persona | JTBD | What do we want? |
| :---- | :---- | :---- |
| **Environment authors** *e.g. the long tail on the Hub* | Publish an environment once and have it run everywhere, and know it is genuinely useful for training. | •  Publish to the standard •  Flag gaps in the spec •  Contribute environments that pass the validation bar |
| **Model authors / labs** *e.g. Meta, Reflection, and others* | Train and judge models on diverse, realistic tasks without bespoke integration per environment or harness. | •  Standardize on the interface •  Contribute realistic environments •  Report what the spec is missing |
| **Trainer & RL-framework builders** *e.g. TitanRL, TRL, prime-rl, SkyRL, Unsloth* | A stable contract for environments and trajectories so they do not re-implement adapters per source. | •  Ship first-class OpenEnv support •  Co-design the interface via RFCs |
| **Harness & agent builders** *e.g. OpenClaw, Hermes, Claude Code, Goose* | Wrap a harness once and have it work for training, evaluation, and production, with a standard trajectory format. | •  Adopt the harness wrapping pattern (RFC 005) •  Emit the standard trajectory and event schema •  Keep their control loop, gain portability |
| **Reward & eval library authors** *e.g. verifiers, Archipelago, APEX, τ-bench, SkillsBench / BenchFlow* | Define rewards and rubrics once and have them inspectable and comparable across environments. | •  Keep reward logic in their libraries •  Expose reward components through the rubric tree (RFC 004) •  Build evals on top of OpenEnv |
| **Infra & compute providers** *e.g. Modal, neoclouds* | A predictable, container-native deployment and serving target. | •  Support the packaging and transports •  Host environments and agents at scale |
| **Researchers** *e.g. Stanford, UC Berkeley* | Low-friction access to many environments and reproducible baselines. | •  Use and cite OpenEnv •  Contribute environments and feedback |

## Governance

***OpenEnv will adopt a lightweight governance structure built on mutual trust and shared ownership, aiming for impact over bureaucracy.***

The committee gathers on a biweekly cadence to discuss project roadmap, technical direction, RFCs, allocation of work (code reviews, integrations, etc.), GTM plan, partnerships, and community events.

Technical authority is earned by individuals on the merit of their PRs, commits, and reviews. In the early stages of the project, Ben Burtenshaw acts as the BDFL of the project in technical matters, ensuring that we move fast and don’t deviate from our collective vision.

By default, everyone should be encouraged to work asynchronously on [GitHub](https://github.com/huggingface/OpenEnv) (issues, PRs, RFCs) and [Slack](https://app.slack.com/client/T06LG3JF5DL/C09FNMB4BK3), and only use the recurring meeting as a means to unblock work. If you don’t have access to Slack channel, request an invitation from [benjamin.burtenshaw@huggingface.co](mailto:benjamin.burtenshaw@huggingface.co).

### The RFC process and the contribution ladder

Work flows through a simple loop: an RFC proposes a change, discussion happens on the issue and in syncs, an accepted RFC becomes a PR, and the PR is backed by advocacy to drive usage. The same loop is how a contributor grows into ownership:

* **Rung 1, give feedback:** review and comment on others' RFCs. Every member is expected to be active here.

* **Rung 2, (co)own an RFC:** propose and lead a change end to end, which is the template for how a partner owns a component while the community gives feedback.

The current RFC slate (numbers and owners evolve in the open):

| RFC | Topic | Status |
| :---- | :---- | :---- |
| [001](https://github.com/huggingface/OpenEnv/blob/main/rfcs/001-abstractions.md) | Basic abstractions and boundaries | Shipped |
| [002](https://github.com/huggingface/OpenEnv/blob/main/rfcs/002-env-spec.md) | Framework spec and environment-computed rewards | Shipped |
| [003](https://github.com/huggingface/OpenEnv/blob/main/rfcs/003-mcp-support.md) | MCP support | Shipped |
| [004](https://github.com/huggingface/OpenEnv/blob/main/rfcs/004-rubrics.md) | Rubric system (rewards stay in the environment) | Shipped |
| [005](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md) | Agentic harness integration (OpenClaw first) | In PR |
| [006](https://github.com/huggingface/OpenEnv/blob/main/rfcs/006-agentic-rl-harness-interception.md) | Agentic RL through harness interception: the token capture contract | In review |
| [007](https://github.com/huggingface/OpenEnv/pull/727) | Environment datasets | In PR |
| [008](https://github.com/huggingface/OpenEnv/issues/778) | Environment auto-validation and trainability bar | In review |

### Terms and Conditions

**Patents**. OpenEnv is built on open collaboration. By contributing code, participating in the Technical Committee, or using OpenEnv, participants agree not to assert patent claims against any person or entity for implementations of the OpenEnv standard, provided that such commitment shall not apply to patent claims made as a counterclaim, cross-claim or affirmative defense in direct response to patent claims asserted for implementation of the OpenEnv standard.

**Controls.** All participants are responsible for ensuring that their use, distribution, hosting, or transfer of OpenEnv technologies complies with applicable export control, sanctions, and trade laws.

**Antitrust.** Participants must comply with applicable antitrust and competition laws. Technical Committee discussions must be limited to technical specifications, interoperability, and project governance, and must not include competitively sensitive matters.

**Independent Contractors.** Participation in OpenEnv does not create a partnership, joint venture, agency relationship, or other legal entity among participants. Each participant acts solely as an independent contractor.

**Governing law and dispute resolution.** These terms are governed by the laws of the State of New York, without regard to its conflict-of-laws rules. If there is a dispute, the parties will first try in good faith to resolve it outside of court. If that doesn’t work, any legal case must be filed only in the state or federal courts in New York City, USA.

## Appendix A: Additional references

* [Building the open agent ecosystem together: introducing OpenEnv (2025)](https://huggingface.co/blog/openenv)

* [The open source community is backing OpenEnv for agentic RL (2026)](https://huggingface.co/blog/openenv-agentic-rl)

* [OpenEnv in practice: evaluating tool-using agents](https://huggingface.co/blog/openenv-turing)

* [OpenEnv Org on HF](https://huggingface.co/openenv)
