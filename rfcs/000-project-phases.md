# RFC: Design Principles and Broad Roadmap

**Status**: In Review
**Created**: 10/17/2025
**Amended**: October 1, 2026
**Authors**: @Darktex, @pankit-eng, @jspisak, @zkwentz
**RFC ID:** 000

## Amendment History

**October 1, 2026**: Removals of APIs, CLI flags and environments go through a deprecation first, as described in `.claude/docs/INVARIANTS.md` (Deprecations).

**November 12, 2025**: Added design principles, target audience, and updated roadmap to reference RFCs 005-007.

## Design Principles

These principles guide every decision in OpenEnv:

1. **Minimize deltas across lifecycle**: Training → Evals → Production should use identical tool interfaces
2. **Minimize human-agent divergence**: Tools that work for humans should work for agents
3. **Be hands-on**: Provide ready-to-use implementations, not just specs
4. **Design for LLMs**: Optimize for context efficiency, in-distribution behavior, and token costs

## Target Audience

- **Environment builders**: Get reach across projects without custom adapters
- **Model builders**: Access massive inventory of environments and tools
- **Researchers**: Reproducible setups with versioned tools, rewards, and evals
- **Infrastructure engineers**: Clear contracts enabling backend optimization

## Summary
Before jumping into the actual concrete proposals, this RFC introduces how we are going to approach this problem space, what problems we want to solve with this project, and how we plan to prioritize and solve them systematically.

We recommend starting here to get a mental model of what's in here already, what is missing, and what's next.

## Goals
This project aims at standardizing environments for both training and evaluation. In the training space, this means also standardizing reward pipelines, while in the eval space this means helping with reproducibility where a model can be shipped with a complete set of agentic evals that can be easily run by others.

### The problem with abstraction boundaries
Ideally, we would draw a boundary between environments and everything else (orchestration, resource allocation, RPCs, etc.). We will try to do this as much as possible, but we will have to create additional interfaces so that if folks want to cross this boundary, they can. This will likely be necessary for things like:
- Reward pipelines that call reward models (which will very likely need to RPC to GPU machines)
- Agentic evals like Tau where the eval itself involves two agents interacting with one another (and sending many RPCs)
- Container provider interfaces to support different deployment targets (Docker, Kubernetes, cloud providers, etc.)

## Phases
We plan to build things incrementally, from inside out, adding and expanding only whenever necessary.

We will group development from now till version 1.0 into three phases.

In the **first phase** of this project, we will focus **exclusively** on the narrowest definition of environments, without even worrying about rewards nor evals. Instead, the focus in this phase (and in the RFCs you find in this directory) is going to be on:
1. Establishing a convention on what is an environment and where we draw the "environment" box (RFC 001).
2. Landing the basics of _sandboxing_, _versioning_, _binary distribution_, _dependency management_ (RFC 002).
3. Nailing our tools support through MCP (Model Context Protocol) integration:
   - RFC 003: Traditional tool calling (ListToolsAction, CallToolAction)
   - RFC 004: CodeAct support (agents write executable code)
   - RFC 005: MCP as the universal interface (policy and rationale)
4. Establishing tool registry and distribution patterns via Hugging Face Hub (upcoming RFC).
5. Enabling production performance simulation to minimize training-production delta (RFC 006).
6. Exploring MCP protocol interception for observability (RFC 007).
7. Exploring RPC communication patterns beyond HTTP for long-running sessions (particularly for interpreted languages like Python, Bash, Ruby, etc.). Coming in an upcoming RFC.

We will conclude this phase with version 0.3.

**Note on versioning**: We're using 0.3 increments for each phase to leave room for minor releases and patches within each phase. This gives us flexibility to ship iteratively while working toward each phase's goals.

In the **second phase** of this project, we will add rewards. Reward pipelines are crucial to get right since that is one of the main levels that ML engineers have to improve the model (much more than the general algorithm) so all the questions around versioning and deps that we tackled in the first phase will start being useful right away. We will introduce a way to RPC outside the Environment boundary: we expect this to require active discussion.

We will conclude this phase with version 0.6.

In the **third phase** of this project, we will add evals, starting from the simplest ones (simple evals that can be scored by a script), then extending to evals using LLM judges and then going into fully agentic evals.

We will conclude this phase with version 0.9.

Finally, we'll work on the finishing touches to get us from 0.9 to 1.0!

Throughout every phase, we will author concrete environments and we encourage everyone to do the same as developing together with use cases will give us line of sight into where we fall short and what needs to be changed/fixed.

Detailed success criteria and progress tracking for each version will be managed through GitHub milestones and issues.

## Our approach towards breaking changes
Until Version 1.0, we plan to move fast and will accept breaking changes if needed. They are documented in the release notes, and removing an API, CLI flag or environment goes through a deprecation first: it keeps working with a `FutureWarning` that names the release that removes it and what to use instead, for at least two minor releases (see Deprecations in `.claude/docs/INVARIANTS.md`). We will have stronger guarantees in place after 1.0, to be determined down the road (probably together with the other "finishing touches" between 0.9 and 1.0...).

While APIs may not be 100% stable, we expect that any breaking changes should be relatively minor and can be fixed by LLM coders.

## Today: Version 0.1
We are unveiling this project early on to invite contributions from the get-go!

The official date for our 0.1 launch is October 22, 2025. If you are reading this sooner, spoiler alert!
