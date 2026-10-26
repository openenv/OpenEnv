# OpenEnv RFCs

This directory contains Requests for Comments (RFCs) for the OpenEnv framework. RFCs are used to propose, discuss, and document significant changes, new features, or architectural decisions.

## RFC Process

### 1. Proposing an RFC

When you want to propose a significant change or feature:

1. **Create a new RFC file** in this directory named `NNN-short-title.md` where `NNN` is the next available RFC number based on open RFCs (e.g., `002-render-api.md`)
2. **Use the RFC template** structure outlined below
3. **Submit for review** by submitting a PR
4. **Iterate based on feedback** until consensus is reached

### 2. RFC Lifecycle

RFCs go through several stages:

- **In Review**: Open for feedback and discussion
- **Accepted**: Consensus reached, ready for implementation
- **Implemented**: Changes have been merged
- **Rejected**: Proposal was not accepted (document why)
- **Superseded**: Replaced by a newer RFC

### 3. What Requires an RFC?

You should write an RFC for:

- **New core APIs** (e.g., new methods on `Environment` or `EnvClient`)
- **Breaking changes** to existing interfaces
- **Major architectural decisions** (e.g., communication protocol changes)
- **New abstractions** or design patterns

You generally don't need an RFC for:

- Bug fixes
- Documentation improvements
- Minor refactoring
- New example environments (unless they introduce new patterns)

## RFC Template

Each RFC should include the following sections:

### Required Sections

1. **Header**
   ```markdown
   # RFC: [Title]
   
   **Status**: [Draft|In Review|Accepted|Implemented|Rejected|Superseded]
   **Created**: [Date]
   **Authors**: [@username1, @username2]
   **RFC ID**: [NNN]
   ```

2. **Summary**
   - Brief 1-2 paragraph overview of the proposal
   - Should be clear enough for someone to understand the essence without reading the full RFC

3. **Motivation**
   - **Problem Statement**: What problem are you solving?
   - **Goals**: What are you trying to achieve?
   - Clear explanation of why this change is needed

4. **Design**
   - **Architecture Overview**: High-level view (diagrams encouraged)
   - **Core Abstractions**: Key interfaces, classes, or concepts
   - **Key Design Decisions**: For each significant decision:
     - **Chosen Approach**: What you're proposing
     - **Rationale**: Why this approach
     - **Trade-offs**: What are the pros/cons if any

5. **Examples**
   - Code examples showing how the feature would be used
   - Should demonstrate common use cases
   - Include both client and server perspectives where relevant

## Current RFCs

### Core Abstractions & Design
- [000-project-phases.md](./000-project-phases.md) - Design Principles and Broad Roadmap
- [001-abstractions.md](./001-abstractions.md) - Basic Abstractions (Environment, Agent, State)
- [002-env-spec.md](./002-env-spec.md) - Framework Spec for Agent Execution Environments

### MCP Integration
- [003-mcp-support.md](./003-mcp-support.md) - MCP Support: Traditional Tool Calling

### Reward & Evaluation
- [004-rubrics.md](./004-rubrics.md) - Rubric System for Reward Computation

### Agentic Harnesses
- [005-agentic-harnesses.md](./005-agentic-harnesses.md) - Agentic Harness Integration (OpenClaw, Claude Code, etc.)
- [006-agentic-rl-harness-interception.md](./006-agentic-rl-harness-interception.md) - Agentic RL through Harness Interception: the token capture contract
- [012-harbor-capture-providers.md](./012-harbor-capture-providers.md) - Harbor evaluation/training purpose, provider fidelity, exact export and live-session ownership

### Validation
- [008-environment-auto-validation.md](./008-environment-auto-validation.md) - Environment Auto-Validation: local `openenv validate` levels 1-3 + the contracts for operator-run hubs

### World Modeling
- [010-echo-env-token-world-model.md](./010-echo-env-token-world-model.md) - Env-token World Modeling (ECHO): trajectory token-role masks + an optimizer world-loss seam

### Discovery & Distribution
- [011-ard-catalog-discovery.md](./011-ard-catalog-discovery.md) - Versioned environment metadata, ARD interchange, and a read-only producer-consumer path

## Questions?

For questions about the RFC process, reach out to the core team or open a discussion in the project repository.
