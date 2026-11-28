# OpenEnv: Agentic Execution Environments

  
    An end-to-end framework for creating, deploying and using isolated execution environments for agentic RL, with a simple Gymnasium-style API.
  

  
    
      One API for every environment
      reset(), step() and state(), sync or async, over a WebSocket.
    
    
      Isolated and deployable
      Each environment is a Docker image that runs locally, on a cloud sandbox, or as a Hugging Face Space.
    
    
      40+ environments
      Games, coding sandboxes, browsers, finance, simulators and more, in the catalog.
    
    
      Train with your framework
      TRL, Unsloth, SkyRL, ART, Oumi, torchforge, Miles and more. See Training with OpenEnv.
    
    
      Train real coding agents
      Harbor runs Claude Code, Codex, OpenCode and other harnesses, and captures their tokens for RL.
    
    
      Serve tools to agents
      MCP environments expose their tools over /mcp in production mode.
    
  

## Where to start

1. **[Getting Started](getting-started)**: install OpenEnv, connect to an environment and run your first step.
2. **[Train an agent](guides/training)**: pick a way to train and a framework. [Harbor](environments/harbor) captures coding agents that run their own loop, for training.
3. **[Build your own environment](guides/first-environment)**, then [deploy it](getting_started/environment-builder) to Hugging Face Spaces.
4. **[Explore environments](environments)**: browse the catalog.

The [tutorials](tutorials/index) include a 5-part Getting Started series that needs no GPU, and the [Concepts](guides/concepts) pages explain how the pieces fit.

## Contributing

OpenEnv is openly governed by a technical committee that coordinates project direction, RFCs and releases through the public [GitHub repository](https://github.com/openenv/OpenEnv). The [charter](https://github.com/openenv/OpenEnv/blob/main/GOVERNANCE.md) explains how it works, and the [README](https://github.com/openenv/OpenEnv#community-support--acknowledgments) lists the committee members and supporters. Bug reports, feature requests and new environments are welcome as issues or pull requests, see [Contributing](contributing). For the changelog, see [GitHub Releases](https://github.com/openenv/OpenEnv/releases).

> [!NOTE]
> OpenEnv is in early development, so APIs may still change. Bug fixes are welcome. For larger changes, open or claim an issue first so the change can be discussed.

