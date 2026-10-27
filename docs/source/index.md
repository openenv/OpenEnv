# OpenEnv: Agentic Execution Environments

<div class="hero">
  <p class="hero__subtitle">
    An end-to-end framework for creating, deploying and using isolated execution environments for agentic RL, with a simple Gymnasium-style API.
  </p>
</div>

<div class="mt-6">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">One API for every environment</div>
      <p><code>reset()</code>, <code>step()</code> and <code>state()</code>, sync or async, over a WebSocket.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Isolated and deployable</div>
      <p>Each environment is a Docker image that runs locally, on a cloud sandbox, or as a Hugging Face Space.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">40+ environments</div>
      <p>Games, coding sandboxes, browsers, finance, simulators and more, in the <a href="environments">catalog</a>.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Train with your framework</div>
      <p>TRL, Unsloth, SkyRL, ART, Oumi, torchforge, Miles and more. See <a href="guides/training">Training with OpenEnv</a>.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Train real coding agents</div>
      <p><a href="environments/harbor">Harbor</a> runs Claude Code, Codex, OpenCode and other harnesses, and captures their tokens for RL.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Serve tools to agents</div>
      <p>MCP environments expose their tools over <code>/mcp</code> in production mode.</p>
    </div>
  </div>
</div>

## Where to start

1. **[Getting Started](getting-started)**: install OpenEnv, connect to an environment and run your first step.
2. **[Train an agent](guides/training)**: pick a way to train and a framework. [Harbor](environments/harbor) captures coding agents that run their own loop, for training.
3. **[Build your own environment](guides/first-environment)**, then [deploy it](getting_started/environment-builder) to Hugging Face Spaces.
4. **[Explore environments](environments)**: browse the catalog.

The [tutorials](tutorials/index) include a 5-part Getting Started series that needs no GPU, and the [Concepts](guides/concepts) pages explain how the pieces fit.

## Contributing

OpenEnv is openly governed by a technical committee that coordinates project direction, RFCs and releases through the public [GitHub repository](https://github.com/huggingface/OpenEnv). The [charter](https://github.com/huggingface/OpenEnv/blob/main/GOVERNANCE.md) explains how it works, and the [README](https://github.com/huggingface/OpenEnv#community-support--acknowledgments) lists the committee members and supporters. Bug reports, feature requests and new environments are welcome as issues or pull requests, see [Contributing](contributing). For the changelog, see [GitHub Releases](https://github.com/huggingface/OpenEnv/releases).

> [!NOTE]
> OpenEnv is in early development, so APIs may still change. Bug fixes are welcome. For larger changes, open or claim an issue first so the change can be discussed.
