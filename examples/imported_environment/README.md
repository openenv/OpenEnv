# Imported Environment Usage

This example imports the tiny ORS/OpenReward-style source in `source/` and then
uses the generated OpenEnv wrapper.

```bash
uv run python examples/imported_environment/use_imported_environment.py
```

To keep the generated wrapper for inspection:

```bash
uv run python examples/imported_environment/use_imported_environment.py --output-dir /tmp/openenv-import-demo
```

The script runs `openenv import`, imports the generated
`ImportedOrsDemoEnvironment`, calls `reset()`, lists the generated MCP-style
tools, and submits an answer with `CallToolAction`.
