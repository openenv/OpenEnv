# Async vs Sync

OpenEnv supports both asynchronous and synchronous usage patterns.

## When to Use Each

| Pattern | Best For | Performance |
|---------|----------|-------------|
| **Async** | Production, multiple environments, high throughput | ⚡ Best |
| **Sync** | Scripts, notebooks, quick experiments | 🐢 Good enough |

## Sync Usage

For simple scripts and notebooks, use the `.sync()` wrapper:

```python
from openenv import AutoEnv

env = AutoEnv.from_env("echo")

with env.sync() as client:
    result = client.reset()
    result = client.step(action)
```

## Async Usage (Recommended)

For production and parallel environments, use async directly:

```python
import asyncio
from openenv import AutoEnv

async def main():
    env = AutoEnv.from_env("echo")

    async with env as client:
        result = await client.reset()
        result = await client.step(action)

asyncio.run(main())
```

## Parallel Environments

Run multiple environments concurrently:

```python
import asyncio
from openenv import AutoEnv

async def run_episode(env_name: str):
    env = AutoEnv.from_env(env_name)
    async with env as client:
        result = await client.reset()
        # ... run episode
        return result

async def main():
    # Run 4 environments in parallel
    results = await asyncio.gather(
        run_episode("echo"),
        run_episode("echo"),
        run_episode("echo"),
        run_episode("echo"),
    )

asyncio.run(main())
```

## Next Steps

- [Training with OpenEnv](training.md) - Using async with RL training
- [Auto-Discovery](auto-discovery.md) - Discover and load environments
