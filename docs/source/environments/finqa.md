<!-- openenv-source: finqa_env -->
# FinQA Environment

A financial question-answering environment for RL training. Evaluates LLMs on their ability to answer complex financial questions using tool calls on SEC 10-K filing data.

Based on [FinQABenchmark](https://github.com/snorkel-ai/FinQABenchmark) from Snorkel AI.

## Overview

FinQA tests an agent's ability to:
- Explore available financial tables for a company
- Query table metadata and execute SQL queries
- Perform calculations on extracted data
- Submit final answers to financial questions

**Dataset**: 290 questions from SEC 10-K filings across multiple companies (Alphabet, Amazon, Apple, AT&T, etc.)

**Reward**: Binary (1.0 for correct answer, 0.0 for incorrect) using fuzzy numerical matching with 1% tolerance.

> **Note**: This dataset is for evaluation only. Do not train on it.

## Quick Start

### Using Docker

```bash
# Build the image (from OpenEnv repo root)
docker build -t finqa-env:latest -f envs/finqa_env/server/Dockerfile .

# Run the server
docker run -p 8000:8000 finqa-env:latest

# To run evaluation script (example model gpt-5)
API_BASE_URL=https://api.openai.com/v1 API_KEY=$OPENAI_API_KEY MODEL=gpt-5 python examples/finqa_inference.py
```

### Local Development

```bash
# Install dependencies
uv pip install pandas

# Download data from HuggingFace
cd envs/finqa_env
./download_data.sh
```

### Using the Client

The client uses the MCP protocol and is async by default:

```python
import asyncio
from envs.finqa_env import FinQAEnv, CallToolAction

async def main():
    async with FinQAEnv(base_url="http://localhost:8000") as env:
        # Reset to get a question
        result = await env.reset()
        question = result.observation.metadata["question"]
        company = result.observation.metadata["company"]
        print(f"Question: {question}")
        print(f"Company: {company}")

        # Discover available tools
        tools = await env.list_tools()
        print([t.name for t in tools])

        # Use tools via call_tool (convenience method)
        result = await env.call_tool("get_descriptions", company_name=company)
        print(f"Available tables: {result}")

        # Or use step() with CallToolAction for full observation access
        step_result = await env.step(CallToolAction(
            tool_name="sql_query",
            arguments={
                "company_name": "alphabet",
                "table_name": "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock",
                # Name the columns you need (see get_table_info); SELECT * is rejected
                "query": "SELECT <columns> FROM data WHERE year = '2022'"
            }
        ))
        print(f"Done: {step_result.done}, Reward: {step_result.reward}")

        # Submit answer
        result = await env.call_tool("submit_answer", answer="6.118")

asyncio.run(main())
```

## Available Tools

Tools are auto-discovered via MCP. Use `await env.list_tools()` to see all available tools at runtime.

| Tool | Description | Arguments |
|------|-------------|-----------|
| `get_descriptions` | Get list of available table names for a company | `company_name: str` |
| `get_table_info` | Get table metadata (columns, dtypes, unique values) | `company_name: str, table_name: str` |
| `sql_query` | Execute SQL query on a table (requires filters) | `company_name: str, table_name: str, query: str` |
| `submit_answer` | Submit final answer (ends episode) | `answer: str` |

Once the episode has ended (an answer was submitted or `FINQA_MAX_STEPS` was reached), further `step()` calls (the WebSocket session) do not run tools or score answers. They return the final observation again (`done=True`, same reward) until the next `reset()`.

### Tool Constraints

- **sql_query**: Must include filters (`WHERE`, `HAVING`, etc.). `SELECT *` is not allowed.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FINQA_DATA_PATH` | `/app/env/data` | Path to data directory |
| `FINQA_MAX_STEPS` | `50` | Maximum tool calls per episode |
| `FINQA_TASK` | `finqa` | Task name |

## Tasks

The environment implements the [Task API](https://huggingface.co/docs/openenv/guides/task-api). The dataset is for evaluation only, so it publishes one split, `test`, holding every row of `benchmark_questions/finqa.csv` in file order. Each task spec is `{"id": "finqa-test-<i>", "split": "test", "index": <i>, "group": <company>}`. There is no `difficulty` key, and a spec carries neither the question nor the answer.

The server exposes `/finqa_env/splits`, `/finqa_env/tasks`, `/finqa_env/num_tasks`, `/finqa_env/task` and `/finqa_env/task_range`. `reset(split="test", index=i)` loads row `i`, over `POST /reset` or the WebSocket `reset` message, and `/schema` publishes `split` and `index` under `reset`:

```bash
curl -s localhost:8000/finqa_env/num_tasks -H 'Content-Type: application/json' -d '{"split": "test"}'
curl -s localhost:8000/reset -H 'Content-Type: application/json' -d '{"split": "test", "index": 3}'
```

- A reset without `index` keeps the seeded shuffle: `seed` sets the order of the unindexed resets, and an indexed reset does not move it.
- `seed` does not change an indexed question, since FinQA has no seeded variants. For pass@k, run several samples per task at a temperature above 0.
- An unknown split raises `ValueError`, with or without `index`, and an index outside the split raises `IndexError` (`400` on `/finqa_env/task`).
- The routes need `openenv-core` 0.4.1 or later on the server, and the reset schema on `/schema` needs 0.11.0 or later.

## Reward Computation

Rewards use fuzzy numerical matching:

- Extracts numbers from `\boxed{...}` format
- Handles percentages, fractions, and decimals
- 1% relative tolerance or 0.01 absolute tolerance
- Returns `1.0` for correct, `0.0` for incorrect

## Local Development

```bash
# From OpenEnv repo root
cd envs/finqa_env

# Run server locally
FINQA_DATA_PATH=./data uvicorn server.app:app --reload --port 8000

# Test with curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
```

## Integration with RL Frameworks

### TRL (GRPO)

```python
import asyncio
from trl import GRPOTrainer
from envs.finqa_env import FinQAEnv

async def rollout_func(prompts, trainer):
    async with FinQAEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        # Your agent logic here using await env.call_tool(...)
        return {"reward": obs.reward, "completion": completion}

trainer = GRPOTrainer(
    model=model,
    rollout_func=rollout_func,
    ...
)
```

## Project Structure

```
finqa_env/
├── __init__.py           # Exports FinQAEnv, CallToolAction, ListToolsAction
├── models.py             # FinQAState and tool name constants
├── client.py             # MCP client (subclasses MCPToolClient)
├── pyproject.toml        # Dependencies
├── README.md             # This file
├── data/                 # Benchmark data (run download_data.sh)
│   ├── benchmark_questions/
│   │   └── finqa.csv
│   └── input_companies/
│       └── [company folders]
├── download_data.sh      # Downloads data from HuggingFace
└── server/
    ├── __init__.py
    ├── finqa_environment.py  # MCPEnvironment subclass with @mcp.tool decorators
    ├── tools.py              # Tool implementations
    ├── rewards.py            # Reward computation
    ├── app.py                # FastAPI server
    └── Dockerfile
```

## References

- [HuggingFace Dataset](https://huggingface.co/datasets/snorkelai/agent-finance-reasoning)
- [Leaderboard](https://leaderboard.snorkel.ai/category/snorkelfinance)
