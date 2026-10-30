<!-- openenv-source: sql_optim_env -->
# SQL Query Optimization Environment

An **execution-grounded** environment for training agents to write fast SQL. The
agent is given a slow query plus its schema and must return a rewrite. Reward is
**not** a keyword heuristic: the environment runs both the original and the
optimized query against a real temporary-file-backed **DuckDB** database seeded with
realistic synthetic data (10k users, 500k orders, 1k products, 1M events), then
scores the agent on measured speedup and result-correctness.

Five tasks scale from basic anti-patterns to expert window-function audits.

## Quick Start

```python
from sql_optim_env import SQLOptimAction, SQLOptimEnv

with SQLOptimEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="task_1_basic_antipatterns").observation
    print(obs.sql_query)          # the slow query to fix
    print(obs.schema_info)        # tables, row counts

    result = env.step(
        SQLOptimAction(
            suggestions=[
                {"issue_type": "select_star", "line": 1,
                 "description": "SELECT * reads every column", "severity": "high",
                 "fix": "project only the needed columns"},
            ],
            optimized_query="SELECT id FROM orders WHERE customer_id = 5000",
            summary="Dropped SELECT * and the non-sargable CAST.",
            estimated_improvement="~10x faster",
            approved=False,
        )
    )
    print(result.reward, result.done)
    print(result.metadata["reward_breakdown"])
    print(result.metadata["execution"]["speedup"])   # real DuckDB speedup
```

Async usage mirrors the other environments:

```python
async with SQLOptimEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task_id="task_1_basic_antipatterns")
```

## Action

| Field | Type | Meaning |
|-------|------|---------|
| `suggestions` | `list[dict]` | Issues found: `issue_type`, `line`, `description`, `severity`, `fix` |
| `optimized_query` | `str` | The rewritten SQL — executed for real |
| `summary` | `str` | Overall analysis |
| `estimated_improvement` | `str` | Agent's own speedup estimate |
| `approved` | `bool` | `True` if the query is already optimal |

## Observation

`sql_query`, `schema_info`, `task_description`, `difficulty`, `step_count`,
`max_steps`, `issues_found_so_far`, and `last_execution` (the previous step's
real timing comparison, so the agent can refine its rewrite). The step reward is
on `reward`; the composite `reward_breakdown`, `feedback`, and full `execution`
comparison are under `metadata`.

## Reward

Composite score in `[0, 1]`, computed inside the environment:

| Component | Weight | Source |
|-----------|--------|--------|
| Execution speedup | 35% | real DuckDB timing ratio |
| Result correctness | 20% | do both queries return identical rows? |
| Issue detection | 25% | flagged issues vs. ground truth |
| Approval correctness | 8% | correctly judged the query bad/good |
| Summary quality | 7% | thoroughness of the written analysis |
| Severity labels | 5% | severities present |

## Tasks

Five tasks (`task_1_basic_antipatterns` … `task_5_*`) covering `SELECT *`,
non-sargable predicates, functions on filter columns, join ordering, and window
functions, spanning `easy` → `expert`.

## Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Resource budgets

`openenv.yaml` declares 1024 MiB memory and 1024 MiB writable storage for
Docker validation. DuckDB uses a **256 MiB** `memory_limit` and a **128 MiB**
`max_temp_directory_size`, for both dataset construction and query execution.
Explicit `MiB` units avoid confusion with DuckDB's decimal `GB`.

The `docker-local` validation provider enforces the memory limit with swap
disabled and splits writable storage between `/tmp` tmpfs and `/dev/shm`
(up to 64 MiB). The temporary database and spill files in `/tmp` therefore
also consume container memory. A regression checks the real seeded database,
spill allowance and shared-memory allowance plus DuckDB's memory cap fit in
half the declared memory budget, leaving at least 512 MiB for runtime overhead.
This is budget headroom, not a guarantee that every agent query fits.

DuckDB's [memory limit covers its buffer manager](https://duckdb.org/docs/current/operations_manual/limits),
not the Python heap or every DuckDB allocation. Its spill cap excludes the
base database. The result-fetch row cap is not a byte cap, and the interrupt
deadline cannot stop Python object conversion. Untrusted workloads still need
external process/container memory and time enforcement; the resource-isolation
policy remains a maintainer review gate. The Dockerfile and a direct `uvicorn`
launch do not enforce the YAML budgets themselves: deployments must configure
their runtime limits explicitly.

## Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_sql_optim_environment.py -v
```
