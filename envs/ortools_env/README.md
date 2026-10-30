---
title: OR-Tools Environment Server
emoji: 🏭
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OR-Tools Environment

Train and evaluate LLMs on **designing solutions and algorithms for hard operations research problems**, with a
verifiable reward: the objective of the agent's solution against an OR-Tools reference.

The setup follows [LLMs Can Design Near-Optimal OR Algorithms](https://huggingface.co/papers/2608.27296). Each episode
is one query: the agent reads a problem statement, uses a Python sandbox under a compute budget, and returns one
deliverable.

| Level | The agent gets | It submits | Scored on |
|---|---|---|---|
| 1 | one instance | a solution (`submit_solution`) | that instance |
| 2 | the problem class and an example instance | code defining `solve(instance)` (`submit_algorithm`) | unseen instances of the same distribution |

## Problems

The problem classes were picked where exact methods still struggle at scale, so a strong agent has room to match or
beat the reference instead of saturating at the optimum. Each class has a generator that follows the standard benchmark
design, a loader for the standard file format, a pure-Python verifier, and an OR-Tools reference solver taken from the
[OR-Tools examples](https://github.com/google/or-tools/tree/stable/examples/python).

| `problem` | Problem | Generator | Sizes | Benchmark files | Reference |
|---|---|---|---|---|---|
| `jobshop` | Job shop, minimize makespan | Taillard (1993), reproduces `ta01`-`ta80` from their seeds | `6x6` … `100x20` | OR-Library / JSPLIB | CP-SAT (optimum and bound) |
| `rcpsp` | Resource-constrained project scheduling | PSPLIB design (resource factor, resource strength) | `j10` … `j300` | PSPLIB `.sm` | CP-SAT (optimum and bound) |
| `cvrp` | Capacitated vehicle routing | Uchoa et al. (2017) "X" design | `n20` … `n1000` | CVRPLIB `.vrp` | Routing solver, guided local search |
| `cvrptw` | Vehicle routing with time windows | Solomon (1987) / Gehring & Homberger classes R, C, RC, series 1 and 2 | `n25` … `n1000` | Solomon / Gehring-Homberger | Routing solver, guided local search |
| `tardiness` | Single-machine total weighted tardiness | OR-Library (Potts & Van Wassenhove) | `n10` … `n500` | OR-Library `wtN.txt` | CP-SAT (optimum and bound) |
| `rostering` | Nurse rostering with hard and soft work rules | Rules of `shift_scheduling_sat.py`, random demands and requests | `4x1` … `80x8` (employees x weeks) | — | CP-SAT (optimum and bound) |
| `timetabling` | Curriculum-based course timetabling | ITC-2007 track 3 rules, generated around a hidden feasible timetable | `c10` … `c150` (courses) | ITC-2007 `.ctt` | CP-SAT (optimum and bound) |

The smallest size of each problem is for smoke tests. Every generated instance is reproducible from `(seed, size)`,
and the generators randomize the structural parameters (resource strength, due-date tightness, depot and customer
placement, demand distribution) per seed.

## Reward

All objectives are minimized. The reward places the agent's objective between a naive solution and the OR-Tools
reference:

```
reward = (naive - objective) / (naive - reference)
```

1.0 matches the reference and above 1.0 beats it; 0 is no better than the naive solution, and worse solutions score
below 0 down to -1.0, which is also the score of an infeasible solution, a crash or a timeout. At level 2 the reward is
the mean over the evaluation instances. The episode ends at the first submission, whose observation carries
`done=True` and the reward.

The naive solution ignores the instance data, so a policy cannot collect reward by submitting any feasible answer
(a plain `reference / objective` ratio gives a random job order on 10 jobs about 0.5). Each problem describes its
naive solution in `BASELINE` and computes it with `baseline(instance)`:

| `problem` | Naive solution |
|---|---|
| `jobshop` | operations scheduled round by round in job order, each as early as possible |
| `rcpsp` | serial schedule in activity index order |
| `cvrp` | customers in index order, a new route whenever the next one does not fit |
| `cvrptw` | customers in index order, a new route whenever the next one breaks the capacity or a time window |
| `tardiness` | jobs in index order |
| `rostering` | first feasible roster CP-SAT finds when all penalties are ignored |
| `timetabling` | first feasible timetable CP-SAT finds when all penalties are ignored |

References are best computed offline with a long time limit:

```bash
PYTHONPATH=src:envs python -m ortools_env.build_dataset --problem rcpsp --sizes j60 j120 --level 1 \
    --num-rows 256 --reference-time-limit 60 --out rcpsp_l1.jsonl
```

Each row holds the `reset` arguments of one episode (`problem`, `size`, `level`, `instance_seed`, `reference` and
`baseline`, or `eval_seeds`, `eval_references` and `eval_baselines`). A reference missing from `reset` is computed at
submission time with `reference_time_limit_s`, and so is a missing baseline.

## Quick start

```bash
PYTHONPATH=src:envs uvicorn ortools_env.server.app:app --port 8000
```

```python
from openenv.core.env_server.mcp_types import CallToolAction

from ortools_env import ORToolsEnv

with ORToolsEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(problem="jobshop", size="15x15", level=1, seed=0)
    print(result.observation.metadata["prompt"])

    print(env.call_tool("run_python", code="print(instance['num_jobs'])"))

    # `call_tool` returns only the tool text; use `step` to read the reward of the submission.
    result = env.step(
        CallToolAction(
            tool_name="submit_solution", arguments={"solution": {"starts": [...]}}
        )
    )
    print(result.reward, result.done)
    print(env.state())  # ORToolsState: objective, reference, compute used, ...
```

## Tools

| Tool | Description |
|---|---|
| `run_python(code)` | Runs a script in the sandbox with the instance bound to `instance` (and in `instance.json`); files persist between calls. Counts against the compute budget. |
| `check_solution(solution)` | Level 1: returns the objective of a candidate solution and the naive objective, or why it is infeasible. Counts as a Python call. |
| `test_algorithm(code)` | Level 2: runs `solve` on the example instance as at submission and returns its objective and the naive objective, or why it failed. Counts against the Python calls and the compute budget. |
| `submit_solution(solution)` | Level 1: submits the solution as a JSON object. Ends the episode. |
| `submit_algorithm(code)` | Level 2: submits code defining `solve(instance)`, run in a fresh process per evaluation instance with `solve_time_limit_s`. Ends the episode. |

## Reset arguments

| Argument | Default | Description |
|---|---|---|
| `problem` | `"jobshop"` | `jobshop`, `rcpsp`, `cvrp`, `cvrptw`, `tardiness`, `rostering` or `timetabling` |
| `size` | the problem's default | a key of the problem's `SIZES` |
| `level` | `1` | `1` (solution) or `2` (algorithm) |
| `instance` / `instance_seed` | generated from `seed` | an explicit instance, e.g. read with the problem's `load`, or the seed of a generated one |
| `reference`, `baseline`, `lower_bound` | computed at submission | precomputed reference, naive objective and bound (level 1) |
| `eval_seeds`, `eval_references`, `eval_baselines` | `num_eval_instances` seeds from `seed` | hidden evaluation instances, their references and naive objectives (level 2) |
| `allow_ortools` | `False` | whether the agent's code may import `ortools` |
| `compute_budget_s`, `max_python_calls`, `python_timeout_s` | `600`, `50`, `120` | sandbox budget, shared by `run_python` and the checks |
| `solve_time_limit_s` | `30` | per-instance time limit of `solve` (level 2) |
| `reference_time_limit_s` | `10` | OR-Tools time limit for references computed at submission |

## Sandbox

Code runs in a subprocess with a wall-clock limit and, on Linux, a 4 GB per-process address-space limit. Cleanup
kills the original process group, but detached children can survive. This does not provide aggregate resource
containment. Hostile submissions remain blocked on per-run Linux PID/cgroup isolation; see
[Sandbox status and deployment holds](SECURITY.md).

Keeping `ortools` (unless `allow_ortools=True`) and the environment's own modules out of reach prevents direct imports
of the reference solver and verifier. This is only one part of the required isolation boundary.

In the container, the code runs as an unprivileged `sandbox` user with its own interpreter (numpy and scipy only), and
the environment source and the server's packages are not readable by that user; with `allow_ortools=True` it uses a
second interpreter with `ortools`. This is configured by `SANDBOX_USER`, `SANDBOX_PYTHON` and `SANDBOX_ORTOOLS_PYTHON`
in `server/Dockerfile`. Outside the container (`SANDBOX_USER` unset), the code runs with the server's interpreter and
an in-process import blocker that it can remove, which is fine for development but does not keep rewards honest
against a policy that looks for it.

## Harbor export

The same episodes can be exported as [Harbor](https://harborframework.com) tasks to measure frontier agents
(claude-code, codex, ...) with their own harnesses, for example through the `harbor_env` environment:

```bash
PYTHONPATH=src:envs python -m ortools_env.export_harbor --dataset rcpsp_l1.jsonl --out tasks/
```

Each task gets the instance in `/app/instance.json` and asks for `/app/solution.json` (level 1) or `/app/solve.py`
(level 2). The verifier, the problem module and the hidden evaluation instances are in `tests/`, which Harbor only
uploads after the agent has finished.
