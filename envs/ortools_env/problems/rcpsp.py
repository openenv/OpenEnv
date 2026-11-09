# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Resource-constrained project scheduling (PS | prec | Cmax).

Instances follow the PSPLIB design parameters (Kolisch & Sprecher 1997): 4 renewable resources, durations in
[1, 10], a resource factor (share of resources each activity uses) and a resource strength (how tight capacities are
between the largest single demand and the peak demand of the earliest-start schedule). Low resource strength makes
instances hard; j120 with RS = 0.1 is where most PSPLIB instances remain open. PSPLIB `.sm` files load through the
OR-Tools parser, and the reference model is the CP-SAT formulation of `or-tools/examples/python/rcpsp_sat.py`.
"""

import random

NAME = "rcpsp"

SIZES = {"j10": 10, "j30": 30, "j60": 60, "j120": 120, "j300": 300}
DEFAULT_SIZE = "j120"

STATEMENT = """\
Resource-constrained project scheduling (minimize makespan).

There are `len(instance["tasks"])` activities and `len(instance["capacities"])` renewable resources. Activity `i` has
an integer `duration`, a list `demands` (units of each resource it uses while running) and a list `successors`:
every successor may only start once activity `i` has finished. At every time `t`, the total demand of the activities
running at `t` (those with `start <= t < start + duration`) must not exceed the capacity of each resource. Activities
cannot be interrupted.

Solution: `{"starts": starts}` where `starts[i]` is the non-negative integer start time of activity `i`.
Objective: the makespan, i.e. the latest completion time over all activities. Lower is better.
"""


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    rng = random.Random(seed)
    num_tasks = SIZES[size]
    num_resources = 4
    resource_factor = rng.choice([0.5, 0.75, 1.0])
    resource_strength = rng.choice([0.1, 0.2, 0.3, 0.5])

    # Random precedence DAG in topological order: every activity after the first few gets at least one predecessor,
    # and every activity before the last few gets at least one successor (network complexity ~1.5-2.1 as in PSPLIB).
    successors = [set() for _ in range(num_tasks)]
    num_starts = max(3, num_tasks // 10)
    for i in range(num_starts, num_tasks):
        successors[rng.randrange(0, i)].add(i)
    for i in range(num_tasks - num_starts):
        while len(successors[i]) < rng.randint(1, 3):
            successors[i].add(rng.randrange(i + 1, num_tasks))

    durations = [rng.randint(1, 10) for _ in range(num_tasks)]
    demands = []
    for _ in range(num_tasks):
        used = [r for r in range(num_resources) if rng.random() < resource_factor] or [
            rng.randrange(num_resources)
        ]
        demands.append(
            [rng.randint(1, 10) if r in used else 0 for r in range(num_resources)]
        )

    # Capacities from the resource strength: K = Kmin + RS * (Kmax - Kmin), where Kmin is the largest single demand
    # and Kmax the peak usage of the earliest-start (resource-unconstrained) schedule.
    earliest = [0] * num_tasks
    for i in range(num_tasks):
        for s in successors[i]:
            earliest[s] = max(earliest[s], earliest[i] + durations[i])
    horizon = max(earliest[i] + durations[i] for i in range(num_tasks))
    capacities = []
    for r in range(num_resources):
        usage = [0] * horizon
        for i in range(num_tasks):
            for t in range(earliest[i], earliest[i] + durations[i]):
                usage[t] += demands[i][r]
        k_min = max(demand[r] for demand in demands)
        capacities.append(k_min + round(resource_strength * (max(usage) - k_min)))

    tasks = [
        {
            "duration": durations[i],
            "demands": demands[i],
            "successors": sorted(successors[i]),
        }
        for i in range(num_tasks)
    ]
    return {"capacities": capacities, "tasks": tasks}


def load(path: str) -> dict:
    """Load a single-mode PSPLIB `.sm` file (dropping the dummy source and sink activities)."""
    from ortools.scheduling.python import rcpsp

    parser = rcpsp.RcpspParser()
    if not parser.parse_file(path):
        raise ValueError(f"Could not parse {path}.")
    problem = parser.problem()
    if any(len(task.recipes) != 1 for task in problem.tasks):
        raise ValueError("Only single-mode RCPSP instances are supported.")
    real = range(1, len(problem.tasks) - 1)
    num_resources = len(problem.resources)
    tasks = []
    for t in real:
        recipe = problem.tasks[t].recipes[0]
        demands = [0] * num_resources
        for demand, resource in zip(recipe.demands, recipe.resources):
            demands[resource] = demand
        successors = [s - 1 for s in problem.tasks[t].successors if s in real]
        tasks.append(
            {"duration": recipe.duration, "demands": demands, "successors": successors}
        )
    return {
        "capacities": [resource.max_capacity for resource in problem.resources],
        "tasks": tasks,
    }


def evaluate(instance: dict, solution: dict) -> int:
    """Return the makespan of `solution`; raise `ValueError` if it is malformed or infeasible."""
    tasks = instance["tasks"]
    starts = solution["starts"]
    if len(starts) != len(tasks):
        raise ValueError(f"`starts` must have {len(tasks)} entries, got {len(starts)}.")
    for i, start in enumerate(starts):
        if not isinstance(start, int) or start < 0:
            raise ValueError(
                f"Start of activity {i} must be a non-negative integer, got {start!r}."
            )
    for i, task in enumerate(tasks):
        for s in task["successors"]:
            if starts[s] < starts[i] + task["duration"]:
                raise ValueError(
                    f"Activity {s} starts before its predecessor {i} ends."
                )
    # Sweep over start/end events; ends sort before starts at the same time since intervals are half-open.
    events = []
    for i, task in enumerate(tasks):
        events.append((starts[i] + task["duration"], 0, i))
        events.append((starts[i], 1, i))
    events.sort()
    usage = [0] * len(instance["capacities"])
    for time, is_start, i in events:
        sign = 1 if is_start else -1
        for r, demand in enumerate(tasks[i]["demands"]):
            usage[r] += sign * demand
            if usage[r] > instance["capacities"][r]:
                raise ValueError(f"Resource {r} is over capacity at time {time}.")
    return max(starts[i] + task["duration"] for i, task in enumerate(tasks))


BASELINE = (
    "a serial schedule: activities in increasing index order among those whose predecessors are scheduled, each at "
    "the earliest time its predecessors and the resources allow"
)


def baseline(instance: dict) -> int:
    """Makespan of the naive serial schedule described in `BASELINE`."""
    tasks, capacities = instance["tasks"], instance["capacities"]
    predecessors = [[] for _ in tasks]
    for i, task in enumerate(tasks):
        for s in task["successors"]:
            predecessors[s].append(i)
    horizon = sum(task["duration"] for task in tasks) + 1
    usage = [[0] * horizon for _ in capacities]
    ends = [None] * len(tasks)
    while None in ends:
        i = next(
            i
            for i in range(len(tasks))
            if ends[i] is None and all(ends[p] is not None for p in predecessors[i])
        )
        duration, demands = tasks[i]["duration"], tasks[i]["demands"]
        start = max((ends[p] for p in predecessors[i]), default=0)
        while any(
            usage[r][t] + demands[r] > capacities[r]
            for r in range(len(capacities))
            for t in range(start, start + duration)
        ):
            start += 1
        for r in range(len(capacities)):
            for t in range(start, start + duration):
                usage[r][t] += demands[r]
        ends[i] = start + duration
    return max(ends)


def reference(
    instance: dict, time_limit: float, num_workers: int = 8
) -> tuple[dict, int, int]:
    """Solve with CP-SAT; return `(solution, objective, lower_bound)`."""
    from ortools.sat.python import cp_model

    tasks = instance["tasks"]
    horizon = sum(task["duration"] for task in tasks)
    model = cp_model.CpModel()
    starts, ends, intervals = [], [], []
    for i, task in enumerate(tasks):
        start = model.new_int_var(0, horizon, f"start_{i}")
        end = model.new_int_var(0, horizon, f"end_{i}")
        intervals.append(
            model.new_interval_var(start, task["duration"], end, f"interval_{i}")
        )
        starts.append(start)
        ends.append(end)
    for i, task in enumerate(tasks):
        for s in task["successors"]:
            model.add(starts[s] >= ends[i])
    for r, capacity in enumerate(instance["capacities"]):
        model.add_cumulative(
            intervals, [task["demands"][r] for task in tasks], capacity
        )
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution within {time_limit}s.")
    return (
        {"starts": [solver.value(s) for s in starts]},
        int(solver.objective_value),
        int(solver.best_objective_bound),
    )
