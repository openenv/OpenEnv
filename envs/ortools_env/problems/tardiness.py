# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single-machine total weighted tardiness (1 || sum wj Tj), strongly NP-hard.

Instances follow the OR-Library generator (Potts & Van Wassenhove 1985; Crauwels, Potts & Van Wassenhove 1998):
processing times in [1, 100], weights in [1, 10] and due dates drawn from
[P (1 - TF - RDD / 2), P (1 - TF + RDD / 2)] with P the total processing time, tardiness factor TF and relative range
of due dates RDD. High TF with low RDD gives the hardest instances. The CP-SAT lower bound is weak on this problem,
so the gap to the bound overstates how far the reference is from optimal. OR-Library `wtN.txt` files load directly. The
reference model is a CP-SAT disjunctive model in the style of
`or-tools/examples/python/single_machine_scheduling_with_setup_release_due_dates_sat.py`.
"""

import random

NAME = "tardiness"

SIZES = {"n10": 10, "n40": 40, "n100": 100, "n200": 200, "n500": 500}
DEFAULT_SIZE = "n100"

STATEMENT = """\
Single-machine total weighted tardiness (minimize total weighted tardiness).

There are `n = len(instance["processing_times"])` jobs to process one at a time, without interruption or idle time,
on a single machine starting at time 0. Job `j` has an integer processing time `instance["processing_times"][j]`,
weight `instance["weights"][j]` and due date `instance["due_dates"][j]`. If job `j` completes at time `C_j`, its
tardiness is `T_j = max(0, C_j - due_dates[j])`.

Solution: `{"sequence": sequence}`, a permutation of `range(n)` giving the processing order.
Objective: the total weighted tardiness `sum(weights[j] * T_j)`. Lower is better.
"""


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    rng = random.Random(seed)
    n = SIZES[size]
    # OR-Library also uses TF in {0.2, 0.4} and RDD up to 1.0, but those are often solved by a sort with zero
    # tardiness, which gives no training signal; keep the hard corner.
    tardiness_factor = rng.choice([0.6, 0.8, 1.0])
    due_date_range = rng.choice([0.2, 0.4, 0.6])
    processing_times = [rng.randint(1, 100) for _ in range(n)]
    total = sum(processing_times)
    low = round(total * (1 - tardiness_factor - due_date_range / 2))
    high = round(total * (1 - tardiness_factor + due_date_range / 2))
    return {
        "processing_times": processing_times,
        "weights": [rng.randint(1, 10) for _ in range(n)],
        "due_dates": [rng.randint(max(0, low), max(0, high)) for _ in range(n)],
    }


def load(path: str, index: int = 0, n: int | None = None) -> dict:
    """Load instance `index` from an OR-Library `wtN.txt` file (`n` defaults to the number in the file name)."""
    if n is None:
        n = int("".join(c for c in path.rsplit("/", 1)[-1] if c.isdigit()))
    with open(path) as f:
        values = [int(x) for x in f.read().split()]
    block = values[3 * n * index : 3 * n * (index + 1)]
    return {
        "processing_times": block[:n],
        "weights": block[n : 2 * n],
        "due_dates": block[2 * n :],
    }


def evaluate(instance: dict, solution: dict) -> int:
    """Return the total weighted tardiness of `solution`; raise `ValueError` if it is not a permutation."""
    sequence = solution["sequence"]
    n = len(instance["processing_times"])
    if sorted(sequence) != list(range(n)):
        raise ValueError(f"`sequence` must be a permutation of range({n}).")
    time = 0
    total = 0
    for j in sequence:
        time += instance["processing_times"][j]
        total += instance["weights"][j] * max(0, time - instance["due_dates"][j])
    return total


BASELINE = "process the jobs in index order"


def baseline(instance: dict) -> int:
    """Total weighted tardiness of the naive sequence described in `BASELINE`."""
    return evaluate(
        instance, {"sequence": list(range(len(instance["processing_times"])))}
    )


def reference(
    instance: dict, time_limit: float, num_workers: int = 8
) -> tuple[dict, int, int]:
    """Solve with CP-SAT; return `(solution, objective, lower_bound)`."""
    from ortools.sat.python import cp_model

    processing_times, weights, due_dates = (
        instance["processing_times"],
        instance["weights"],
        instance["due_dates"],
    )
    n = len(processing_times)
    horizon = sum(processing_times)
    model = cp_model.CpModel()
    starts, intervals, weighted_tardiness = [], [], []
    for j in range(n):
        start = model.new_int_var(0, horizon - processing_times[j], f"start_{j}")
        end = model.new_int_var(processing_times[j], horizon, f"end_{j}")
        intervals.append(
            model.new_interval_var(start, processing_times[j], end, f"interval_{j}")
        )
        tardiness = model.new_int_var(
            0, max(0, horizon - due_dates[j]), f"tardiness_{j}"
        )
        model.add_max_equality(tardiness, [0, end - due_dates[j]])
        starts.append(start)
        weighted_tardiness.append(weights[j] * tardiness)
    model.add_no_overlap(intervals)
    model.minimize(sum(weighted_tardiness))
    # Weighted shortest processing time first is a decent warm start.
    time = 0
    for j in sorted(range(n), key=lambda j: processing_times[j] / weights[j]):
        model.add_hint(starts[j], time)
        time += processing_times[j]

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution within {time_limit}s.")
    sequence = sorted(range(n), key=lambda j: solver.value(starts[j]))
    solution = {"sequence": sequence}
    # Re-evaluate: the sequence closes any idle time the solver may have left, which can only lower the objective.
    return solution, evaluate(instance, solution), int(solver.best_objective_bound)
