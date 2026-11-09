# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Job shop scheduling (J || Cmax).

Instances follow Taillard's generator (Taillard 1993, "Benchmarks for basic scheduling problems"), so the classic
`ta01`-`ta80` instances are reproduced exactly from their published seeds, and fresh instances of the same
distribution come from any other seed. The reference model is the CP-SAT formulation of
`or-tools/examples/python/jobshop_ft06_sat.py`.
"""

import random

NAME = "jobshop"

# Jobs x machines. 6x6 is for smoke tests; the others are Taillard sizes: 15x15 is mostly closed, while 20x20,
# 30x15 and 30x20 still contain open instances.
SIZES = {
    "6x6": (6, 6),
    "15x15": (15, 15),
    "20x20": (20, 20),
    "30x15": (30, 15),
    "30x20": (30, 20),
    "50x20": (50, 20),
    "100x20": (100, 20),
}
DEFAULT_SIZE = "30x20"

STATEMENT = """\
Job shop scheduling (minimize makespan).

There are `num_jobs` jobs and `num_machines` machines. Job `j` is a fixed sequence of operations
`instance["jobs"][j] = [[machine, duration], ...]`: operation `k + 1` of a job cannot start before operation `k` of the
same job has finished. Every machine processes at most one operation at a time, and operations cannot be interrupted.
All durations are positive integers.

Solution: `{"starts": starts}` where `starts[j][k]` is the integer start time of operation `k` of job `j`.
Objective: the makespan, i.e. the latest completion time over all operations. Lower is better.
"""


def _taillard_unif(seed: int, low: int, high: int) -> tuple[int, int]:
    # Lehmer generator from Taillard (1993): returns (next_seed, value uniform in [low, high]).
    k = seed // 127773
    seed = 16807 * (seed % 127773) - k * 2836
    if seed < 0:
        seed += 2147483647
    return seed, low + int(seed / 2147483647 * (high - low + 1))


def taillard(
    num_jobs: int, num_machines: int, time_seed: int, machine_seed: int
) -> dict:
    """Taillard job shop instance from its two published seeds."""
    durations = []
    for _ in range(num_jobs):
        row = []
        for _ in range(num_machines):
            time_seed, value = _taillard_unif(time_seed, 1, 99)
            row.append(value)
        durations.append(row)
    jobs = []
    for j in range(num_jobs):
        machines = list(range(num_machines))
        for k in range(num_machines):
            machine_seed, swap = _taillard_unif(machine_seed, k, num_machines - 1)
            machines[k], machines[swap] = machines[swap], machines[k]
        jobs.append([[machines[k], durations[j][k]] for k in range(num_machines)])
    return {"num_jobs": num_jobs, "num_machines": num_machines, "jobs": jobs}


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    num_jobs, num_machines = SIZES[size]
    rng = random.Random(seed)
    return taillard(
        num_jobs, num_machines, rng.randint(1, 2**31 - 2), rng.randint(1, 2**31 - 2)
    )


def load(path: str) -> dict:
    """Load an instance in the OR-Library / JSPLIB format (`n m` header, then `machine duration` pairs per job)."""
    with open(path) as f:
        rows = [line.split() for line in f]
    # Skip the free-text header of OR-Library files: the data starts at the first line holding exactly two integers.
    start = next(
        i
        for i, row in enumerate(rows)
        if len(row) == 2 and all(x.isdigit() for x in row)
    )
    num_jobs, num_machines = int(rows[start][0]), int(rows[start][1])
    jobs = []
    for row in rows[start + 1 : start + 1 + num_jobs]:
        values = [int(x) for x in row]
        jobs.append([[values[2 * k], values[2 * k + 1]] for k in range(num_machines)])
    return {"num_jobs": num_jobs, "num_machines": num_machines, "jobs": jobs}


def evaluate(instance: dict, solution: dict) -> int:
    """Return the makespan of `solution`; raise `ValueError` if it is malformed or infeasible."""
    jobs = instance["jobs"]
    starts = solution["starts"]
    if len(starts) != len(jobs) or any(
        len(s) != len(job) for s, job in zip(starts, jobs)
    ):
        raise ValueError(
            "`starts` must have one row per job and one entry per operation."
        )
    by_machine = [[] for _ in range(instance["num_machines"])]
    makespan = 0
    for j, job in enumerate(jobs):
        previous_end = 0
        for k, (machine, duration) in enumerate(job):
            start = starts[j][k]
            if not isinstance(start, int) or start < 0:
                raise ValueError(
                    f"Start of job {j} operation {k} must be a non-negative integer, got {start!r}."
                )
            if start < previous_end:
                raise ValueError(
                    f"Job {j} operation {k} starts at {start} before its predecessor ends at {previous_end}."
                )
            previous_end = start + duration
            by_machine[machine].append((start, previous_end, j, k))
        makespan = max(makespan, previous_end)
    for machine, intervals in enumerate(by_machine):
        intervals.sort()
        for (_, end, j1, k1), (start, _, j2, k2) in zip(intervals, intervals[1:]):
            if start < end:
                raise ValueError(
                    f"Machine {machine}: job {j1} op {k1} and job {j2} op {k2} overlap."
                )
    return makespan


BASELINE = (
    "schedule the operations round by round (operation 0 of every job in job order, then operation 1, ...), each "
    "as early as possible"
)


def baseline(instance: dict) -> int:
    """Makespan of the naive round-robin schedule described in `BASELINE`."""
    job_end = [0] * instance["num_jobs"]
    machine_end = [0] * instance["num_machines"]
    for k in range(max(len(job) for job in instance["jobs"])):
        for j, job in enumerate(instance["jobs"]):
            if k < len(job):
                machine, duration = job[k]
                job_end[j] = machine_end[machine] = (
                    max(job_end[j], machine_end[machine]) + duration
                )
    return max(job_end)


def reference(
    instance: dict, time_limit: float, num_workers: int = 8
) -> tuple[dict, int, int]:
    """Solve with CP-SAT; return `(solution, objective, lower_bound)`."""
    from ortools.sat.python import cp_model

    jobs = instance["jobs"]
    horizon = sum(duration for job in jobs for _, duration in job)
    model = cp_model.CpModel()
    starts, ends = [], []
    machine_to_intervals = [[] for _ in range(instance["num_machines"])]
    for j, job in enumerate(jobs):
        job_starts, job_ends = [], []
        for k, (machine, duration) in enumerate(job):
            start = model.new_int_var(0, horizon, f"start_{j}_{k}")
            end = model.new_int_var(0, horizon, f"end_{j}_{k}")
            machine_to_intervals[machine].append(
                model.new_interval_var(start, duration, end, f"interval_{j}_{k}")
            )
            if job_ends:
                model.add(start >= job_ends[-1])
            job_starts.append(start)
            job_ends.append(end)
        starts.append(job_starts)
        ends.append(job_ends)
    for intervals in machine_to_intervals:
        model.add_no_overlap(intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [job_ends[-1] for job_ends in ends])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution within {time_limit}s.")
    solution = {
        "starts": [[solver.value(s) for s in job_starts] for job_starts in starts]
    }
    return solution, int(solver.objective_value), int(solver.best_objective_bound)
