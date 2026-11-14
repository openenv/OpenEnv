# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Nurse rostering: assign each employee one shift (off, morning, afternoon, night) per day, meeting the cover demand
and the hard work rules, and minimizing soft-rule penalties.

The rules are those of `or-tools/examples/python/shift_scheduling_sat.py`: bounds on consecutive runs of a shift,
weekly bounds on shift counts, penalized or forbidden shift transitions and a penalty per employee above the cover
demand. Employee requests are unwanted assignments with a positive penalty, so the objective is non-negative. Demands,
requests and penalty weights are drawn per seed, and the reference is the CP-SAT model of the example.
"""

import random

NAME = "rostering"

# Employees x weeks.
SIZES = {
    "4x1": (4, 1),
    "8x3": (8, 3),
    "20x4": (20, 4),
    "40x6": (40, 6),
    "80x8": (80, 8),
}
DEFAULT_SIZE = "20x4"

STATEMENT = """\
Nurse rostering (minimize total penalty).

There are `num_employees` employees and `num_days = 7 * num_weeks` days (day 0 is a Monday). Each employee works
exactly one shift per day: 0 = off, 1 = morning, 2 = afternoon, 3 = night.

Hard rules (a schedule breaking any of them is infeasible):
- Cover: on day `d`, at least `cover[d][s - 1]` employees work shift `s`, for `s` in 1, 2, 3.
- Runs: for each `[shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost]` in `sequence_rules`, every maximal
  run of consecutive days on which an employee works `shift` has a length in `[hard_min, hard_max]`.
- Weekly counts: for each `[shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost]` in `weekly_rules`, each
  employee works `shift` on between `hard_min` and `hard_max` days of every week (days `7w` to `7w + 6`).
- Transitions: for each `[previous, next, cost]` in `transitions` with `cost == 0`, no employee works `previous` on a day
  and `next` on the following day.

Penalties (the objective is their sum):
- Runs: a run of length `L < soft_min` costs `min_cost * (soft_min - L)`; `L > soft_max` costs `max_cost * (L - soft_max)`.
- Weekly counts: a count `c < soft_min` costs `min_cost * (soft_min - c)`; `c > soft_max` costs `max_cost * (c - soft_max)`.
- Transitions: each occurrence of a transition with `cost > 0` costs `cost`.
- Over-cover: each employee above `cover[d][s - 1]` on shift `s` costs `excess_cover_costs[s - 1]`.
- Requests: for each `[employee, shift, day, cost]` in `requests`, assigning that shift on that day costs `cost`.

Solution: `{"schedule": schedule}` where `schedule[e][d]` is the shift (0-3) of employee `e` on day `d`.
Objective: the total penalty. Lower is better.
"""


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    rng = random.Random(seed)
    num_employees, num_weeks = SIZES[size]
    num_days = 7 * num_weeks
    # Between 1 and 3 days off per week, so on average about 5/7 of the staff works; demand most of that.
    cover = []
    for _ in range(num_days):
        total = max(3, round(num_employees * rng.uniform(0.5, 0.62)))
        night = max(1, round(total * rng.uniform(0.15, 0.3)))
        morning = max(1, round((total - night) * rng.uniform(0.4, 0.6)))
        cover.append([morning, max(1, total - night - morning), night])
    requests = []
    for _ in range(num_employees * num_weeks):
        requests.append(
            [
                rng.randrange(num_employees),
                rng.randint(1, 3),
                rng.randrange(num_days),
                rng.randint(1, 4),
            ]
        )
    return {
        "num_employees": num_employees,
        "num_weeks": num_weeks,
        "cover": cover,
        "sequence_rules": [
            # One or two consecutive days off (hard).
            [0, 1, 1, 0, 2, 2, 0],
            # Two or three consecutive nights; one or four are allowed but penalized.
            [3, 1, 2, rng.randint(10, 20), 3, 4, rng.randint(3, 6)],
        ],
        "weekly_rules": [
            # One to three days off per week, two preferred.
            [0, 1, 2, rng.randint(4, 8), 2, 3, rng.randint(2, 5)],
            # At most four nights per week, at least one preferred.
            [3, 0, 1, rng.randint(2, 4), 4, 4, 0],
        ],
        "transitions": [
            # Afternoon to night is penalized; night to morning is forbidden.
            [2, 3, rng.randint(2, 5)],
            [3, 1, 0],
        ],
        "excess_cover_costs": [rng.randint(1, 3), rng.randint(1, 3), rng.randint(3, 6)],
        "requests": requests,
    }


def _runs(values: list, shift: int) -> list[int]:
    lengths, length = [], 0
    for value in values + [None]:
        if value == shift:
            length += 1
        elif length:
            lengths.append(length)
            length = 0
    return lengths


def evaluate(instance: dict, solution: dict) -> int:
    """Return the total penalty of `solution`; raise `ValueError` if it is malformed or breaks a hard rule."""
    num_employees, num_days = instance["num_employees"], 7 * instance["num_weeks"]
    schedule = solution["schedule"]
    if len(schedule) != num_employees or any(len(row) != num_days for row in schedule):
        raise ValueError(
            f"`schedule` must be {num_employees} rows of {num_days} shifts."
        )
    if any(shift not in (0, 1, 2, 3) for row in schedule for shift in row):
        raise ValueError("Shifts must be integers in 0-3.")
    penalty = 0
    for e, row in enumerate(schedule):
        for (
            shift,
            hard_min,
            soft_min,
            min_cost,
            soft_max,
            hard_max,
            max_cost,
        ) in instance["sequence_rules"]:
            for length in _runs(row, shift):
                if not hard_min <= length <= hard_max:
                    raise ValueError(
                        f"Employee {e} has a run of {length} days on shift {shift}."
                    )
                penalty += min_cost * max(0, soft_min - length) + max_cost * max(
                    0, length - soft_max
                )
        for (
            shift,
            hard_min,
            soft_min,
            min_cost,
            soft_max,
            hard_max,
            max_cost,
        ) in instance["weekly_rules"]:
            for w in range(instance["num_weeks"]):
                count = row[7 * w : 7 * w + 7].count(shift)
                if not hard_min <= count <= hard_max:
                    raise ValueError(
                        f"Employee {e} works shift {shift} on {count} days of week {w}."
                    )
                penalty += min_cost * max(0, soft_min - count) + max_cost * max(
                    0, count - soft_max
                )
        for previous, following, cost in instance["transitions"]:
            for d in range(num_days - 1):
                if row[d] == previous and row[d + 1] == following:
                    if cost == 0:
                        raise ValueError(
                            f"Employee {e} goes from shift {previous} to {following} on day {d + 1}."
                        )
                    penalty += cost
    for d in range(num_days):
        for s in (1, 2, 3):
            worked = sum(row[d] == s for row in schedule)
            demand = instance["cover"][d][s - 1]
            if worked < demand:
                raise ValueError(
                    f"Day {d}, shift {s}: {worked} employees for a demand of {demand}."
                )
            penalty += instance["excess_cover_costs"][s - 1] * (worked - demand)
    for e, shift, d, cost in instance["requests"]:
        if schedule[e][d] == shift:
            penalty += cost
    return penalty


BASELINE = "the first feasible roster that CP-SAT finds (single worker) when all penalties are ignored"


def baseline(instance: dict) -> int:
    """Total penalty of the naive roster described in `BASELINE`."""
    return _solve(instance, time_limit=60.0, num_workers=1, minimize=False)[1]


def reference(
    instance: dict, time_limit: float, num_workers: int = 8
) -> tuple[dict, int, int]:
    """Solve with CP-SAT; return `(solution, objective, lower_bound)`."""
    return _solve(instance, time_limit, num_workers, minimize=True)


def _solve(
    instance: dict, time_limit: float, num_workers: int, minimize: bool
) -> tuple[dict, int, int]:
    from ortools.sat.python import cp_model

    num_employees, num_days = instance["num_employees"], 7 * instance["num_weeks"]
    model = cp_model.CpModel()
    work = {
        (e, s, d): model.new_bool_var(f"work_{e}_{s}_{d}")
        for e in range(num_employees)
        for s in range(4)
        for d in range(num_days)
    }
    terms = []
    for e in range(num_employees):
        for d in range(num_days):
            model.add_exactly_one(work[e, s, d] for s in range(4))

    # Runs: forbid runs shorter than hard_min or longer than hard_max, penalize those outside the soft bounds. A run
    # of exactly `length` days from `start` is the conjunction of `works` in the span and its negated borders.
    def run_literal(works, start, length):
        span = [works[start + i] for i in range(length)]
        if start > 0:
            span.append(~works[start - 1])
        if start + length < len(works):
            span.append(~works[start + length])
        return span

    for shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost in instance[
        "sequence_rules"
    ]:
        for e in range(num_employees):
            works = [work[e, shift, d] for d in range(num_days)]
            for length in range(1, hard_max + 1):
                cost = min_cost * max(0, soft_min - length) + max_cost * max(
                    0, length - soft_max
                )
                if length >= hard_min and cost == 0:
                    continue
                for start in range(num_days - length + 1):
                    span = run_literal(works, start, length)
                    if length < hard_min:
                        model.add_bool_or([~lit for lit in span])
                    else:
                        is_run = model.new_bool_var("")
                        model.add_bool_or([~lit for lit in span] + [is_run])
                        terms.append(cost * is_run)
            for start in range(num_days - hard_max):
                model.add_bool_or([~works[start + i] for i in range(hard_max + 1)])

    for shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost in instance[
        "weekly_rules"
    ]:
        for e in range(num_employees):
            for w in range(instance["num_weeks"]):
                count = model.new_int_var(hard_min, hard_max, "")
                model.add(count == sum(work[e, shift, 7 * w + d] for d in range(7)))
                under = model.new_int_var(0, 7, "")
                over = model.new_int_var(0, 7, "")
                model.add(under >= soft_min - count)
                model.add(over >= count - soft_max)
                terms += [min_cost * under, max_cost * over]

    for previous, following, cost in instance["transitions"]:
        for e in range(num_employees):
            for d in range(num_days - 1):
                clause = [~work[e, previous, d], ~work[e, following, d + 1]]
                if cost == 0:
                    model.add_bool_or(clause)
                else:
                    happens = model.new_bool_var("")
                    model.add_bool_or(clause + [happens])
                    terms.append(cost * happens)

    for d in range(num_days):
        for s in (1, 2, 3):
            worked = sum(work[e, s, d] for e in range(num_employees))
            demand = instance["cover"][d][s - 1]
            model.add(worked >= demand)
            terms.append(instance["excess_cover_costs"][s - 1] * (worked - demand))

    for e, shift, d, cost in instance["requests"]:
        terms.append(cost * work[e, shift, d])
    if minimize:
        model.minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution within {time_limit}s.")
    schedule = [
        [
            next(s for s in range(4) if solver.value(work[e, s, d]))
            for d in range(num_days)
        ]
        for e in range(num_employees)
    ]
    solution = {"schedule": schedule}
    # Re-evaluate: penalty literals are only forced up, so the solver objective can overcount by slack.
    return (
        solution,
        evaluate(instance, solution),
        max(0, int(solver.best_objective_bound)),
    )
