# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Curriculum-based course timetabling: place every lecture of every course in a period and a room, without conflicts,
minimizing soft-constraint penalties.

The rules and penalty weights are those of track 3 of the International Timetabling Competition 2007 (Di Gaspero,
McCollum & Schaerf), which extends the teachers, rooms and restricted slots of `ortools/scheduling/course_scheduling`
with curricula and soft constraints. Instances are generated around a hidden feasible timetable, so every instance has
a solution: curricula only group courses whose planted lectures do not overlap, and unavailable periods avoid the
planted ones. ITC-2007 `.ctt` files load directly. The reference is a CP-SAT model.
"""

import math
import random

NAME = "timetabling"

# Number of courses.
SIZES = {"c10": 10, "c30": 30, "c60": 60, "c100": 100, "c150": 150}
DEFAULT_SIZE = "c60"

# Penalty per student above the room capacity, per missing working day, per isolated lecture and per extra room.
CAPACITY_COST, MIN_DAYS_COST, ISOLATED_COST, ROOM_COST = 1, 5, 2, 1

STATEMENT = """\
Curriculum-based course timetabling (minimize total penalty).

There are `num_days` days of `periods_per_day` periods each; period `p` is slot `p % periods_per_day` of day
`p // periods_per_day`. Course `c` is taught by teacher `teachers[c]`, has `lectures[c]` lectures, `students[c]`
students and should spread its lectures over at least `min_days[c]` days. Room `r` seats `room_capacities[r]`
students. Each curriculum in `curricula` is a list of courses taken by the same students.

Hard rules (a timetable breaking any of them is infeasible):
- Every lecture of every course is placed in exactly one period and one room.
- A room holds at most one lecture per period.
- A course has at most one lecture per period.
- Courses of the same teacher, or of the same curriculum, are never in the same period.
- Course `c` is never in period `p` for each `[c, p]` in `unavailable`.

Penalties (the objective is their sum):
- Room capacity: each lecture costs 1 per student above the capacity of its room.
- Minimum working days: each course costs 5 per day below `min_days[c]` among the days it has lectures.
- Curriculum compactness: each lecture of a curriculum with no lecture of the same curriculum in the slot just
  before or just after it on the same day costs 2 (per curriculum it belongs to).
- Room stability: each course costs 1 per distinct room it uses beyond the first.

Solution: `{"assignments": assignments}`, one `[course, period, room]` triple per lecture, for example
`{"assignments": [[0, 3, 1], [0, 7, 1], [1, 3, 0]]}`.
Objective: the total penalty. Lower is better.
"""


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    rng = random.Random(seed)
    num_courses = SIZES[size]
    num_days, periods_per_day = 5, rng.choice([4, 5, 6])
    num_periods = num_days * periods_per_day
    lectures = [rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5, 6]) for _ in range(num_courses)]
    min_days = [min(n, rng.randint(1, 5)) for n in lectures]
    students = [rng.randint(5, 200) for _ in range(num_courses)]
    num_teachers = max(2, round(num_courses * rng.uniform(0.5, 0.8)))
    occupancy = rng.uniform(0.6, 0.85)
    num_rooms = max(2, math.ceil(sum(lectures) / (num_periods * occupancy)))
    room_capacities = sorted(
        rng.choice([30, 50, 80, 100, 150, 200, 250]) for _ in range(num_rooms)
    )

    # Plant a feasible timetable: each course gets free periods of a free teacher, spread over days, and free rooms.
    rooms_used = [0] * num_periods
    teacher_busy = [set() for _ in range(num_teachers)]
    teachers, planted = [], []
    for c in range(num_courses):
        for t in rng.sample(range(num_teachers), num_teachers):
            free = [
                p
                for p in range(num_periods)
                if p not in teacher_busy[t] and rooms_used[p] < num_rooms
            ]
            if len(free) >= lectures[c]:
                break
        else:
            raise RuntimeError("Could not plant a timetable; retry with another seed.")
        rng.shuffle(free)
        days = {}
        for p in free:  # one lecture per day first, then fill
            days.setdefault(p // periods_per_day, p)
        periods = (list(days.values()) + [p for p in free if p not in days.values()])[
            : lectures[c]
        ]
        for p in periods:
            rooms_used[p] += 1
            teacher_busy[t].add(p)
        teachers.append(t)
        planted.append(set(periods))

    # Curricula only group courses whose planted periods do not overlap.
    curricula = []
    for _ in range(round(num_courses * rng.uniform(0.4, 0.8))):
        curriculum, busy = [], set()
        for c in rng.sample(range(num_courses), num_courses):
            if not planted[c] & busy:
                curriculum.append(c)
                busy |= planted[c]
            if len(curriculum) == rng.randint(3, 7):
                break
        if len(curriculum) >= 2:
            curricula.append(sorted(curriculum))

    unavailable = []
    for c in range(num_courses):
        if rng.random() < 0.3:
            options = [p for p in range(num_periods) if p not in planted[c]]
            for p in rng.sample(options, min(len(options), rng.randint(1, 6))):
                unavailable.append([c, p])
    return {
        "num_days": num_days,
        "periods_per_day": periods_per_day,
        "teachers": teachers,
        "lectures": lectures,
        "min_days": min_days,
        "students": students,
        "room_capacities": room_capacities,
        "curricula": curricula,
        "unavailable": sorted(unavailable),
    }


def load(path: str) -> dict:
    """Load an ITC-2007 track 3 `.ctt` file."""
    with open(path) as f:
        lines = [line.split() for line in f if line.strip()]
    header = {line[0].rstrip(":"): line[1] for line in lines if len(line) == 2}
    sections, current = {}, None
    for line in lines:
        if line[0].endswith(":") and len(line) == 1:
            current = line[0].rstrip(":")
            sections[current] = []
        elif line[0] == "END.":
            break
        elif current is not None:
            sections[current].append(line)
    course_ids = [row[0] for row in sections["COURSES"]]
    teacher_ids = sorted({row[1] for row in sections["COURSES"]})
    index = {name: c for c, name in enumerate(course_ids)}
    periods_per_day = int(header["Periods_per_day"])
    return {
        "num_days": int(header["Days"]),
        "periods_per_day": periods_per_day,
        "teachers": [teacher_ids.index(row[1]) for row in sections["COURSES"]],
        "lectures": [int(row[2]) for row in sections["COURSES"]],
        "min_days": [int(row[3]) for row in sections["COURSES"]],
        "students": [int(row[4]) for row in sections["COURSES"]],
        "room_capacities": [int(row[1]) for row in sections["ROOMS"]],
        "curricula": [
            sorted(index[name] for name in row[2:]) for row in sections["CURRICULA"]
        ],
        "unavailable": sorted(
            [index[row[0]], int(row[1]) * periods_per_day + int(row[2])]
            for row in sections["UNAVAILABILITY_CONSTRAINTS"]
        ),
    }


def evaluate(instance: dict, solution: dict) -> int:
    """Return the total penalty of `solution`; raise `ValueError` if it is malformed or breaks a hard rule."""
    num_courses = len(instance["lectures"])
    num_rooms = len(instance["room_capacities"])
    periods_per_day = instance["periods_per_day"]
    num_periods = instance["num_days"] * periods_per_day
    periods = [set() for _ in range(num_courses)]
    rooms = [set() for _ in range(num_courses)]
    occupied = set()
    penalty = 0
    for entry in solution["assignments"]:
        if len(entry) != 3 or not all(isinstance(v, int) for v in entry):
            raise ValueError(
                f"Assignment {entry!r} must be a [course, period, room] triple of integers."
            )
        c, p, r = entry
        if not (0 <= c < num_courses and 0 <= p < num_periods and 0 <= r < num_rooms):
            raise ValueError(f"Assignment {entry!r} is out of range.")
        if p in periods[c]:
            raise ValueError(f"Course {c} has two lectures in period {p}.")
        if (p, r) in occupied:
            raise ValueError(f"Room {r} holds two lectures in period {p}.")
        periods[c].add(p)
        rooms[c].add(r)
        occupied.add((p, r))
        penalty += CAPACITY_COST * max(
            0, instance["students"][c] - instance["room_capacities"][r]
        )
    for c in range(num_courses):
        if len(periods[c]) != instance["lectures"][c]:
            raise ValueError(
                f"Course {c} has {len(periods[c])} lectures instead of {instance['lectures'][c]}."
            )
    for c, p in instance["unavailable"]:
        if p in periods[c]:
            raise ValueError(f"Course {c} is placed in its unavailable period {p}.")
    groups = [
        [c for c in range(num_courses) if instance["teachers"][c] == t]
        for t in set(instance["teachers"])
    ]
    for group in groups + instance["curricula"]:
        seen = {}
        for c in group:
            for p in periods[c]:
                if p in seen:
                    raise ValueError(
                        f"Courses {seen[p]} and {c} share a teacher or a curriculum and are both in period {p}."
                    )
                seen[p] = c
    for c in range(num_courses):
        days = {p // periods_per_day for p in periods[c]}
        penalty += MIN_DAYS_COST * max(0, instance["min_days"][c] - len(days))
        penalty += ROOM_COST * (len(rooms[c]) - 1)
    for curriculum in instance["curricula"]:
        busy = set().union(*(periods[c] for c in curriculum))
        for p in busy:
            slot = p % periods_per_day
            before = slot > 0 and p - 1 in busy
            after = slot < periods_per_day - 1 and p + 1 in busy
            if not before and not after:
                penalty += ISOLATED_COST
    return penalty


BASELINE = "the first feasible timetable that CP-SAT finds (single worker) when all penalties are ignored"


def baseline(instance: dict) -> int:
    """Total penalty of the naive timetable described in `BASELINE`."""
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

    num_courses = len(instance["lectures"])
    capacities = instance["room_capacities"]
    num_rooms = len(capacities)
    periods_per_day = instance["periods_per_day"]
    num_days = instance["num_days"]
    num_periods = num_days * periods_per_day
    unavailable = {(c, p) for c, p in instance["unavailable"]}
    model = cp_model.CpModel()
    x = {
        (c, p, r): model.new_bool_var("")
        for c in range(num_courses)
        for p in range(num_periods)
        for r in range(num_rooms)
        if (c, p) not in unavailable
    }
    # y[c, p]: course c has a lecture in period p.
    y = {}
    for c in range(num_courses):
        for p in range(num_periods):
            if (c, p) not in unavailable:
                y[c, p] = model.new_bool_var("")
                model.add(y[c, p] == sum(x[c, p, r] for r in range(num_rooms)))
        model.add(
            sum(y[c, p] for p in range(num_periods) if (c, p) in y)
            == instance["lectures"][c]
        )
    for p in range(num_periods):
        for r in range(num_rooms):
            model.add_at_most_one(
                x[c, p, r] for c in range(num_courses) if (c, p, r) in x
            )
    groups = [
        [c for c in range(num_courses) if instance["teachers"][c] == t]
        for t in set(instance["teachers"])
    ]
    for group in groups + instance["curricula"]:
        for p in range(num_periods):
            model.add_at_most_one(y[c, p] for c in group if (c, p) in y)

    terms = []
    for (c, p, r), var in x.items():
        excess = instance["students"][c] - capacities[r]
        if excess > 0:
            terms.append(CAPACITY_COST * excess * var)
    for c in range(num_courses):
        working = []
        for d in range(num_days):
            day = [
                y[c, p]
                for p in range(d * periods_per_day, (d + 1) * periods_per_day)
                if (c, p) in y
            ]
            if day:
                works = model.new_bool_var("")
                model.add_max_equality(works, day)
                working.append(works)
        missing = model.new_int_var(0, num_days, "")
        model.add(missing >= instance["min_days"][c] - sum(working))
        terms.append(MIN_DAYS_COST * missing)
        used = []
        for r in range(num_rooms):
            uses = model.new_bool_var("")
            model.add_max_equality(
                uses, [x[c, p, r] for p in range(num_periods) if (c, p, r) in x]
            )
            used.append(uses)
        terms.append(ROOM_COST * (sum(used) - 1))
    for curriculum in instance["curricula"]:
        busy = [
            sum(y[c, p] for c in curriculum if (c, p) in y) for p in range(num_periods)
        ]
        for p in range(num_periods):
            slot = p % periods_per_day
            neighbors = [busy[p - 1]] if slot > 0 else []
            if slot < periods_per_day - 1:
                neighbors.append(busy[p + 1])
            isolated = model.new_bool_var("")
            model.add(isolated >= busy[p] - sum(neighbors))
            terms.append(ISOLATED_COST * isolated)
    if minimize:
        model.minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution within {time_limit}s.")
    assignments = sorted([c, p, r] for (c, p, r), var in x.items() if solver.value(var))
    solution = {"assignments": assignments}
    # Re-evaluate: penalty literals are only forced up, so the solver objective can overcount by slack.
    return (
        solution,
        evaluate(instance, solution),
        max(0, int(solver.best_objective_bound)),
    )
