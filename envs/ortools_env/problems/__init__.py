# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OR problem classes: instance generators, benchmark loaders, verifiers and OR-Tools reference solvers.

Every module exposes the same functions:

- `generate(seed, size)`: a fresh instance (a JSON-serializable dict), reproducible from `(seed, size)`.
- `load(path)`: an instance from the standard benchmark file format of the problem.
- `evaluate(instance, solution)`: the objective value of a solution, raising `ValueError` if it is infeasible.
  Pure Python, so it can run in a verifier image without OR-Tools.
- `reference(instance, time_limit)`: `(solution, objective, lower_bound)` from OR-Tools (`lower_bound` may be None).
- `baseline(instance)`: the objective of a naive solution that ignores the instance data, described in `BASELINE`.

All objectives are minimized and non-negative.
"""

from . import cvrp, cvrptw, jobshop, rcpsp, rostering, tardiness, timetabling

PROBLEMS = {
    module.NAME: module
    for module in (jobshop, rcpsp, cvrp, cvrptw, tardiness, rostering, timetabling)
}


def score(objective: float | None, reference: float, baseline: float) -> float:
    """
    Reward of a solution with `objective` (`None` if infeasible), normalized between a naive solution and the reference.

    0.0 is no better than the naive `baseline`, 1.0 matches the `reference` and above 1.0 beats it. Solutions worse than
    the baseline score below 0, down to -1.0, which is also the score of an infeasible solution. Scoring against the
    naive solution keeps a policy from collecting reward by submitting any feasible answer.
    """
    if objective is None:
        return -1.0
    if baseline <= reference:  # the naive solution already matches the reference
        return 1.0 if objective <= reference else -1.0
    return max(-1.0, (baseline - objective) / (baseline - reference))
