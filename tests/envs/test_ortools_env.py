# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the OR-Tools environment: generators, verifiers, references and the MCP episode flow."""

import random
import time

import pytest

pytest.importorskip("ortools")
pytest.importorskip("fastmcp")

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from ortools_env import sandbox
from ortools_env.problems import jobshop, PROBLEMS, score
from ortools_env.server.ortools_environment import ORToolsEnvironment

TINY_SIZES = {
    "jobshop": "6x6",
    "rcpsp": "j10",
    "cvrp": "n20",
    "cvrptw": "n25",
    "tardiness": "n10",
    "rostering": "4x1",
    "timetabling": "c10",
}


def _text(obs) -> str:
    if obs.error is not None:
        return obs.error.message
    return obs.result.content[0].text


def _call(env, tool, **arguments):
    return env.step(CallToolAction(tool_name=tool, arguments=arguments))


def test_taillard_generator_reproduces_ta01():
    # ta01 (Taillard 1993): time seed 840612802, machine seed 398197754; first job of the published instance.
    instance = jobshop.taillard(15, 15, 840612802, 398197754)
    machines = [7, 13, 5, 8, 4, 3, 11, 12, 9, 15, 10, 14, 6, 1, 2]
    durations = [94, 66, 10, 53, 26, 15, 65, 82, 10, 27, 93, 92, 96, 70, 83]
    assert instance["jobs"][0] == [[m - 1, d] for m, d in zip(machines, durations)]


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_generate_is_deterministic(problem):
    module = PROBLEMS[problem]
    assert module.generate(3, TINY_SIZES[problem]) == module.generate(
        3, TINY_SIZES[problem]
    )


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_reference_solution_passes_the_verifier(problem):
    module = PROBLEMS[problem]
    instance = module.generate(0, TINY_SIZES[problem])
    solution, objective, lower_bound = module.reference(
        instance, time_limit=5, num_workers=1
    )
    assert module.evaluate(instance, solution) == objective
    assert lower_bound is None or lower_bound <= objective


# Seeds of tiny instances that CP-SAT solves to proven optimality within seconds.
OPTIMAL_SEEDS = {
    "jobshop": 0,
    "rcpsp": 0,
    "tardiness": 0,
    "rostering": 2,
    "timetabling": 1,
}


@pytest.mark.parametrize("problem", sorted(OPTIMAL_SEEDS))
def test_verifier_matches_the_proven_optimum(problem):
    # Cross-checks the pure-Python verifier against the CP-SAT model: at a proven optimum, they must agree.
    module = PROBLEMS[problem]
    instance = module.generate(OPTIMAL_SEEDS[problem], TINY_SIZES[problem])
    solution, _, lower_bound = module.reference(instance, time_limit=30)
    assert module.evaluate(instance, solution) == lower_bound


def _routing_objective(module, instance, routes):
    """Cost of `routes` in an OR-Tools routing model of `instance`, or None if OR-Tools rejects them."""
    from ortools.constraint_solver import pywrapcp

    coords = instance["coords"]
    manager = pywrapcp.RoutingIndexManager(len(coords), len(routes), 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance(a, b):
        return module._distance(coords, manager.IndexToNode(a), manager.IndexToNode(b))

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance))
    demand = routing.RegisterUnaryTransitCallback(
        lambda a: instance["demands"][manager.IndexToNode(a)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand, 0, [instance["capacity"]] * len(routes), True, "load"
    )
    if "due_times" in instance:
        travel = routing.RegisterTransitCallback(
            lambda a, b: (
                instance["service_times"][manager.IndexToNode(a)] + distance(a, b)
            )
        )
        horizon = instance["due_times"][0]
        routing.AddDimension(travel, horizon, horizon, True, "time")
        time = routing.GetDimensionOrDie("time")
        for node in range(1, len(coords)):
            time.CumulVar(manager.NodeToIndex(node)).SetRange(
                instance["ready_times"][node], instance["due_times"][node]
            )
    routing.CloseModelWithParameters(pywrapcp.DefaultRoutingSearchParameters())
    assignment = routing.ReadAssignmentFromRoutes(routes, True)
    return None if assignment is None else assignment.ObjectiveValue()


@pytest.mark.parametrize("problem", ["cvrp", "cvrptw"])
def test_routing_verifiers_match_ortools(problem):
    # Cross-checks feasibility and cost of reference and perturbed routes against an OR-Tools routing model.
    module = PROBLEMS[problem]
    rng = random.Random(0)
    for seed in range(3):
        instance = module.generate(seed, TINY_SIZES[problem])
        routes = module.reference(instance, time_limit=2)[0]["routes"]
        candidates = [routes]
        for _ in range(20):
            perturbed = [list(route) for route in routes]
            a, b = rng.randrange(len(perturbed)), rng.randrange(len(perturbed))
            i, j = rng.randrange(len(perturbed[a])), rng.randrange(len(perturbed[b]))
            perturbed[a][i], perturbed[b][j] = perturbed[b][j], perturbed[a][i]
            candidates.append(perturbed)
        for candidate in candidates:
            try:
                objective = module.evaluate(instance, {"routes": candidate})
            except ValueError:
                objective = None
            assert objective == _routing_objective(module, instance, candidate)


def test_verifiers_reject_infeasible_solutions():
    instance = jobshop.generate(0, "6x6")
    with pytest.raises(ValueError):
        jobshop.evaluate(instance, {"starts": [[0] * 6 for _ in range(6)]})

    rcpsp = PROBLEMS["rcpsp"]
    instance = rcpsp.generate(0, "j10")
    with pytest.raises(ValueError):
        rcpsp.evaluate(instance, {"starts": [0] * 10})

    cvrp = PROBLEMS["cvrp"]
    instance = cvrp.generate(0, "n20")
    with pytest.raises(ValueError, match="not visited"):
        cvrp.evaluate(instance, {"routes": [[1]]})
    with pytest.raises(ValueError, match="more than once"):
        cvrp.evaluate(instance, {"routes": [[1], [1]]})

    tardiness = PROBLEMS["tardiness"]
    instance = tardiness.generate(0, "n10")
    with pytest.raises(ValueError):
        tardiness.evaluate(instance, {"sequence": [0] * 10})

    rostering = PROBLEMS["rostering"]
    instance = rostering.generate(0, "4x1")
    with pytest.raises(ValueError):
        rostering.evaluate(instance, {"schedule": [[0] * 7 for _ in range(4)]})

    cvrptw = PROBLEMS["cvrptw"]
    instance = {
        "capacity": 10,
        "coords": [[0, 0], [10, 0], [20, 0]],
        "demands": [0, 1, 1],
        "ready_times": [0, 0, 0],
        "due_times": [100, 15, 50],
        "service_times": [0, 1, 1],
    }
    assert cvrptw.evaluate(instance, {"routes": [[1], [2]]}) == 60
    with pytest.raises(ValueError, match="after its due time"):
        cvrptw.evaluate(instance, {"routes": [[2, 1]]})

    timetabling = PROBLEMS["timetabling"]
    instance = timetabling.generate(0, "c10")
    lectures = [[c, 0, 0] for c, n in enumerate(instance["lectures"]) for _ in range(n)]
    with pytest.raises(ValueError):
        timetabling.evaluate(instance, {"assignments": lectures})


def test_score():
    # Normalized between the naive baseline (0.0) and the reference (1.0).
    assert score(100, reference=100, baseline=300) == 1.0
    assert score(300, reference=100, baseline=300) == 0.0
    assert score(200, reference=100, baseline=300) == 0.5
    assert score(50, reference=100, baseline=300) == 1.25
    assert score(500, reference=100, baseline=300) == -1.0
    assert score(None, reference=100, baseline=300) == -1.0
    # The naive solution already matches the reference.
    assert score(100, reference=100, baseline=100) == 1.0
    assert score(101, reference=100, baseline=100) == -1.0


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_baseline_is_no_better_than_the_reference(problem):
    module = PROBLEMS[problem]
    instance = module.generate(0, TINY_SIZES[problem])
    _, objective, _ = module.reference(instance, time_limit=5, num_workers=1)
    assert module.baseline(instance) >= objective


def test_naive_solution_scores_zero():
    env = ORToolsEnvironment()
    env.reset(problem="tardiness", size="n10", seed=0, reference_time_limit_s=5)
    obs = _call(env, "submit_solution", solution={"sequence": list(range(10))})
    assert obs.reward == 0.0


def test_sandbox_blocks_ortools_and_times_out(tmp_path):
    result = sandbox.run("import ortools", {}, str(tmp_path), timeout_s=30)
    assert "not allowed" in result["output"]
    result = sandbox.run(
        "import ortools", {}, str(tmp_path), timeout_s=30, allow_ortools=True
    )
    assert result["returncode"] == 0
    result = sandbox.run("while True: pass", {}, str(tmp_path), timeout_s=1)
    assert result["timed_out"]


def test_tools_are_listed():
    env = ORToolsEnvironment()
    names = {tool.name for tool in env.step(ListToolsAction()).tools}
    assert names == {
        "run_python",
        "check_solution",
        "test_algorithm",
        "submit_solution",
        "submit_algorithm",
    }


def test_level_1_episode():
    env = ORToolsEnvironment()
    obs = env.reset(problem="jobshop", size="6x6", seed=0, reference_time_limit_s=5)
    assert "Job shop scheduling" in obs.metadata["prompt"]

    obs = _call(env, "run_python", code="print(instance['num_jobs'])")
    assert _text(obs).startswith("6\n")
    assert not obs.done

    solution, objective, _ = jobshop.reference(env._instance, 5, 1)
    obs = _call(env, "submit_solution", solution=solution)
    assert obs.done
    assert obs.reward == 1.0
    assert env.state.objective == objective

    obs = _call(env, "submit_solution", solution=solution)
    assert "episode is over" in _text(obs)


def test_level_1_infeasible_solution_scores_minus_one():
    env = ORToolsEnvironment()
    env.reset(problem="tardiness", size="n10", seed=0, reference=1)
    obs = _call(env, "submit_solution", solution={"sequence": [0, 0]})
    assert obs.done
    assert obs.reward == -1.0


def test_level_2_episode():
    env = ORToolsEnvironment()
    env.reset(
        problem="tardiness",
        size="n10",
        level=2,
        seed=0,
        num_eval_instances=2,
        solve_time_limit_s=10,
        reference_time_limit_s=5,
    )
    code = (
        "def solve(instance):\n"
        "    n = len(instance['processing_times'])\n"
        "    return {'sequence': sorted(range(n), key=lambda j: instance['due_dates'][j])}\n"
    )
    obs = _call(env, "submit_algorithm", code=code)
    assert obs.done
    assert "naive=" in _text(obs)
    assert -1.0 <= obs.reward <= 1.0


def test_level_2_crashing_algorithm_scores_minus_one():
    env = ORToolsEnvironment()
    env.reset(
        problem="tardiness",
        size="n10",
        level=2,
        seed=0,
        eval_seeds=[1, 2],
        eval_references=[1, 1],
    )
    obs = _call(
        env, "submit_algorithm", code="def solve(instance):\n    raise RuntimeError"
    )
    assert obs.done
    assert obs.reward == -1.0


def test_level_2_algorithm_may_return_numpy_values():
    env = ORToolsEnvironment()
    env.reset(
        problem="tardiness",
        size="n10",
        level=2,
        seed=0,
        num_eval_instances=1,
        solve_time_limit_s=10,
        reference_time_limit_s=5,
    )
    code = (
        "import numpy as np\n"
        "def solve(instance):\n"
        "    return {'sequence': np.argsort(instance['due_dates'])}\n"
    )
    obs = _call(env, "submit_algorithm", code=code)
    assert "objective=" in _text(obs)
    assert obs.reward > -1.0


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_check_solution_reports_objective_without_ending_the_episode(problem):
    env = ORToolsEnvironment()
    env.reset(problem=problem, size=TINY_SIZES[problem], seed=0, max_python_calls=2)
    solution, objective, _ = PROBLEMS[problem].reference(env._instance, 5, 1)
    obs = _call(env, "check_solution", solution=solution)
    assert f"objective={objective} naive=" in _text(obs)
    assert "reference" not in _text(obs)
    assert not obs.done
    obs = _call(env, "check_solution", solution={})
    assert "Infeasible" in _text(obs)
    assert not obs.done
    obs = _call(env, "check_solution", solution=solution)
    assert "no Python calls left" in _text(obs)


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_test_algorithm_runs_solve_on_the_example(problem):
    env = ORToolsEnvironment()
    env.reset(
        problem=problem,
        size=TINY_SIZES[problem],
        level=2,
        seed=0,
        num_eval_instances=1,
        solve_time_limit_s=20,
    )
    # A constant `solve` returning the reference solution of the example instance.
    code = f"def solve(instance):\n    return {PROBLEMS[problem].reference(env._instance, 5, 1)[0]!r}\n"
    obs = _call(env, "test_algorithm", code=code)
    assert "objective=" in _text(obs)
    assert "naive=" in _text(obs)
    assert not obs.done
    obs = _call(
        env, "test_algorithm", code="def solve(instance):\n    raise RuntimeError"
    )
    assert "failed" in _text(obs)
    assert env.state.python_calls == 2
    assert not obs.done


def test_submitted_main_block_does_not_run(tmp_path):
    code = (
        "def solve(instance):\n"
        "    return {'sequence': [0]}\n"
        "if __name__ == '__main__':\n"
        "    raise RuntimeError\n"
    )
    result = sandbox.run(code, {}, str(tmp_path), timeout_s=10, solve=True)
    assert result["solution"] == {"sequence": [0]}


def test_loaders_read_benchmark_formats(tmp_path):
    solomon = tmp_path / "c101.txt"
    solomon.write_text(
        "C101\n\nVEHICLE\nNUMBER     CAPACITY\n  25         200\n\nCUSTOMER\n"
        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME\n\n"
        "    0      40         50          0          0       1236          0\n"
        "    1      45         68         10        912        967         90\n"
    )
    instance = PROBLEMS["cvrptw"].load(str(solomon))
    assert instance["capacity"] == 200
    assert instance["due_times"] == [1236, 967]

    ctt = tmp_path / "toy.ctt"
    ctt.write_text(
        "Name: Toy\nCourses: 2\nRooms: 1\nDays: 5\nPeriods_per_day: 4\nCurricula: 1\nConstraints: 1\n\n"
        "COURSES:\nSceCosC Ocra 3 3 30\nArcTec Indaco 2 2 42\n\nROOMS:\nA 32\n\n"
        "CURRICULA:\nCur1 2 SceCosC ArcTec\n\nUNAVAILABILITY_CONSTRAINTS:\nArcTec 1 2\n\nEND.\n"
    )
    instance = PROBLEMS["timetabling"].load(str(ctt))
    assert instance["lectures"] == [3, 2]
    assert instance["curricula"] == [[0, 1]]
    assert instance["unavailable"] == [[1, 6]]
    _, objective, _ = PROBLEMS["timetabling"].reference(instance, 5, 1)
    assert objective >= 0


def test_solve_failures_explain_themselves(tmp_path):
    result = sandbox.run("x = 1", {}, str(tmp_path), timeout_s=10, solve=True)
    assert "must define solve(instance)" in result["output"]
    code = "def solve(instance):\n    pass\n"
    result = sandbox.run(code, {}, str(tmp_path), timeout_s=10, solve=True)
    assert "must return a dict, not NoneType" in result["output"]
    assert result["solution"] is None


def test_background_processes_do_not_outlive_the_call(tmp_path):
    code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', "
        '\'import time; time.sleep(2); open("late.txt", "w").write("x")\'])\n'
    )
    result = sandbox.run(code, {}, str(tmp_path), timeout_s=10)
    assert result["returncode"] == 0
    time.sleep(3)
    assert not (tmp_path / "late.txt").exists()


def test_level_2_malformed_solution_scores_minus_one():
    env = ORToolsEnvironment()
    env.reset(
        problem="tardiness",
        size="n10",
        level=2,
        seed=0,
        eval_seeds=[1],
        eval_references=[1],
    )
    obs = _call(
        env,
        "submit_algorithm",
        code="import os\nopen('solution.json', 'w').write('{broken')\nos._exit(0)",
    )
    assert obs.done
    assert obs.reward == -1.0
    assert "failed" in _text(obs)
