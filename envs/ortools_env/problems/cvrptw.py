# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Capacitated vehicle routing with time windows (CVRPTW), unlimited homogeneous fleet.

Instances follow the design of Solomon (1987) and Gehring & Homberger (1999): customers placed at random (R), in
clusters (C) or both (RC), a depot in the center, series 1 (short horizon, small capacity) or 2 (long horizon, large
capacity), and time windows on 25 to 100% of the customers. The grid grows with the square root of the number of
customers so the density stays the same. Distances are rounded Euclidean distances and travel times equal distances.
Solomon and Gehring-Homberger files load directly. The reference is the OR-Tools routing solver with guided local
search, which yields good solutions but no lower bound.
"""

import math
import random

NAME = "cvrptw"

# Number of customers (the depot is node 0 on top of these).
SIZES = {"n25": 25, "n100": 100, "n200": 200, "n400": 400, "n1000": 1000}
DEFAULT_SIZE = "n100"

STATEMENT = """\
Capacitated vehicle routing with time windows (minimize total distance).

Node 0 is the depot; nodes `1..n` are customers. `instance["coords"][i]` is the `[x, y]` position of node `i`,
`instance["demands"][i]` its integer demand (0 for the depot), `instance["ready_times"][i]` and
`instance["due_times"][i]` its time window and `instance["service_times"][i]` its service duration.
`instance["capacity"]` is the vehicle capacity. The distance between nodes `i` and `j` is
`round(math.hypot(xi - xj, yi - yj))` (Python `round`), and travelling from `i` to `j` takes that many time units.

Any number of vehicles may be used. Each route leaves the depot at time 0 and visits its customers in order. On
arriving at a customer, the vehicle waits until its ready time if it is early; service must start no later than its
due time and lasts its service time, after which the vehicle leaves for the next node. The vehicle must be back at
the depot by `instance["due_times"][0]`. The total demand served by a route must not exceed the capacity, and every
customer must be visited by exactly one route.

Solution: `{"routes": routes}` where each route is the list of customers it visits in order, without the depot.
Customers are identified by their integer index `i` (from 1 to n), not by their coordinates: for example
`{"routes": [[3, 1], [2, 4, 5]]}`.
Objective: the sum of the route lengths (distances only, not waiting or service), each including the legs from and
back to the depot. Lower is better.
"""

# Horizon, service time and capacity of Solomon's classes on a 100x100 grid.
_HORIZON = {
    ("R", 1): 230,
    ("C", 1): 1236,
    ("RC", 1): 240,
    ("R", 2): 1000,
    ("C", 2): 3390,
    ("RC", 2): 960,
}
_SERVICE = {"R": 10, "C": 90, "RC": 10}
_CAPACITY = {1: 200, 2: 1000}


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    rng = random.Random(seed)
    num_customers = SIZES[size]
    side = round(10 * math.sqrt(num_customers))
    scale = side / 100
    kind, series = rng.choice(["R", "C", "RC"]), rng.choice([1, 2])
    depot = (side // 2, side // 2)

    coords = []
    num_clustered = {"R": 0, "C": num_customers, "RC": num_customers // 2}[kind]
    centers = [
        (rng.uniform(0, side), rng.uniform(0, side)) for _ in range(rng.randint(5, 10))
    ]
    while len(coords) < num_clustered:
        cx, cy = rng.choice(centers)
        x, y = round(rng.gauss(cx, 0.04 * side)), round(rng.gauss(cy, 0.04 * side))
        if 0 <= x <= side and 0 <= y <= side:
            coords.append((x, y))
    while len(coords) < num_customers:
        coords.append((rng.randint(0, side), rng.randint(0, side)))
    coords = [depot] + coords

    service = _SERVICE[kind]
    distances = [_distance(coords, 0, i) for i in range(num_customers + 1)]
    # Every customer must be reachable alone, from the depot and back.
    horizon = max(
        round(_HORIZON[kind, series] * scale), 2 * max(distances) + service + 1
    )
    density = rng.choice([0.25, 0.5, 0.75, 1.0])
    width = (0.02, 0.1) if series == 1 else (0.05, 0.3)
    ready_times, due_times = [0], [horizon]
    for i in range(1, num_customers + 1):
        latest = horizon - distances[i] - service
        if rng.random() < density:
            center = rng.randint(distances[i], latest)
            half = round(rng.uniform(*width) * horizon / 2)
            ready_times.append(max(0, center - half))
            due_times.append(max(distances[i], min(latest, center + half)))
        else:
            ready_times.append(0)
            due_times.append(latest)
    return {
        "capacity": _CAPACITY[series],
        "coords": [list(c) for c in coords],
        "demands": [0] + [rng.randint(1, 50) for _ in range(num_customers)],
        "ready_times": ready_times,
        "due_times": due_times,
        "service_times": [0] + [service] * num_customers,
    }


def load(path: str) -> dict:
    """Load a Solomon or Gehring-Homberger file; customer 0 must be the depot."""
    with open(path) as f:
        lines = [line.split() for line in f]
    capacity = next(
        int(lines[k + 1][1]) for k, line in enumerate(lines) if line[:1] == ["NUMBER"]
    )
    rows = [
        [int(float(v)) for v in line]
        for line in lines
        if len(line) == 7 and line[0].isdigit()
    ]
    return {
        "capacity": capacity,
        "coords": [[row[1], row[2]] for row in rows],
        "demands": [row[3] for row in rows],
        "ready_times": [row[4] for row in rows],
        "due_times": [row[5] for row in rows],
        "service_times": [row[6] for row in rows],
    }


def _distance(coords: list, i: int, j: int) -> int:
    return round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))


def _check_route(instance: dict, route: list, r: int) -> int:
    """Return the length of `route`; raise `ValueError` if it breaks the capacity or a time window."""
    coords = instance["coords"]
    load = sum(instance["demands"][node] for node in route)
    if load > instance["capacity"]:
        raise ValueError(f"Route {r} carries {load} > capacity {instance['capacity']}.")
    time, length, previous = 0, 0, 0
    for node in route:
        leg = _distance(coords, previous, node)
        length += leg
        time = max(time + leg, instance["ready_times"][node])
        if time > instance["due_times"][node]:
            raise ValueError(
                f"Route {r} starts serving customer {node} at {time}, after its due time "
                f"{instance['due_times'][node]}."
            )
        time += instance["service_times"][node]
        previous = node
    leg = _distance(coords, previous, 0)
    if time + leg > instance["due_times"][0]:
        raise ValueError(
            f"Route {r} returns to the depot at {time + leg}, after {instance['due_times'][0]}."
        )
    return length + leg


def evaluate(instance: dict, solution: dict) -> int:
    """Return the total distance of `solution`; raise `ValueError` if it is malformed or infeasible."""
    num_nodes = len(instance["coords"])
    seen = set()
    total = 0
    for r, route in enumerate(solution["routes"]):
        if not route:
            continue
        for node in route:
            if not isinstance(node, int) or not 1 <= node < num_nodes:
                raise ValueError(
                    f"Route {r} contains an invalid customer {node!r}; customers are the integers 1 to "
                    f"{num_nodes - 1} and routes leave out the depot 0."
                )
            if node in seen:
                raise ValueError(f"Customer {node} is visited more than once.")
            seen.add(node)
        total += _check_route(instance, route, r)
    if len(seen) != num_nodes - 1:
        raise ValueError(f"{num_nodes - 1 - len(seen)} customers are not visited.")
    return total


BASELINE = (
    "visit the customers in index order, starting a new route whenever the next customer does not fit (capacity or "
    "time windows)"
)


def _baseline_routes(instance: dict) -> list:
    routes = [[]]
    for node in range(1, len(instance["coords"])):
        try:
            _check_route(instance, routes[-1] + [node], 0)
            routes[-1].append(node)
        except ValueError:
            routes.append([node])
    return routes


def baseline(instance: dict) -> int:
    """Total distance of the naive routes described in `BASELINE`."""
    return evaluate(instance, {"routes": _baseline_routes(instance)})


def reference(
    instance: dict, time_limit: float, num_workers: int = 8
) -> tuple[dict, int, None]:
    """Solve with the OR-Tools routing solver (guided local search); return `(solution, objective, None)`."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    coords, demands = instance["coords"], instance["demands"]
    ready_times, due_times = instance["ready_times"], instance["due_times"]
    service_times = instance["service_times"]
    num_nodes = len(coords)
    # The naive routes are feasible, so that many vehicles always suffice; unused vehicles cost nothing.
    num_vehicles = len(_baseline_routes(instance))
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    matrix = [
        [_distance(coords, i, j) for j in range(num_nodes)] for i in range(num_nodes)
    ]
    distance = routing.RegisterTransitCallback(
        lambda a, b: matrix[manager.IndexToNode(a)][manager.IndexToNode(b)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(distance)
    demand = routing.RegisterUnaryTransitCallback(
        lambda a: demands[manager.IndexToNode(a)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand, 0, [instance["capacity"]] * num_vehicles, True, "load"
    )
    # The time cumul of a node is the start of its service; the slack is the waiting time.
    travel = routing.RegisterTransitCallback(
        lambda a, b: (
            service_times[manager.IndexToNode(a)]
            + matrix[manager.IndexToNode(a)][manager.IndexToNode(b)]
        )
    )
    horizon = due_times[0]
    routing.AddDimension(travel, horizon, horizon, True, "time")
    time = routing.GetDimensionOrDie("time")
    for node in range(1, num_nodes):
        time.CumulVar(manager.NodeToIndex(node)).SetRange(
            ready_times[node], due_times[node]
        )
    for v in range(num_vehicles):
        time.CumulVar(routing.End(v)).SetRange(0, horizon)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromMilliseconds(int(time_limit * 1000))
    assignment = routing.SolveWithParameters(params)
    if assignment is None:
        raise RuntimeError(
            f"The routing solver found no solution within {time_limit}s."
        )
    routes = []
    for v in range(num_vehicles):
        index = assignment.Value(routing.NextVar(routing.Start(v)))
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = assignment.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    solution = {"routes": routes}
    return solution, evaluate(instance, solution), None
