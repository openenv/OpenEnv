# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Capacitated vehicle routing (CVRP), unlimited homogeneous fleet.

Instances follow the "X" benchmark design of Uchoa et al. (2017, "New benchmark instances for the Capacitated Vehicle
Routing Problem"): points on a 1000x1000 grid, depot in the center, a corner or at random, customers placed at random,
in clusters or both, several demand distributions, and a capacity set from a target number of customers per route.
Distances are rounded Euclidean distances, as in CVRPLIB. CVRPLIB `.vrp` files load directly. The reference is the
OR-Tools routing solver with guided local search, which yields good solutions but no lower bound.
"""

import math
import random

NAME = "cvrp"

# Number of customers (the depot is node 0 on top of these).
SIZES = {"n20": 20, "n100": 100, "n200": 200, "n400": 400, "n1000": 1000}
DEFAULT_SIZE = "n200"

STATEMENT = """\
Capacitated vehicle routing (minimize total distance).

Node 0 is the depot; nodes `1..n` are customers. `instance["coords"][i]` is the `[x, y]` position of node `i`,
`instance["demands"][i]` its integer demand (0 for the depot) and `instance["capacity"]` the vehicle capacity. The
distance between nodes `i` and `j` is `round(math.hypot(xi - xj, yi - yj))` (Python `round`). Any number of vehicles
may be used. Each route starts and ends at the depot, and the total demand served by a route must not exceed the
capacity. Every customer must be visited by exactly one route.

Solution: `{"routes": routes}` where each route is the list of customers it visits in order, without the depot.
Customers are identified by their integer index `i` in `instance["coords"]` (from 1 to n), not by their
coordinates: for example `{"routes": [[3, 1], [2, 4, 5]]}`.
Objective: the sum of the route lengths, each including the legs from and back to the depot. Lower is better.
"""


def generate(seed: int, size: str = DEFAULT_SIZE) -> dict:
    rng = random.Random(seed)
    num_customers = SIZES[size]
    depot = rng.choice(
        [(500, 500), (0, 0), (rng.randint(0, 1000), rng.randint(0, 1000))]
    )
    positioning = rng.choice(["random", "clustered", "random-clustered"])

    coords = []
    if positioning != "random":
        seeds = [
            (rng.randint(0, 1000), rng.randint(0, 1000))
            for _ in range(rng.randint(3, 8))
        ]
        num_clustered = (
            num_customers if positioning == "clustered" else num_customers // 2
        )
        while len(coords) < num_clustered:
            # Each point is accepted with a probability that decays exponentially with its distance to a cluster seed.
            x, y = rng.randint(0, 1000), rng.randint(0, 1000)
            if rng.random() < max(
                math.exp(-math.hypot(x - sx, y - sy) / 40) for sx, sy in seeds
            ):
                coords.append((x, y))
    while len(coords) < num_customers:
        coords.append((rng.randint(0, 1000), rng.randint(0, 1000)))

    demand_type = rng.choice(["unitary", "small", "large", "skewed"])
    if demand_type == "unitary":
        demands = [1] * num_customers
    elif demand_type == "small":
        demands = [rng.randint(1, 10) for _ in range(num_customers)]
    elif demand_type == "large":
        demands = [rng.randint(50, 100) for _ in range(num_customers)]
    else:  # many small demands and a few large ones
        demands = [
            rng.randint(50, 100) if rng.random() < 0.15 else rng.randint(1, 10)
            for _ in range(num_customers)
        ]
    customers_per_route = rng.uniform(3, 25)
    capacity = max(
        max(demands), math.ceil(customers_per_route * sum(demands) / num_customers)
    )
    return {
        "capacity": capacity,
        "coords": [list(depot)] + [list(c) for c in coords],
        "demands": [0] + demands,
    }


def load(path: str) -> dict:
    """Load a CVRPLIB `.vrp` file with `EUC_2D` distances; the depot is moved to node 0."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    header, sections, current = {}, {}, None
    for line in lines:
        if line.endswith("_SECTION"):
            current = line
            sections[current] = []
        elif line == "EOF":
            break
        elif ":" in line and current is None:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
        else:
            sections[current].append(line.split())
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("Only EUC_2D instances are supported.")
    coords = {
        int(row[0]): [float(row[1]), float(row[2])]
        for row in sections["NODE_COORD_SECTION"]
    }
    demands = {int(row[0]): int(row[1]) for row in sections["DEMAND_SECTION"]}
    depot = int(sections["DEPOT_SECTION"][0][0])
    order = [depot] + [node for node in sorted(coords) if node != depot]
    return {
        "capacity": int(header["CAPACITY"]),
        "coords": [coords[node] for node in order],
        "demands": [0] + [demands[node] for node in order[1:]],
    }


def _distance(coords: list, i: int, j: int) -> int:
    return round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))


def evaluate(instance: dict, solution: dict) -> int:
    """Return the total distance of `solution`; raise `ValueError` if it is malformed or infeasible."""
    coords, demands, capacity = (
        instance["coords"],
        instance["demands"],
        instance["capacity"],
    )
    num_nodes = len(coords)
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
        load = sum(demands[node] for node in route)
        if load > capacity:
            raise ValueError(f"Route {r} carries {load} > capacity {capacity}.")
        path = [0] + route + [0]
        total += sum(_distance(coords, a, b) for a, b in zip(path, path[1:]))
    if len(seen) != num_nodes - 1:
        raise ValueError(f"{num_nodes - 1 - len(seen)} customers are not visited.")
    return total


BASELINE = "visit the customers in index order, starting a new route whenever the next customer does not fit"


def baseline(instance: dict) -> int:
    """Total distance of the naive routes described in `BASELINE`."""
    routes, load = [[]], 0
    for node in range(1, len(instance["coords"])):
        if load + instance["demands"][node] > instance["capacity"]:
            routes.append([])
            load = 0
        routes[-1].append(node)
        load += instance["demands"][node]
    return evaluate(instance, {"routes": routes})


def reference(
    instance: dict, time_limit: float, num_workers: int = 8
) -> tuple[dict, int, None]:
    """Solve with the OR-Tools routing solver (guided local search); return `(solution, objective, None)`."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    coords, demands, capacity = (
        instance["coords"],
        instance["demands"],
        instance["capacity"],
    )
    num_nodes = len(coords)
    # Unused vehicles cost nothing, so give the solver enough slack over the bin-packing bound.
    num_vehicles = 2 * math.ceil(sum(demands) / capacity) + 2
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    matrix = [
        [_distance(coords, i, j) for j in range(num_nodes)] for i in range(num_nodes)
    ]
    transit = routing.RegisterTransitCallback(
        lambda a, b: matrix[manager.IndexToNode(a)][manager.IndexToNode(b)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    demand = routing.RegisterUnaryTransitCallback(
        lambda a: demands[manager.IndexToNode(a)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand, 0, [capacity] * num_vehicles, True, "load"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
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
