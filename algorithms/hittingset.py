import numpy as np

from typing import List
from scipy.optimize import linprog
from enum import Enum

from core.ranges import Range
from algorithms.epsnet import build_epsnet_sample
from core.points import Point


class HittingSetStrategy(Enum):
    GREEDY = "greedy"
    GEOMETRIC = "geometric"


def find_hitting_set(strategy: HittingSetStrategy, **kwargs) -> List[Point]:
    if strategy == HittingSetStrategy.GREEDY:
        return find_hitting_set_greedy(**kwargs)
    elif strategy == HittingSetStrategy.GEOMETRIC:
        return find_hitting_set_geometric(**kwargs)
    else:
        raise NotImplementedError("Strategy not implemented.")


def find_hitting_set_greedy(
    points: List[Point], rangespace: List[Range], limit=-1
) -> List[Point]:
    """
    Find a hitting set for the given ranges using a greedy algorithm.

    Parameters:
        points (List[Point]): The points to consider.
        ranges (List[Range]): The ranges to cover.
        limit (int): The maximum size of the hitting set. Default is -1 (no limit).
    """

    hitting_set = []  # The resulting hitting set
    remaining_ranges = rangespace.copy()  # Copy of ranges to track uncovered ranges
    counter = limit

    while remaining_ranges and (limit == -1 or counter > 0):
        # Count how many ranges each point hits
        point_hits = {point: 0 for point in points}
        for r in remaining_ranges:
            for point in r:
                if point in point_hits:
                    point_hits[point] += 1

        # Find the point that hits the most ranges
        best_point = max(point_hits, key=point_hits.get)

        # Add the best point to the hitting set
        hitting_set.append(best_point)

        # Remove all ranges hit by the best point
        remaining_ranges = [r for r in remaining_ranges if best_point not in r]

        counter -= 1
    print(f"[find_hitting_set_greedy] hitting set size: {len(hitting_set)}")
    return hitting_set


def find_hitting_set_geometric(
    points: List[Point], rangespace: List[Range], vc
) -> List[Point]:
    weights, epsilon = _get_reweights(points, rangespace)
    print(f"[find_hitting_set_geometric] epsilon: {epsilon}")
    epsnet = build_epsnet_sample(
        points=points, rangespace=rangespace, epsilon=epsilon, vc=vc, weights=weights
    )
    print(f"[find_hitting_set_geometric] epsnet size: {len(epsnet)}")
    return epsnet


def _get_reweights(points: List[Point], rangespace: List[Range]) -> List[float]:
    """
    Solve the hitting set problem using linear programming.

    Parameters:
        points (List[Point]): List of points.
        ranges (List[Set[Point]]): List of ranges, where each range is a set of points.

    Returns:
        List[float]: Optimal values of z_i for each point.
    """
    n = len(points)  # Number of points
    m = len(rangespace)  # Number of ranges

    # Objective function: Minimize sum(z_i)
    c = np.ones(n)

    # Constraints: For each range r, sum(z_i for p_i in r) >= 1
    A = np.zeros((m, n))
    for j, r in enumerate(rangespace):
        for i, p in enumerate(points):
            if p in r:
                A[j, i] = 1
    b = np.ones(m)

    # Bounds: 0 <= z_i <= 1 for all i
    bounds = [(0, 1) for _ in range(n)]

    # Solve the linear program
    result = linprog(c, A_ub=-A, b_ub=-b, bounds=bounds, method="highs")

    if result.success:
        # Normalize the result by dividing each value by the sum of all values
        z_values = result.x
        z_sum = np.sum(z_values)
        epsilon = 1 / z_sum if z_sum > 0 else 0
        normalized_z_values = z_values / z_sum if z_sum > 0 else z_values
        return normalized_z_values, epsilon
    else:
        raise ValueError("Linear programming failed to find a solution.")
