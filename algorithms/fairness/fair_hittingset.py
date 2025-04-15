import random
import math
import numpy as np
from scipy.optimize import linprog

from typing import List
from core.fairness import *
from algorithms.hittingset import HittingSetStrategy
from core.points import Point
from core.ranges import Range
from algorithms.fairness.fair_epsnet import _augment_epsnet, build_fair_epsnet_sample


def find_fair_hitting_set(
    strategy: HittingSetStrategy, fairconfig: FairConfig, **kwargs
) -> List[Point]:
    if strategy == HittingSetStrategy.GREEDY:
        return find_fair_hitting_set_greedy(fairconfig=fairconfig, **kwargs)
    elif strategy == HittingSetStrategy.GEOMETRIC:
        return find_fair_hitting_set_geometric(fairconfig=fairconfig, **kwargs)
    else:
        raise NotImplementedError("Strategy not implemented.")


def find_fair_hitting_set_greedy(
    points: List[Point], rangespace: List[Range], fairconfig: FairConfig, c1=1
) -> List[Point]:
    """
    This is a naive implementation that simply adds arbitrary points to the hitting set.
    The number of points to add is O(log k) where k is the number of colors.
    """
    hitting_set = []  # The resulting hitting set
    remaining_ranges = rangespace.copy()  # Copy of ranges to track uncovered ranges

    while remaining_ranges:
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

    k = fairconfig.k
    color_ratios = []
    for color in range(k):
        rate = [p for p in points if p.color == color]
        color_ratios.append(len(rate) / len(points))

    # augment more points to the hitting set
    v = c1 * math.ceil(math.log(4 * k))
    hitting_set = _augment_epsnet(hitting_set, points, color_ratios, v, k)

    print(f"[find_fair_hitting_set_greedy] hitting set size: {len(hitting_set)}")
    return hitting_set


def find_fair_hitting_set_geometric(
    points: List[Point], rangespace: List[Range], vc, fairconfig: FairConfig, c1=1
) -> List[Point]:
    k = fairconfig.k
    color_ratios = []
    for color in range(k):
        rate = [p for p in points if p.color == color]
        color_ratios.append(len(rate) / len(points))

    weights, epsilon = _get_fair_reweights(
        points=points, rangespace=rangespace, k=k, color_ratios=color_ratios
    )
    print(f"[find_hitting_set_geometric] epsilon: {epsilon}")
    weights_by_color = []
    for color in range(k):
        weights_by_color.append(
            sum([weights[i] for i in range(len(weights)) if points[i].color == color])
        )
    print(f"[find_hitting_set_geometric] weights by color: {weights_by_color}")

    _reweight_points(points, weights)

    epsnet = build_fair_epsnet_sample(
        points=points,
        rangespace=rangespace,
        epsilon=epsilon,
        vc=vc,
        weights=weights,
        fairconfig=fairconfig,
        c1=c1,
        c2=4,
        color_ratios=color_ratios
    )
    print(f"[find_fair_hitting_set_geometric] epsnet size: {len(epsnet)}")
    return epsnet


def _get_fair_reweights(
    points: List[Point], rangespace: List[set], k: int, color_ratios: List[float] = None
) -> List[float]:
    """
    Solve the hitting set problem with fairness constraints using linear programming.

    Parameters:
        points (List[Point]): List of points, each with a color attribute.
        rangespace (List[Set[Point]]): List of ranges, where each range is a set of points.
        color_ratios (List[float]): Desired ratios for the sum of weights for each color.

    Returns:
        List[float]: Optimal values of z_i for each point.
    """
    n = len(points)  # Number of points
    m = len(rangespace)  # Number of ranges

    # Objective function: Maximize epsilon
    c = np.zeros(n + 1)  # n variables for z_i and 1 for epsilon
    c[-1] = -1  # Coefficient for epsilon (maximize epsilon by minimizing -epsilon)

    # Constraints: For each range r, sum(z_i for p_i in r) >= epsilon
    A = np.zeros((m, n + 1))
    for j, r in enumerate(rangespace):
        for i, p in enumerate(points):
            if p in r:
                A[j, i] = -1  # Negative because linprog minimizes
        A[j, -1] = 1  # Coefficient for epsilon
    b = np.zeros(m)  # Right-hand side for range constraints

    # Constraint: Sum of all z_i's should be equal to 1
    A_eq = [np.ones(n + 1)]
    A_eq[0][-1] = 0  # No epsilon in this constraint
    b_eq = [1.0]

    # Fairness constraints: Sum of z_i for each color must match the given ratios
    if len(color_ratios) != k:
        raise ValueError(
            "The length of color_ratios must match the number of unique colors."
        )

    color_to_indices = {color: [] for color in range(k)}
    for i, p in enumerate(points):
        color_to_indices[p.color].append(i)

    for color, ratio in enumerate(color_ratios):
        row = np.zeros(n + 1)
        for i in color_to_indices[color]:
            row[i] = 1
        A_eq.append(row)
        b_eq.append(ratio)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Bounds: 0 <= z_i <= 1 for all i, and epsilon >= 0
    bounds = [(0, 1) for _ in range(n)] + [(0, None)]  # z_i in [0, 1], epsilon >= 0

    # Solve the linear program
    result = linprog(
        c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if result.success:
        z_values = result.x[:-1]  # Extract z_i values
        epsilon = result.x[-1]  # Extract epsilon
        return z_values, epsilon
    else:
        raise ValueError("Linear programming failed to find a solution.")


def _reweight_points(points: List[Point], weights: List[float]) -> List[Point]:
    """
    Reweight points based on the given weights.

    Parameters:
        points (List[Point]): List of points to be reweighted.
        weights (List[float]): List of weights for each point.

    Returns:
        List[Point]: List of reweighted points.
    """
    for i, weight in enumerate(weights):
        points[i].weight = weight
