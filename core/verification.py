from typing import List, Set
from core.ranges import Point, Range
from core.ranges import get_range_space


def is_epsnet(
    epsnet: List[Point], rangespace: List[Set[Point]], epsilon: float
) -> bool:
    """
    Verify if the given points form an eps-net for the specified ranges.

    Parameters:
        points (List[Point]): The points to verify.
        ranges (List[Range]): The ranges to verify against.
        epsilon (float): The epsilon parameter for the eps-net.

    Returns:
        bool: True if the points form an eps-net, False otherwise.
    """
    n = len(epsnet)
    heavy_ranges = []
    for r in rangespace:
        if len(r) >= epsilon * n:
            heavy_ranges.append(r)

    for r in heavy_ranges:
        if not any(p in r for p in epsnet):
            return False

    return True


def is_hitting_set(hitting_set: List[Point], rangespace: List[Set[Point]]) -> bool:
    """
    Verify if the given points form a hitting set for the specified ranges.

    Parameters:
        points (List[Point]): The points to verify.
        ranges (List[Range]): The ranges to verify against.

    Returns:
        bool: True if the points form a hitting set, False otherwise.
    """
    for r in rangespace:
        if not any(p in r for p in hitting_set):
            return False

    return True


def is_fair_epsnet(
    epsnet: List[Point],
    rangespace: List[Set[Point]],
    epsilon: float,
    points: List[Point],
) -> bool:
    """
    Verify if the given points form a fair eps-net for the specified ranges.

    Parameters:
        points (List[Point]): The points to verify.
        ranges (List[Range]): The ranges to verify against.
        epsilon (float): The epsilon parameter for the eps-net.
        color_ratios (List[float]): The color ratios for fairness.

    Returns:
        bool: True if the points form a fair eps-net, False otherwise.
    """
    color_ratios = []
    for i in range(len(points)):
        rate = [p for p in points if p.color == i]
        color_ratios.append(len(rate) / len(points))

    n = len(epsnet)
    heavy_ranges = []
    for r in rangespace:
        if len(r) >= epsilon * n:
            heavy_ranges.append(r)

    for r in heavy_ranges:
        if not any(p in r for p in epsnet):
            print("Not an eps-net!")
            return False

    # Check color ratios
    for i, color_ratio in enumerate(color_ratios):
        # check if the ratio is almost equal
        if (
            abs(len([p for p in epsnet if p.color == i]) / len(epsnet) - color_ratio)
            > 0.01
        ):
            print("Not a fair eps-net!")
            return False

    return True
