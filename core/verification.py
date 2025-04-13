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
