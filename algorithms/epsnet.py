import math
import random

from enum import Enum
from typing import List, Set

from core.ranges import *


class EpsNetStrategy(Enum):
    SAMPLE = "sample"
    DISCREPANCY = "disc"
    SKETCH_MERGE = "sketch_merge"


def build_epsnet(strategy: EpsNetStrategy = "sample", **kwargs):
    if strategy == "sample":
        build_epsnet_sample(**kwargs)
    elif strategy == "disc":
        build_epsnet_discrepancy(**kwargs)
    elif strategy == "sketch_merge":
        build_epsnet_sketch_merge(**kwargs)
    else:
        raise NotImplementedError("Strategy not implemented.")


def get_epsnet_size(epsilon, vc, success_prob):
    phi = 1 - success_prob
    d = vc
    m = max(
        (4 / epsilon) * math.log2(4 / phi), (8 * d / epsilon) * math.log2(16 / epsilon)
    )
    return math.ceil(m)


def build_epsnet_sample(
    points: List[Point], rangespace: List[Set[Point]], vc, epsilon, success_prob=0.9
) -> List[Point]:
    """
    Build eps-nets by random sampling.

    Reference:
        - Har-Peled, Sariel. Geometric approximation algorithms. No. 173. American Mathematical Soc., 2011.
        - Chapter 5
    """
    d = vc
    m = get_epsnet_size(epsilon, d, success_prob)
    m = min(m, len(points))
    print(f"[build_epsnet_sample] epsnet size m: {int(m)}")
    return random.choices(points, k=math.ceil(m))


def build_epsnet_discrepancy(
    points: List[Point], rangespace: List[Set[Point]], vc, epsilon
) -> List[Point]:
    """Build eps-net by iterative discrepancy halving.

    Reference:
        - Chazelle, Bernard. The Discrepancy Method: Randomness and Complexity. Cambridge University Press, 2000.
        - Chapter 4
    """
    d = vc
    # m = c1 * (d / epsilon) * math.log(d / epsilon)  # TODO: what is constant?
    m = get_epsnet_size(epsilon, d, 0.9)
    m = min(m, len(points))
    print(f"[build_epsnet_discrepancy] epsnet size m: {int(m)}")
    subset = points
    while len(subset) > 2 * m:
        # TODO[optimize]: filter-out ranges not hit by subset
        _, half = _random_halving(subset, rangespace)
        subset = half
    return subset  # Final size is almost m


def _greedy_discrepancy_halving(
    rangespace: List[Set[Point]],
    matching: List[Tuple[Point, Point]],
) -> List[Point]:
    """
    Assigns a coloring χ: X → {-1, +1} to minimize max discrepancy over Ranges.
    Greedy heuristic: for each pair in matching, choose +1 or -1 that minimizes max discrepancy.
    """
    coloring = {}
    half = []

    for k, pair in enumerate(matching):
        print(
            f"[_greedy_discrepancy_halving] counter: {k + 1} / {len(matching)}",
            end="\r",
        )
        coloring[pair[0]], coloring[pair[1]] = 1, -1
        # TODO[optimize]: should we recalculate?
        max_pos = max(abs(sum(coloring.get(p, 0) for p in r)) for r in rangespace)

        coloring[pair[0]], coloring[pair[0]] = -1, 1
        max_neg = max(abs(sum(coloring.get(p, 0) for p in r)) for r in rangespace)

        # Keep the better choice
        if max_pos < max_neg:
            coloring[pair[0]], coloring[pair[1]] = 1, -1
            half.append(pair[0])
        else:
            coloring[pair[0]], coloring[pair[1]] = -1, 1
            half.append(pair[1])
    print()

    return coloring, half


def _random_halving(points: List[Point], rangespace: List[Set[Point]]) -> List[Point]:
    shuffled = points.copy()
    random.shuffle(shuffled)

    # Ensure even number of points (drop last one if needed)
    if len(shuffled) % 2 == 1:
        shuffled = shuffled[:-1]

    matching = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]

    return _greedy_discrepancy_halving(rangespace, matching)


def build_epsnet_sketch_merge(
    points: List[Point], rangespace: List[Set[Point]], vc, epsilon, c1
) -> List[Point]:
    """
    Build eps-net by sketch-and-merge discrepancy.

    Parameters:
        points (List[Point])
        ranges (List[Range])
        epsilon (float): Epsilon parameter for the eps-net.
        c1 (float): Constant for partition size.
        c2 (float): Constant for final size of epsnet.
    """
    d = vc

    m = get_epsnet_size(epsilon, d, 0.9)  # size of final epsnet
    m = min(m, len(points))
    print(f"[build_epsnet_sketch_merge] epsnet size m: {int(m)}")

    # p = c1 * d**3 * (1 / epsilon**2) * math.log(d / epsilon)  # size of each partition
    p = m * 2**c1  # size of each partition
    p = 2 ** math.ceil(math.log2(p))  # round to nearest power of 2

    print(f"[build_epsnet_sketch_merge] partition size p: {p}")
    partitions = []
    for i in range(0, len(points), p):
        partitions.append(points[i : i + p])  # TODO: exclude this from timings
    print(f"[build_epsnet_sketch_merge] Starting sketch-and-merge...")
    root = _sketch_merge(partitions, rangespace)
    # m = c2 * (d / epsilon**2) * math.log(d / epsilon)
    while len(root) > 2 * m:
        _, root = _random_halving(root, rangespace)

    return root


def _sketch_merge(
    partitions: List[Set[Point]], rangespace: List[Set[Point]]
) -> List[Set[Point]]:
    length = len(partitions)
    while length > 1:
        for i in range(length // 2):
            print(
                f"[_sketch_merge] pair: {i + 1} / {length // 2} of total nodes: {length}"
            )
            try:
                # Merge two partitions
                merged = list(
                    set(partitions[2 * i]) | set(partitions[2 * i + 1])
                )  # Merging
                # TODO[optimize]: we are always passing the whole ranges!
                _, partitions[i] = _random_halving(merged, rangespace)  # Halving
            except IndexError:
                # Odd number of partitions, last one remains
                raise ValueError(
                    "Odd number of partitions. The number of points should be 2^k."
                )
        length = length // 2

    print()

    # You are at the root of the tree
    root = partitions[0]
    return root
