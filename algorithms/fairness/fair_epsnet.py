from typing import List, Set, Tuple
from algorithms.epsnet import *
from core.fairness import *
from algorithms.epsnet import _greedy_discrepancy_halving, _sketch_merge


def build_fair_epsnet(strategy: EpsNetStrategy, fairconfig: FairConfig, **kwargs):
    if strategy == EpsNetStrategy.SAMPLE:
        return build_fair_epsnet_sample(fairconfig=fairconfig, **kwargs)
    elif strategy == EpsNetStrategy.DISCREPANCY:
        return build_fair_epsnet_discrepancy(fairconfig=fairconfig, **kwargs)
    elif strategy == EpsNetStrategy.SKETCH_MERGE:
        return build_fair_epsnet_sketch_merge(fairconfig=fairconfig, **kwargs)
    elif strategy == EpsNetStrategy.NAIVE_FAIR:
        return build_fair_epsnet_naive(fairconfig=fairconfig, **kwargs)
    else:  # TODO: implement naive, add as much as we can!
        raise NotImplementedError("Strategy not implemented.")


def build_fair_epsnet_sample(
    points: List[Point],
    rangespace: List[Set[Point]],
    vc,
    epsilon,
    fairconfig: FairConfig,
    success_prob=0.9,
    c1=1,
    c2=1,
    weights=None,  # used for sampling and also fairness (weighted ratios)
) -> List[Point]:
    d = vc
    m = get_epsnet_size(epsilon, d, success_prob, c2)
    m = min(m, len(points))

    fairness = fairconfig.fairness
    k = fairconfig.k
    # v = math.ceil(2 * math.log(4 * k))
    v = c1 * math.ceil(math.log(4 * k))
    print(f"[build_fair_epsnet_sample] epsnet size m: {int(m)}, v: {v}")

    color_ratios = []
    # TODO[optimize]: this can be done faster
    for i in range(k):
        rate = [p for p in points if p.color == i]
        color_ratios.append(len(rate) / len(points))

    if fairness == FairnessMeasure.DP:
        epsnet = random.choices(points, weights=weights, k=math.ceil(m))
        while not _is_good_epsnet(epsnet, k, v, color_ratios):
            print("[build_fair_epsnet_sample] Bad epsnet, resampling...")
            epsnet = random.choices(points, weights=weights, k=math.ceil(m))
        return _augment_epsnet(epsnet, points, color_ratios, v, k)
    else:
        # Custom-ratio
        raise NotImplementedError("Fairness measure not implemented.")


def _is_good_epsnet(
    epsnet: List[Point],
    k: int,
    v: int,
    color_ratios: List[float],
) -> bool:
    """
    Check if the epsnet is good.
    """
    W = len([p for p in epsnet])
    for color in range(k):
        epsnet_ratio = len([p for p in epsnet if p.color == color])

        if epsnet_ratio > v * color_ratios[color] * W:
            return False
    return True


def _augment_epsnet(
    epsnet: List[Point],
    points: List[Point],
    color_ratios: List[float],
    v: int,
    k: int,
) -> List[Point]:
    """
    Augment the epsnet with points from the point-set.
    """
    print("[_augment_epsnet] epsnet colors count:")
    for color in range(k):
        print(
            f"\t[_augment_epsnet] Color {color}: {len([p for p in epsnet if p.color == color])}"
        )
    W = len([p for p in epsnet])
    to_adds = []
    for color in range(k):
        to_add = v * color_ratios[color] * W - len(
            [p for p in epsnet if p.color == color]
        )
        to_add = int(to_add)
        to_adds.append(to_add)
        print(f"[_augment_epsnet] Color {color} to add: {to_add}")

    for color in range(k):
        to_add = to_adds[color]
        if to_add > 0:
            # randomly select points from the point-set
            epsnet += random.sample(
                # [p for p in points if (p.color == color and p not in epsnet)],
                points,
                to_add,
            )
        # if to_add > 0:
        #     # TODO[optimize]: this can be done faster
        #     # first sort colors by weight then randomly select
        #     sorted_points = sorted(
        #         [p for p in points if (p.color == color and p not in epsnet)],
        #         key=lambda x: x.weight,
        #     )
        #     epsnet += sorted_points[:to_add]
    return epsnet


def build_fair_epsnet_discrepancy(
    fairconfig: FairConfig,
    points: List[Point],
    rangespace: List[Set[Point]],
    vc,
    epsilon,
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
    print(f"[build_fair_epsnet_discrepancy] epsnet size m: {int(m)}")
    subset = points
    while len(subset) > 2 * m:
        # TODO[optimize]: filter-out ranges not hit by subset
        _, half = _fair_havling(subset, rangespace, fairconfig)
        subset = half
    return subset  # Final size is almost m


def _fair_havling(
    points: List[Point], rangespace: List[Set[Point]], fairconfig: FairConfig
) -> List[Point]:
    """
    First finds a fair matching of points, then applies the greedy discrepancy halving.
    """
    k = fairconfig.k
    matching = []
    for color in range(k):
        # TODO: what if the number of points are not even?
        p_color = [p for p in points if p.color == color]
        matching += [(p_color[i], p_color[i + 1]) for i in range(0, len(p_color), 2)]

    return _greedy_discrepancy_halving(rangespace, matching)


def build_fair_epsnet_sketch_merge(
    points: List[Point],
    rangespace: List[Set[Point]],
    vc,
    epsilon,
    c1,
    fairconfig: FairConfig,
    c2=1,
) -> List[Point]:
    """
    Build eps-net by sketch-and-merge discrepancy.

    Parameters:
        points (List[Point])
        ranges (List[Range])
        epsilon (float): Epsilon parameter for the eps-net.
        c1 (float): Constant for partition size.
    """
    d = vc

    m = get_epsnet_size(epsilon, d, 0.9, c2)  # size of final epsnet
    m = min(m, len(points))
    print(f"[build_fair_epsnet_sketch_merge] epsnet size m: {int(m)}")

    # p = c1 * d**3 * (1 / epsilon**2) * math.log(d / epsilon)  # size of each partition
    p = m * 2**c1  # size of each partition
    p = 2 ** math.ceil(math.log2(p))  # round to nearest power of 2

    print(f"[build_fair_epsnet_sketch_merge] partition size p: {p}")
    partitions = []
    for i in range(0, len(points), p):
        partitions.append(points[i : i + p])  # TODO: exclude this from timings
    print(f"[build_fair_epsnet_sketch_merge] Starting sketch-and-merge...")
    root = _sketch_merge(
        partitions,
        rangespace,
        halving=lambda *args: _fair_havling(fairconfig=fairconfig, *args),
    )
    # m = c2 * (d / epsilon**2) * math.log(d / epsilon)
    while len(root) > 2 * m:
        _, root = _fair_havling(root, rangespace, fairconfig)

    return root


def build_fair_epsnet_naive(
    points: List[Point],
    rangespace: List[Set[Point]],
    vc,
    epsilon,
    fairconfig: FairConfig,
    success_prob=0.9,
    c1=1,
    weights=None,  # used for sampling and also fairness (weighted ratios)
) -> List[Point]:
    d = vc
    m = get_epsnet_size(epsilon, d, success_prob)
    m = min(m, len(points))

    fairness = fairconfig.fairness
    k = fairconfig.k
    v = c1 * k
    print(f"[build_fair_epsnet_naive] epsnet size m: {int(m)}, v: {v}")

    color_ratios = []
    # TODO[optimize]: this can be done faster
    for i in range(k):
        rate = [p for p in points if p.color == i]
        color_ratios.append(len(rate) / len(points))

    if fairness == FairnessMeasure.DP:
        epsnet = random.choices(points, weights=weights, k=math.ceil(m))
        while not _is_good_epsnet(epsnet, k, v, color_ratios):
            print("[build_fair_epsnet_naive] Bad epsnet, resampling...")
            epsnet = random.choices(points, weights=weights, k=math.ceil(m))
        return _augment_epsnet(epsnet, points, color_ratios, v, k)
    else:
        # Custom-ratio
        raise NotImplementedError("Fairness measure not implemented.")
