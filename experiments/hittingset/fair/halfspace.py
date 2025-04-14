# %%
import sys
sys.path.append('../../..')

import numpy as np
import random
import pandas as pd
import time
from matplotlib import pyplot as plt

from core.points import *
from core.ranges import *
from algorithms.fairness.fair_hittingset import *
from core.ranges import get_range_space
from core.verification import is_fair_hittingset

# Halfspace:

# %%
def report_fair_hittingset_halfspace(n, m, dim, ratios):
    print("generating points and ranges...")
    
    color_counts = []
    for ratio in ratios:
        color_counts.append(int(n * ratio))
    
    points = []
    for i, count in enumerate(color_counts):
        points += [Point(point=[random.uniform(0, 1), random.uniform(0, 1)], color=i) for _ in range(count)]
    
    # points = [Point(point=[random.uniform(0, 1), random.uniform(0, 1)], color=0) for _ in range(n // 2)]
    # points += [Point(point=[random.uniform(0, 1), random.uniform(0, 1)], color=1) for _ in range(n // 2)]
    ranges = []
    for _ in range(m):
        # Generate a random normal vector in R^d
        normal = [random.uniform(-1, 1) for _ in range(dim)]
        
        # Calculate the range of possible dot products with the [0, 1]^d hypercube
        min_dot = sum(min(0, n) for n in normal)  # Minimum dot product with [0, 1]^d
        max_dot = sum(max(0, n) for n in normal)  # Maximum dot product with [0, 1]^d
        
        # Choose an offset within the range [min_dot, max_dot] to ensure intersection
        offset = random.uniform(min_dot, max_dot)
        
        # Create a HalfspaceRange object
        halfspace = HalfspaceRange(normal=normal, offset=offset)
        ranges.append(halfspace)
    # filter out ranges with no points
    ranges = [r for r in ranges if any(r.contains(p) for p in points)]
    m = len(ranges)
    rangespace = get_range_space(points, ranges)
    vc = dim + 1

    print(f"n: {n}, m: {m}, vc: {vc}")
    
    # for _ in range(10):
    start = time.time()
    hittingset = find_fair_hitting_set_geometric(
        points, 
        rangespace, 
        vc,
        fairconfig=FairConfig(k=2, fairness=FairnessMeasure.DP),
        c1=1/2    
    )
    end = time.time()

    print(f"Number of points in hittingset: {len(hittingset)}, time taken: {end - start:.6f} seconds")

    success = is_fair_hittingset(hitting_set=hittingset, rangespace=rangespace, points=points)
    print(f"Success: {success}")  # Verify the eps-net
    
    return (n, m, end - start, success, 
            len([p for p in hittingset if p.color == 0]), len([p for p in hittingset if p.color == 1]),
            len(ratios), # k
            ratios,
            [len([p for p in hittingset if p.color == i]) for i in range(len(ratios))]) # color counts)

# %%
n_values = [2**13, 2**14, 2**15, 2**16, 2**17]
dims = [4, 8, 16, 32]
m_values = [2**10]
ratios = [[0.5, 0.5]]


aggregated_results = {
    "n": [],
    "m": [],
    "time": [],
    "success": [],
    "red_points": [],
    "blue_points": [],
    "k": [],
    "ratios": [],
    "color_counts": [],
    "dim": []
}

for n in n_values:
    for m in m_values:
        for dim in dims:
            ratio = [0.5, 0.5]
            tries = 10
            print(f"Running for n={n}, m={m}, dim={dim}")
            result = report_fair_hittingset_halfspace(n, m, dim, ratio)
            while not result[3] and tries > 0:
                print(f"Retrying... {tries} tries left")
                result = report_fair_hittingset_halfspace(n, m, dim, ratio)
                tries -= 1
            
            aggregated_results["n"].append(result[0])
            aggregated_results["m"].append(result[1])
            aggregated_results["time"].append(result[2])
            aggregated_results["success"].append(result[3])
            aggregated_results["red_points"].append(result[4])
            aggregated_results["blue_points"].append(result[5])
            aggregated_results["k"].append(result[6])
            aggregated_results["ratios"].append(result[7])
            aggregated_results["color_counts"].append(result[8])
            aggregated_results["dim"].append(dim)

result = pd.DataFrame(aggregated_results)

# %%
result.to_csv("fair_hitting_set_results_halfspace.csv", index=False)


