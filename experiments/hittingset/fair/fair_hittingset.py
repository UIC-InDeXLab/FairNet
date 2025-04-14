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

# %% [markdown]
# Rectanlges:

# %%
def report_fair_hittingset(n, m, ratios):
    print("generating points...")
    color_counts = []
    for ratio in ratios:
        color_counts.append(int(n * ratio))
    
    points = []
    for i, count in enumerate(color_counts):
        points += [Point(point=[random.uniform(0, 1), random.uniform(0, 1)], color=i) for _ in range(count)]

    ranges = []
    for _ in range(m):
        xmin = random.uniform(0, 1)
        ymin = random.uniform(0, 1)
        xmax = random.uniform(xmin, 1)  # Ensure xmax > xmin
        ymax = random.uniform(ymin, 1)  # Ensure ymax > ymin
        ranges.append(RectangleRange(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
    # filter out ranges with no points
    ranges = [r for r in ranges if any(r.contains(p) for p in points)]
    m = len(ranges)
    rangespace = get_range_space(points, ranges)
    vc = 2
            
    print(f"n: {n}, m: {m}, vc: {vc}")

    # for _ in range(10):
    start = time.time()
    hittingset = find_fair_hitting_set_geometric(
        points, 
        rangespace, 
        vc, 
        FairConfig(k=len(ratios), fairness=FairnessMeasure.DP), 
        c1=1/2
    )
    end = time.time()

    print(f"Number of points in epsnet: {len(hittingset)}, time taken: {end - start:.6f} seconds")

    success = is_fair_hittingset(hitting_set=hittingset, rangespace=rangespace, points=points)
    print(f"Success: {success}")  # Verify the eps-net
    
    return (n, m, end - start, success, 
            len([p for p in hittingset if p.color == 0]), len([p for p in hittingset if p.color == 1]),
            len(ratios), # k
            ratios,
            [len([p for p in hittingset if p.color == i]) for i in range(len(ratios))]) # color counts)

# %%
n_values = [2**13, 2**14, 2**15, 2**16, 2**17]
m_values = [2**10]
rates = [
    [0.5, 0.5],
    [0.6, 0.4],
    [0.7, 0.3],
    [0.8, 0.2],
    [0.25, 0.25, 0.25, 0.25],
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
    [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
]

aggregated_results = {
    "n": [],
    "m": [],
    "time": [],
    "success": [],
    "red_points": [],
    "blue_points": [],
    "k": [],
    "ratios": [],
    "color_counts": []
}

for n in n_values:
    for m in m_values:
        for ratio in rates:
            tries = 10
            print(f"Running for n={n}, m={m}, ratio={ratio}")
            result = report_fair_hittingset(n, m, ratio)
            while not result[3] and tries > 0:
                print(f"Retrying... {tries} tries left")
                result = report_fair_hittingset(n, m, ratio)
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

result = pd.DataFrame(aggregated_results)

# %%
result.to_csv("fair_hitting_set_results_rectangle.csv", index=False)


