# FairNet

**FairNet** is a Python library for geometric approximation algorithms focused on sampling problems such as:

- **Epsilon-nets**
- **Epsilon-samples**
- **Geometric hitting sets**

The repository also provides *fair* variants of these algorithms that return representative subsets respecting target color distributions on the points (e.g., An $\varepsilon$-net containing 50% red and 50% blue).

This project is designed for **research purposes** and **quick proof-of-concept** implementations in geometric approximation and fair sampling.

## 🚀 Getting Started
### Installation
```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
```
./run_tests.sh
```

## 🧪 Minimal Example
The following snippet constructs a random colored point set in $[0, 1]^2$, generates axis-aligned rectangles, and builds an $\varepsilon$-net:

```
import random
from core.ranges import Point, RectangleRange, get_range_space
from algorithms.epsnet import build_epsnet_sample

# Configuration
n, m = 1024, 1024
success_prob, eps = 0.9, 0.1

# Generate random points in unit square
points = [Point(point=[random.uniform(0, 1), random.uniform(0, 1)], color=0) for _ in range(n)]

# Generate axis-aligned rectangles
ranges = []
for _ in range(m):
    x1, x2 = sorted([random.uniform(0, 1), random.uniform(0, 1)])
    y1, y2 = sorted([random.uniform(0, 1), random.uniform(0, 1)])
    ranges.append(RectangleRange(x1, x2, y1, y2))

# Build the range space
rangespace = get_range_space(points, ranges)
vc = 4  # VC-dimension for axis-aligned rectangles in 2D

# Build ε-net
epsnet = build_epsnet_sample(points, rangespace, vc, eps, success_prob)
print(f"Size of ε-net: {len(epsnet)}")
```

For advanced examples or real datasets, see the [tests/](./tests/) directory.

## 📂 Repository Structure
```
algorithms/     # Core algorithms for epsilon-net and hitting set construction (standard and fair)
core/           # Foundational data structures for geometric range spaces
tests/          # Example scripts and usage tests
```
- `core/` includes:
    - Abstract definition of ranges
    - Implementations such as:
        - Axis-aligned rectangles (2D)
        - Half-spaces in $ℝ^d$

## ⚙️ Features
- Randomized and deterministic algorithms for constructing $\varepsilon$-nets:
    - Sampling-based
    - Discrepancy-based
- Fair variants that ensure **demographic parity** over color-labeled subsets

## 📚 Citation
If you use this codebase in your research, please cite:
```
[BibTeX or citation coming soon...]
```
