import unittest
import random

from algorithms.epsnet import build_epsnet, EpsNetStrategy
from core.verification import is_epsnet
from core.ranges import RectangleRange, get_range_space
from core.points import Point


class TestEpsNet(unittest.TestCase):

    def setUp(self):
        random.seed(42)  # For reproducibility

        self.n = 2**10
        self.m = 2**9
        self.points = [
            Point((random.uniform(0, 1), random.uniform(0, 1)), random.randint(0, 10))
            for _ in range(self.n)
        ]
        self.epsilon = 0.7

        # Generate self.m rectangles with random bounds
        self.ranges = [
            RectangleRange(
                random.uniform(0, 0.5),  # x_min
                random.uniform(0.5, 1),  # x_max
                random.uniform(0, 0.5),  # y_min
                random.uniform(0.5, 1),  # y_max
            )
            for _ in range(self.m)
        ]

        self.rangespace = get_range_space(self.points, self.ranges)

        heavy_ranges = []
        for r in self.rangespace:
            if len(r) >= self.epsilon * self.n:
                heavy_ranges.append(r)

        print(f"n: {len(self.points)}, m: {len(self.ranges)}")
        print(f"Heavy ranges: {len(heavy_ranges)}")

    def test_epsnet_sampling(self):
        epsnet = build_epsnet(
            strategy=EpsNetStrategy.SAMPLE,
            points=self.points,
            rangespace=self.rangespace,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            success_prob=0.9,
        )
        self.assertTrue(is_epsnet(epsnet, self.rangespace, self.epsilon))

    def test_epsnet_discrepancy(self):
        epsnet = build_epsnet(
            strategy=EpsNetStrategy.DISCREPANCY,
            points=self.points,
            rangespace=self.rangespace,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
        )
        self.assertTrue(is_epsnet(epsnet, self.rangespace, self.epsilon))

    def test_epsnet_sketch_merge(self):
        epsnet = build_epsnet(
            strategy=EpsNetStrategy.SKETCH_MERGE,
            points=self.points,
            rangespace=self.rangespace,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            c1=0,
        )
        self.assertTrue(is_epsnet(epsnet, self.rangespace, self.epsilon))


if __name__ == "__main__":
    unittest.main()
