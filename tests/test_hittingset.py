import unittest
import random

from algorithms.hittingset import find_hitting_set, HittingSetStrategy
from core.verification import is_hitting_set
from core.ranges import RectangleRange, get_range_space
from core.points import Point


class TestHittingSet(unittest.TestCase):

    def setUp(self):
        random.seed(42)  # For reproducibility

        self.n = 2**10
        self.m = 2**9
        self.points = [
            Point((random.uniform(0, 1), random.uniform(0, 1)), random.randint(0, 10))
            for _ in range(self.n)
        ]

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

        print(f"n: {len(self.points)}, m: {len(self.ranges)}")

    def test_hitting_set_greedy(self):
        hitting_set = find_hitting_set(
            strategy=HittingSetStrategy.GREEDY,
            points=self.points,
            rangespace=self.rangespace,
        )
        self.assertTrue(is_hitting_set(hitting_set, self.rangespace))

    def test_hitting_set_geometric(self):
        hitting_set = find_hitting_set(
            strategy=HittingSetStrategy.GEOMETRIC,
            points=self.points,
            rangespace=self.rangespace,
            vc=self.ranges[0].vc_dim,
        )
        self.assertTrue(is_hitting_set(hitting_set, self.rangespace))


if __name__ == "__main__":
    unittest.main()
