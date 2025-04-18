import unittest
import random

from algorithms.fairness.fair_hittingset import *
from core.verification import is_fair_hittingset
from core.ranges import RectangleRange, get_range_space
from core.points import Point
from core.fairness import FairConfig, FairnessMeasure


class TestFairHittingSet(unittest.TestCase):

    def setUp(self):
        random.seed(42)  # For reproducibility

        self.n = 2**10
        self.m = 2**9
        # exactly 500 blue and 500 red points
        self.points_55 = [
            Point((random.uniform(0, 1), random.uniform(0, 1)), 0)
            for i in range(self.n)
            if i % 2 == 0
        ] + [
            Point((random.uniform(0, 1), random.uniform(0, 1)), 1)
            for i in range(self.n)
            if i % 2 == 1
        ]
        self.points_28 = [
            Point((random.uniform(0, 1), random.uniform(0, 1)), 0)
            for i in range(self.n)
            if i % 4 == 0
        ] + [
            Point((random.uniform(0, 1), random.uniform(0, 1)), 1)
            for i in range(self.n)
            if i % 4 != 1
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

        self.rangespace_55 = get_range_space(self.points_55, self.ranges)
        self.rangespace_28 = get_range_space(self.points_28, self.ranges)

        print(f"[Setup] n: {len(self.points_55)}, m: {len(self.ranges)}")
        print(f"[Setup] n: {len(self.points_28)}, m: {len(self.ranges)}")

        blue = [p for p in self.points_55 if p.color == 0]
        red = [p for p in self.points_55 if p.color == 1]
        print(f"[Setup] Blue points: {len(blue)}, Red points: {len(red)}")

        blue = [p for p in self.points_28 if p.color == 0]
        red = [p for p in self.points_28 if p.color == 1]
        print(f"[Setup] Blue points: {len(blue)}, Red points: {len(red)}")

    def test_fair_hittingset_greedy55(self):
        fairconfig = FairConfig(fairness=FairnessMeasure.DP, k=2)
        hitting_set = find_fair_hitting_set(
            strategy=HittingSetStrategy.GREEDY,
            points=self.points_55,
            rangespace=self.rangespace_55,
            fairconfig=fairconfig,
        )
        self.assertTrue(
            is_fair_hittingset(
                hitting_set=hitting_set,
                rangespace=self.rangespace_55,
                points=self.points_55,
            )
        )

    def test_fair_hittingset_greedy28(self):
        fairconfig = FairConfig(fairness=FairnessMeasure.DP, k=2)
        hitting_set = find_fair_hitting_set(
            strategy=HittingSetStrategy.GREEDY,
            points=self.points_28,
            rangespace=self.rangespace_28,
            fairconfig=fairconfig,
            c1=2,  # larger size
        )
        self.assertTrue(
            is_fair_hittingset(
                hitting_set=hitting_set,
                rangespace=self.rangespace_28,
                points=self.points_28,
            )
        )

    def test_fair_hittingset_geometric55(self):
        fairconfig = FairConfig(fairness=FairnessMeasure.DP, k=2)
        hitting_set = find_fair_hitting_set(
            strategy=HittingSetStrategy.GEOMETRIC,
            points=self.points_55,
            rangespace=self.rangespace_55,
            fairconfig=fairconfig,
            vc=self.ranges[0].vc_dim,
        )
        self.assertTrue(
            is_fair_hittingset(
                hitting_set=hitting_set,
                rangespace=self.rangespace_55,
                points=self.points_55,
            )
        )

    def test_fair_hittingset_geometric28(self):
        fairconfig = FairConfig(fairness=FairnessMeasure.DP, k=2)
        hitting_set = find_fair_hitting_set(
            strategy=HittingSetStrategy.GEOMETRIC,
            points=self.points_28,
            rangespace=self.rangespace_28,
            fairconfig=fairconfig,
            vc=self.ranges[0].vc_dim,
        )
        self.assertTrue(
            is_fair_hittingset(
                hitting_set=hitting_set,
                rangespace=self.rangespace_28,
                points=self.points_28,
            )
        )
