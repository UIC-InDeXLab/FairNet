import unittest
import random

from algorithms.fairness.fair_epsnet import *
from core.verification import is_fair_epsnet
from core.ranges import RectangleRange, get_range_space
from core.points import Point
from core.fairness import FairConfig, FairnessMeasure


class TestFairEpsNet(unittest.TestCase):

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

        heavy_ranges = []
        for r in self.rangespace_55:
            if len(r) >= self.epsilon * self.n:
                heavy_ranges.append(r)

        print(f"[Setup] n: {len(self.points_55)}, m: {len(self.ranges)}")
        print(f"[Setup] n: {len(self.points_28)}, m: {len(self.ranges)}")

        print(f"[Setup] Heavy ranges: {len(heavy_ranges)}")
        blue = [p for p in self.points_55 if p.color == 0]
        red = [p for p in self.points_55 if p.color == 1]
        print(f"[Setup] Blue points: {len(blue)}, Red points: {len(red)}")

        blue = [p for p in self.points_28 if p.color == 0]
        red = [p for p in self.points_28 if p.color == 1]
        print(f"[Setup] Blue points: {len(blue)}, Red points: {len(red)}")

    def test_fair_epsnet_sampling_55(self):
        fairconfig = FairConfig(k=2, fairness=FairnessMeasure.DP)
        epsnet = build_fair_epsnet(
            strategy=EpsNetStrategy.SAMPLE,
            points=self.points_55,
            rangespace=self.rangespace_55,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            fairconfig=fairconfig,
            success_prob=0.9,
            color_ratios=[0.5, 0.5],
        )
        self.assertTrue(
            is_fair_epsnet(epsnet, self.rangespace_55, self.epsilon, self.points_55)
        )

    def test_fair_epsnet_sampling_28(self):
        fairconfig = FairConfig(k=2, fairness=FairnessMeasure.DP)
        epsnet = build_fair_epsnet(
            strategy=EpsNetStrategy.SAMPLE,
            points=self.points_28,
            rangespace=self.rangespace_28,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            fairconfig=fairconfig,
            success_prob=0.9,
            color_ratios=[0.25, 0.75],
        )
        self.assertTrue(
            is_fair_epsnet(epsnet, self.rangespace_28, self.epsilon, self.points_28)
        )

    def test_fair_epsnet_discrepancy55(self):
        fairconfig = FairConfig(k=2, fairness=FairnessMeasure.DP)
        epsnet = build_fair_epsnet(
            strategy=EpsNetStrategy.DISCREPANCY,
            points=self.points_55,
            rangespace=self.rangespace_55,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            fairconfig=fairconfig,
        )
        self.assertTrue(
            is_fair_epsnet(epsnet, self.rangespace_55, self.epsilon, self.points_55)
        )

    def test_fair_epsnet_discrepancy28(self):
        fairconfig = FairConfig(k=2, fairness=FairnessMeasure.DP)
        epsnet = build_fair_epsnet(
            strategy=EpsNetStrategy.SKETCH_MERGE,
            points=self.points_28,
            rangespace=self.rangespace_28,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            fairconfig=fairconfig,
            c1=0,
        )
        self.assertTrue(
            is_fair_epsnet(epsnet, self.rangespace_28, self.epsilon, self.points_28)
        )

    def test_fair_epsnet_naive(self):
        fairconfig = FairConfig(k=2, fairness=FairnessMeasure.DP)
        epsnet = build_fair_epsnet(
            strategy=EpsNetStrategy.NAIVE_FAIR,
            points=self.points_28,
            rangespace=self.rangespace_28,
            epsilon=self.epsilon,
            vc=self.ranges[0].vc_dim,
            fairconfig=fairconfig,
        )
        self.assertTrue(
            is_fair_epsnet(epsnet, self.rangespace_28, self.epsilon, self.points_28)
        )
