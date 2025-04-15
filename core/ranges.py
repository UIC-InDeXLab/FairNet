from abc import ABC, abstractmethod
from typing import List, Set
from core.points import Point


class Range(ABC):
    """Abstract base class for geometric ranges."""

    vc_dim: int = None  # Must be overridden in subclasses

    @abstractmethod
    def contains(self, point: Point) -> bool:
        """Check if a point is inside the range."""
        pass

    def contains_many(self, points: List[Point]) -> List[bool]:
        """Batch containment check."""
        return [self.contains(p) for p in points]

    @classmethod
    def get_vc_dim(cls) -> int:
        if cls.vc_dim is None:
            raise NotImplementedError(f"VC-dimension not defined for {cls.__name__}")
        return cls.vc_dim


class RectangleRange(Range):
    vc_dim = 2

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """
        Defines a 2D rectangle
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def contains(self, point: Point) -> bool:
        x, y = point.point
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


class BallRange(Range):
    vc_dim = None

    def __init__(self, center: Point, radius: float):
        """
        Represents a closed Euclidean ball in R^d.

        Parameters:
            center (Point): The center of the ball (d-dimensional).
            radius (float): The radius of the ball.
        """
        self.center = center.point
        self.radius = radius
        self.dim = len(center.point)
        self.dim = len(center.point)
        self.__class__.vc_dim = self.dim + 1

    def contains(self, point: Point) -> bool:
        assert len(point.point) == self.dim, "Point dimensionality mismatch."
        point = point.point
        return sum((p - c) ** 2 for p, c in zip(point, self.center)) <= self.radius**2


class HalfspaceRange(Range):
    vc_dim = None

    def __init__(self, normal: Point, offset: float):
        """
        Defines the halfspace: dot(normal, x) <= offset
        """
        self.normal = normal
        self.offset = offset
        self.dim = len(normal)
        self.__class__.vc_dim = self.dim + 1

    def contains(self, point: Point) -> bool:
        point = point.point
        return sum(a * x for a, x in zip(self.normal, point)) <= self.offset


def get_range_space(points: List[Point], ranges: List[Range]) -> List[Set[Point]]:
    """
    Keeps track of points contained in each range.
    """
    rangespace = []
    for r in ranges:
        subset = [p for p in points if r.contains(p)]
        rangespace.append(set(subset))
    # TODO: should we add points as another range? Chazelle
    # rangespace.append(points)
    return rangespace
