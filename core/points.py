from typing import Tuple


class Point:
    def __init__(self, point: Tuple[float, ...], color, weight=1):
        """
        Represents a point in d-dimensional space.
        Parameters:
            point (Tuple[float, ...]): The coordinates of the point.
            color: The color of the point (demographic group).
            weight: The weight of the point (default is 1).
        """
        self.point = point
        self.color = color
        self.weight = weight
