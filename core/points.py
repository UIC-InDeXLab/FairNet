from typing import Tuple


class Point:
    def __init__(self, point: Tuple[float, ...], color):
        """
        Represents a point in d-dimensional space.
        Parameters:
            point (Tuple[float, ...]): The coordinates of the point.
            color: The color of the point (demographic group).
        """
        self.point = point
        self.color = color
