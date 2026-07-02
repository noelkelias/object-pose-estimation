# data/geometry.py

import numpy as np


class CameraIntrinsics:

    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @classmethod
    def from_matrix(cls, k, width, height):
        return cls(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            width=width,
            height=height,
        )

    def as_matrix(self):
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


class Pose3D:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def as_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other):
        return float(np.linalg.norm(self.as_array() - other.as_array()))
