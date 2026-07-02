# rgbd_pose/pose.py

import numpy as np
import open3d as o3d

from data.geometry import Pose3D


class PoseEstimator:

    def __init__(
        self,
        intrinsics,
        depth_scale=0.001,
        depth_min=0.05,
        depth_max=5.0,
        depth_patch_radius=2,
        outlier_nb_neighbors=20,
        outlier_std_ratio=2.0,
        min_points=10,
    ):
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_patch_radius = depth_patch_radius
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.min_points = min_points

    def _o3d_intrinsics(self):
        return o3d.camera.PinholeCameraIntrinsic(
            int(self.intrinsics.width),
            int(self.intrinsics.height),
            float(self.intrinsics.fx),
            float(self.intrinsics.fy),
            float(self.intrinsics.cx),
            float(self.intrinsics.cy),
        )

    def sample_depth_m(self, depth, u, v):
        h, w = depth.shape[:2]
        r = self.depth_patch_radius
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)

        patch = depth[v0:v1, u0:u1].astype(np.float64)
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        depth_m = float(np.median(valid)) * self.depth_scale
        if depth_m < self.depth_min or depth_m > self.depth_max:
            return None
        return depth_m

    def pixel_to_3d(self, u, v, depth_m):
        x = (u - self.intrinsics.cx) * depth_m / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth_m / self.intrinsics.fy
        return Pose3D(x=x, y=y, z=depth_m)

    def _point_cloud_from_mask(self, depth, mask):
        depth_m = depth.astype(np.float64) * self.depth_scale
        masked_depth = np.where(mask > 0, depth_m, 0.0).astype(np.float32)

        depth_image = o3d.geometry.Image(masked_depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            self._o3d_intrinsics(),
            depth_scale=1.0,
            depth_trunc=self.depth_max,
        )

        if len(pcd.points) < self.min_points:
            return None

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio,
        )

        points = np.asarray(pcd.points)
        if points.shape[0] < self.min_points:
            return None

        valid = points[:, 2] >= self.depth_min
        if int(valid.sum()) < self.min_points:
            return None

        return points[valid]

    def point_cloud_points(self, depth, mask):
        return self._point_cloud_from_mask(depth, mask)

    def _estimate_from_center_pixel(self, depth, center_px):
        u, v = center_px
        depth_m = self.sample_depth_m(depth, u, v)
        if depth_m is None:
            return None
        return self.pixel_to_3d(u, v, depth_m)

    def _estimate_from_point_cloud(self, depth, mask):
        points = self._point_cloud_from_mask(depth, mask)
        if points is None:
            return None

        centroid = points.mean(axis=0)
        return Pose3D(
            x=float(centroid[0]),
            y=float(centroid[1]),
            z=float(centroid[2]),
        )

    def estimate(self, depth, center_px, mask=None):
        if mask is not None:
            return self._estimate_from_point_cloud(depth, mask)
        return self._estimate_from_center_pixel(depth, center_px)
