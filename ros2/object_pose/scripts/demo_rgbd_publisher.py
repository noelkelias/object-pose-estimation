#!/usr/bin/env python3
# ros2/object_pose/scripts/demo_rgbd_publisher.py

import os
import sys

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.frame import load_rgbd_sequence
from data.geometry import CameraIntrinsics
from data.synthetic import render_frame


def animated_cube_center(index, rate_hz):
    """Lissajous-style path so TF / RViz show clear motion."""
    phase = 2.0 * np.pi * index / max(rate_hz * 4.0, 1.0)
    return (
        float(0.10 * np.sin(phase)),
        float(0.08 * np.cos(phase * 1.3)),
        float(0.55 + 0.10 * np.sin(phase * 0.5)),
    )


def camera_info_msg(intrinsics, stamp, frame_id):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = int(intrinsics.width)
    msg.height = int(intrinsics.height)
    msg.k = [
        float(intrinsics.fx),
        0.0,
        float(intrinsics.cx),
        0.0,
        float(intrinsics.fy),
        float(intrinsics.cy),
        0.0,
        0.0,
        1.0,
    ]
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [
        float(intrinsics.fx),
        0.0,
        float(intrinsics.cx),
        0.0,
        0.0,
        float(intrinsics.fy),
        float(intrinsics.cy),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return msg


class DemoRgbdPublisher(Node):

    def __init__(self):
        super().__init__("demo_rgbd_publisher")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("animated", True)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        animated = bool(self.get_parameter("animated").value)

        self.frames = []
        if not animated:
            exports_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_exports")
            if not os.path.isdir(exports_dir):
                raise RuntimeError(f"No synthetic exports at {exports_dir}")
            self.frames = load_rgbd_sequence(exports_dir, max_frames=20)
            if not self.frames:
                raise RuntimeError(f"No frames loaded from {exports_dir}")

        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, "/rgb/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/depth/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self.index = 0
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        if animated:
            self.get_logger().info(
                f"Publishing animated synthetic RGB-D at {self.rate_hz:.1f} Hz"
            )
        else:
            self.get_logger().info(
                f"Publishing {len(self.frames)} static frames at {self.rate_hz:.1f} Hz"
            )

    def tick(self):
        if self.frames:
            frame = self.frames[self.index % len(self.frames)]
        else:
            center = animated_cube_center(self.index, self.rate_hz)
            frame = render_frame(cube_center_m=center, seed=42 + self.index)
        self.index += 1

        stamp = self.get_clock().now().to_msg()

        rgb_msg = self.bridge.cv2_to_imgmsg(frame.rgb, encoding="rgb8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.frame_id

        depth_msg = self.bridge.cv2_to_imgmsg(frame.depth, encoding="16UC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.frame_id

        info_msg = camera_info_msg(frame.intrinsics, stamp, self.frame_id)

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)


def main():
    rclpy.init()
    node = DemoRgbdPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
