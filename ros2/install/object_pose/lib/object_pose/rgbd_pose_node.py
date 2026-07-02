#!/usr/bin/env python3
# ros2/object_pose/scripts/rgbd_pose_node.py

import os
import sys

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.geometry import CameraIntrinsics
from rgbd_pose.detection import ObjectDetector
from rgbd_pose.pose import PoseEstimator


def camera_info_to_intrinsics(info):
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    return CameraIntrinsics.from_matrix(k, int(info.width), int(info.height))


class RgbdPoseNode(Node):

    def __init__(self):
        super().__init__("rgbd_pose_node")

        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("hsv_lower", [100, 80, 80])
        self.declare_parameter("hsv_upper", [130, 255, 255])
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("object_frame_id", "object_frame")

        depth_scale = float(self.get_parameter("depth_scale").value)
        hsv_lower = tuple(int(v) for v in self.get_parameter("hsv_lower").value)
        hsv_upper = tuple(int(v) for v in self.get_parameter("hsv_upper").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.object_frame_id = str(self.get_parameter("object_frame_id").value)

        self.detector = ObjectDetector(hsv_lower=hsv_lower, hsv_upper=hsv_upper)
        self.estimator = None
        self.depth_scale = depth_scale
        self.bridge = CvBridge()
        self.pose_pub = self.create_publisher(PoseStamped, "/object_pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        rgb_sub = Subscriber(self, Image, "/rgb/image_raw")
        depth_sub = Subscriber(self, Image, "/depth/image_raw")
        info_sub = Subscriber(self, CameraInfo, "/camera/camera_info")

        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.on_frames)

        self.get_logger().info(
            "Listening on /rgb/image_raw, /depth/image_raw, /camera/camera_info"
        )

    def on_frames(self, rgb_msg, depth_msg, info_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        intrinsics = camera_info_to_intrinsics(info_msg)
        if self.estimator is None:
            self.estimator = PoseEstimator(intrinsics, depth_scale=self.depth_scale)
        else:
            est = self.estimator.intrinsics
            if est.width != intrinsics.width or est.height != intrinsics.height:
                self.estimator = PoseEstimator(intrinsics, depth_scale=self.depth_scale)

        detection = self.detector.detect(rgb)
        if detection is None:
            return

        pose = self.estimator.estimate(
            depth,
            detection.center_px,
            mask=detection.mask,
        )
        if pose is None:
            return

        out = PoseStamped()
        out.header.stamp = rgb_msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.position.x = pose.x
        out.pose.position.y = pose.y
        out.pose.position.z = pose.z
        out.pose.orientation.w = 1.0
        self.pose_pub.publish(out)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = rgb_msg.header.stamp
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id = self.object_frame_id
        tf_msg.transform.translation.x = pose.x
        tf_msg.transform.translation.y = pose.y
        tf_msg.transform.translation.z = pose.z
        tf_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf_msg)


def main():
    rclpy.init()
    node = RgbdPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
