/**
 * ROS2 subscriber: monitors /object_pose latency and FPS.
 *
 *   ros2 run object_pose pose_latency_monitor
 *
 * Local demo (Docker, no system ROS2 install):
 *   ./ros2/run_demo.sh
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class PoseLatencyMonitor : public rclcpp::Node {
 public:
  PoseLatencyMonitor() : Node("pose_latency_monitor") {
    sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/object_pose", rclcpp::SensorDataQoS(),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          this->on_pose(msg);
        });

    RCLCPP_INFO(get_logger(), "Listening on /object_pose");
  }

 private:
  void on_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    const auto now = this->now();
    const rclcpp::Time stamp(msg->header.stamp);

    double latency_ms = (now - stamp).seconds() * 1000.0;

    if (last_wall_time_.nanoseconds() > 0) {
      const double dt = (now - last_wall_time_).seconds();
      if (dt > 0.0) {
        const double instant_fps = 1.0 / dt;
        fps_ema_ = (fps_ema_ <= 0.0) ? instant_fps : 0.9 * fps_ema_ + 0.1 * instant_fps;
      }
    }
    last_wall_time_ = now;
    ++count_;

    const auto& p = msg->pose.position;
    RCLCPP_INFO(
        get_logger(),
        "[%zu] pos=(%.3f, %.3f, %.3f) latency=%.1f ms fps~=%.1f frame=%s",
        count_, p.x, p.y, p.z, latency_ms, fps_ema_,
        msg->header.frame_id.c_str());
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  rclcpp::Time last_wall_time_{0, 0, RCL_ROS_TIME};
  double fps_ema_{0.0};
  size_t count_{0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseLatencyMonitor>());
  rclcpp::shutdown();
  return 0;
}
