cmake_minimum_required(VERSION 3.8)
project(object_pose)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(pose_latency_monitor src/pose_latency_monitor.cpp)
ament_target_dependencies(pose_latency_monitor rclcpp geometry_msgs)

install(TARGETS pose_latency_monitor DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/demo_rgbd_publisher.py
  scripts/rgbd_pose_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
