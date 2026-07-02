#!/usr/bin/env bash
# Build and run the RGB-D pose pipeline in Docker.
#
#   ./ros2/run_demo.sh              # headless: latency monitor logs (~5 s)
#   ./ros2/run_demo.sh --viz        # record RViz TF view to assets/demo/
#   ./ros2/run_demo.sh --viz --interactive   # live RViz (needs XQuartz on macOS)
#
# Requires Docker. No host ROS2 install needed (works on macOS + Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${ROS2_IMAGE:-object-pose-ros2:humble}"
VIZ=0
INTERACTIVE=0

for arg in "$@"; do
  case "$arg" in
    --viz) VIZ=1 ;;
    --interactive) INTERACTIVE=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --viz or --interactive)" >&2
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop, then re-run: ./ros2/run_demo.sh" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image $IMAGE (first run only)..."
  docker build -t "$IMAGE" -f "$ROOT/ros2/Dockerfile" "$ROOT/ros2"
fi

DOCKER_ARGS=(--rm -t -v "$ROOT:/ws" -w /ws/ros2 -e PYTHONPATH=/ws -e PROJECT_ROOT=/ws)

if [[ "$VIZ" -eq 1 && "$INTERACTIVE" -eq 1 ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    export DISPLAY="${DISPLAY:-host.docker.internal:0}"
    echo "macOS interactive RViz: start XQuartz, then run: xhost +localhost"
  else
    export DISPLAY="${DISPLAY:-:0}"
    DOCKER_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
  fi
  DOCKER_ARGS+=(-e DISPLAY -it)
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc "
  set -e
  source /opt/ros/humble/setup.bash
  colcon build --packages-select object_pose
  source install/setup.bash

  RVIZ_CONFIG=/ws/ros2/install/object_pose/share/object_pose/config/demo.rviz
  OUT_DIR=/ws/assets/demo
  mkdir -p \"\$OUT_DIR\"

  ros2 run object_pose demo_rgbd_publisher.py &
  PUB_PID=\$!
  ros2 run object_pose rgbd_pose_node.py &
  POSE_PID=\$!

  cleanup() {
    kill \"\$PUB_PID\" \"\$POSE_PID\" 2>/dev/null || true
  }
  trap cleanup EXIT

  if [[ $VIZ -eq 0 ]]; then
    sleep 2
    timeout 5 ros2 run object_pose pose_latency_monitor 2>&1 \
      | sed 's/\x1b\[[0-9;]*m//g' \
      | tee /ws/assets/demo/ros2_monitor_sample.log || true
    exit 0
  fi

  sleep 2

  if [[ $INTERACTIVE -eq 1 ]]; then
    echo 'RViz: fixed frame camera_optical_frame; object_frame = detected cube.'
    rviz2 -d \"\$RVIZ_CONFIG\"
    exit 0
  fi

  # Headless RViz capture via virtual framebuffer (no host display needed).
  export DISPLAY=:99
  export LIBGL_ALWAYS_SOFTWARE=1
  Xvfb :99 -screen 0 1280x800x24 >/tmp/xvfb.log 2>&1 &
  XVFB_PID=\$!
  sleep 1

  rviz2 -d \"\$RVIZ_CONFIG\" >/tmp/rviz.log 2>&1 &
  RVIZ_PID=\$!
  sleep 2.5

  VIDEO=\"\$OUT_DIR/ros2_rviz_demo.mp4\"
  PNG=\"\$OUT_DIR/ros2_rviz_demo.png\"
  GIF=\"\$OUT_DIR/ros2_rviz_demo.gif\"
  RECORD_SEC=4

  ffmpeg -y -loglevel error \
    -video_size 1280x800 -framerate 10 -f x11grab -i :99.0 \
    -t \$RECORD_SEC -c:v libx264 -pix_fmt yuv420p \"\$VIDEO\"

  ffmpeg -y -loglevel error \
    -video_size 1280x800 -f x11grab -i :99.0 \
    -vframes 1 \"\$PNG\"

  PALETTE=/tmp/ros2_palette.png
  ffmpeg -y -loglevel error -i \"\$VIDEO\" \
    -vf \"fps=8,scale=720:-1:flags=lanczos,palettegen=stats_mode=diff\" \"\$PALETTE\"
  ffmpeg -y -loglevel error -i \"\$VIDEO\" -i \"\$PALETTE\" \
    -lavfi \"fps=8,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer\" \"\$GIF\"

  kill \"\$RVIZ_PID\" \"\$XVFB_PID\" 2>/dev/null || true
  echo \"Saved RViz recording: \$VIDEO (\${RECORD_SEC}s)\"
  echo \"Saved RViz screenshot: \$PNG\"
  echo \"Saved RViz GIF: \$GIF\"
"

if [[ "$VIZ" -eq 1 && "$INTERACTIVE" -eq 0 ]]; then
  echo ""
  echo "Open the recording:"
  echo "  $ROOT/assets/demo/ros2_rviz_demo.gif   (README embed)"
  echo "  $ROOT/assets/demo/ros2_rviz_demo.mp4"
  echo "  $ROOT/assets/demo/ros2_rviz_demo.png"
fi
