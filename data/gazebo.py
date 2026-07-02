# data/gazebo.py

import atexit
import base64
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

from data.frame import (
    CUBE_SIZE_M,
    HSV_RANGE,
    RGBDFrame,
    default_center_fn,
    default_intrinsics,
    display_path,
    has_real_sim_exports,
    optical_to_world,
    save_rgbd_frame,
)
from data.geometry import Pose3D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_CUBE = os.path.join(_ROOT, "assets", "cube")
WORLD_NAME = "cube_rgbd"
CUBE_MODEL = "colored_cube"
WORLD_FILE = os.path.join(_ASSETS_CUBE, "cube_rgbd.sdf")

_SIM_PROC = None


def gazebo_available():
    gz = shutil.which("gz")
    if not gz:
        return False, "Gazebo Sim (gz) CLI not on PATH"
    try:
        result = subprocess.run(
            [gz, "sim", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, "gz sim check failed: %s" % exc
    if result.returncode != 0:
        return False, "gz sim is installed but not runnable"
    return True, "Gazebo Sim available (%s)" % gz


def _run_gz(cmd, timeout=60.0, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def _list_topics():
    result = _run_gz(["gz", "topic", "-l"], timeout=15)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _find_topic(topics, *patterns):
    topic_list = list(topics)
    for pattern in patterns:
        regex = re.compile(pattern)
        for topic in topic_list:
            if regex.search(topic):
                return topic
    return None


def _echo_topic_json(topic, timeout=30.0):
    result = _run_gz(
        ["gz", "topic", "-e", "-t", topic, "-n", "1", "--json-output"],
        timeout=timeout,
    )
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("No data received from topic %s" % topic)
    return json.loads(stdout)


def _decode_gz_image(msg):
    width = int(msg["width"])
    height = int(msg["height"])
    raw = base64.b64decode(msg["data"])
    fmt = str(msg.get("pixelFormatType", msg.get("pixel_format_type", ""))).upper()

    if "RGB" in fmt or fmt in {"3", "PIXEL_FORMAT_RGB_INT8"}:
        channels = 3
        expected = width * height * channels
        if len(raw) < expected:
            raise ValueError("RGB payload too small: got %d, expected %d" % (len(raw), expected))
        return np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, channels).copy()

    if "R_FLOAT32" in fmt or "FLOAT32" in fmt or fmt in {"13", "PIXEL_FORMAT_R_FLOAT32"}:
        expected = width * height * 4
        if len(raw) < expected:
            raise ValueError("Depth payload too small: got %d, expected %d" % (len(raw), expected))
        depth_m = np.frombuffer(raw[:expected], dtype=np.float32).reshape(height, width)
        depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        return np.clip(depth_mm, 0, 65535).astype(np.uint16)

    if "L_INT16" in fmt or fmt in {"4", "PIXEL_FORMAT_L_INT16"}:
        expected = width * height * 2
        depth_raw = np.frombuffer(raw[:expected], dtype=np.int16).reshape(height, width)
        return np.clip(depth_raw.astype(np.float32), 0, 65535).astype(np.uint16)

    raise ValueError("Unsupported Gazebo image pixel format: %r" % fmt)


def _stop_gazebo():
    global _SIM_PROC
    if _SIM_PROC is None:
        return
    proc = _SIM_PROC
    _SIM_PROC = None
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _start_gazebo():
    global _SIM_PROC
    if _SIM_PROC is not None and _SIM_PROC.poll() is None:
        return

    if not os.path.isfile(WORLD_FILE):
        raise FileNotFoundError("Missing Gazebo world: %s" % WORLD_FILE)

    worlds_dir = os.path.abspath(_ASSETS_CUBE)
    env = dict(os.environ)
    existing = env.get("GZ_SIM_RESOURCE_PATH", "")
    env["GZ_SIM_RESOURCE_PATH"] = "%s:%s" % (worlds_dir, existing) if existing else worlds_dir

    _SIM_PROC = subprocess.Popen(
        ["gz", "sim", "-r", "-s", os.path.abspath(WORLD_FILE), "--headless-rendering", "-v", "1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    atexit.register(_stop_gazebo)

    deadline = time.time() + 90.0
    while time.time() < deadline:
        if _SIM_PROC.poll() is not None:
            stderr = (_SIM_PROC.stderr.read() if _SIM_PROC.stderr else "") or ""
            raise RuntimeError(
                "gz sim exited early (code %s): %s" % (_SIM_PROC.returncode, stderr[:500])
            )
        if _find_topic(_list_topics(), r"/rgbd/image$", r"rgbd.*image"):
            return
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for Gazebo RGB-D topics (is ogre2 rendering available?)")


def _set_cube_pose(world_xyz):
    x, y, z = world_xyz
    req = 'name: "%s", position: {x: %s, y: %s, z: %s}' % (CUBE_MODEL, x, y, z)
    _run_gz(
        [
            "gz", "service", "-s", "/world/%s/set_pose" % WORLD_NAME,
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000", "--req", req,
        ],
        timeout=15,
    )


def _sensor_topics():
    topics = _list_topics()
    rgb_topic = _find_topic(topics, r"/rgbd/image$", r"rgbd.*/image$")
    depth_topic = _find_topic(topics, r"/rgbd/depth_image$", r"rgbd.*/depth_image$")
    if not rgb_topic or not depth_topic:
        raise RuntimeError(
            "Could not find RGB-D topics. Available topics: %s"
            % (topics[:20] + ["..."] if len(topics) > 20 else topics)
        )
    return rgb_topic, depth_topic


def _capture_frame(center_optical):
    rgb_topic, depth_topic = _sensor_topics()
    _run_gz(["gz", "world", "-w", WORLD_NAME, "-s", "--iterations", "5"], timeout=20, check=False)
    time.sleep(0.35)

    rgb = _decode_gz_image(_echo_topic_json(rgb_topic))
    depth = _decode_gz_image(_echo_topic_json(depth_topic))

    intrinsics = default_intrinsics()
    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        gt_pose=Pose3D(x=center_optical[0], y=center_optical[1], z=center_optical[2]),
        metadata={
            "source": "gazebo",
            "cube_center_m": list(center_optical),
            "cube_size_m": CUBE_SIZE_M,
            "hsv_range": HSV_RANGE,
            "gazebo_world": WORLD_NAME,
            "gazebo_model": CUBE_MODEL,
        },
        source="gazebo",
    )


def export_gazebo(out_dir, num_frames=5, center_fn=None, force=False):
    out_dir = os.path.abspath(out_dir)
    ok, msg = gazebo_available()
    if not ok:
        raise RuntimeError(msg)

    if not force and has_real_sim_exports(out_dir, min_frames=num_frames):
        print("Reusing existing Gazebo exports in %s" % display_path(out_dir))
        return out_dir

    center_fn = center_fn or default_center_fn
    os.makedirs(out_dir, exist_ok=True)

    try:
        _start_gazebo()
        rgb_topic, depth_topic = _sensor_topics()
        print("Gazebo topics: rgb=%s, depth=%s" % (rgb_topic, depth_topic))

        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            center_optical = center_fn(t)
            world_xyz = optical_to_world(center_optical)
            _set_cube_pose(world_xyz)
            frame = _capture_frame(center_optical)
            save_rgbd_frame(frame, out_dir, stem="frame_%03d" % i)
            print("  frame_%03d: gt_optical=%s world=%s" % (i, center_optical, world_xyz))
    finally:
        _stop_gazebo()

    manifest = {
        "exporter": "data/gazebo.py",
        "backend": "gazebo",
        "num_frames": num_frames,
        "world": os.path.basename(WORLD_FILE),
        "source": "gazebo",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2))
    print("Exported %d Gazebo frames to %s" % (num_frames, display_path(out_dir)))
    return out_dir
