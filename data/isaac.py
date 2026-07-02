# data/isaac.py

import json
import os

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


def isaac_available():
    try:
        from omni.isaac.kit import SimulationApp  # noqa: F401

        return True, "omni.isaac.kit available"
    except ImportError:
        pass
    try:
        from isaacsim import SimulationApp  # noqa: F401

        return True, "isaacsim package available"
    except ImportError:
        pass
    try:
        import isaaclab  # noqa: F401

        return True, "isaaclab available (use omni.isaac.kit python for export)"
    except ImportError:
        pass
    return False, "Isaac Sim not installed"


def _create_sim_app(headless=True):
    config = {"headless": headless, "width": 640, "height": 480}
    try:
        from omni.isaac.kit import SimulationApp

        return SimulationApp(config), "omni.isaac.kit"
    except ImportError:
        from isaacsim import SimulationApp

        return SimulationApp(config), "isaacsim"


def _depth_to_uint16_mm(depth):
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth_m = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)


def _warmup_world(world, camera, steps=20):
    for _ in range(steps):
        world.step(render=True)
    for _ in range(5):
        world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is not None and rgba.size > 0:
            return


def _capture_frame(world, camera, cube, center_optical):
    world_xyz = optical_to_world(center_optical)
    cube.set_world_pose(position=np.array(world_xyz, dtype=np.float64))

    for _ in range(8):
        world.step(render=True)

    rgba = camera.get_rgba()
    if rgba is None or rgba.size == 0:
        raise RuntimeError("Isaac camera returned empty RGBA buffer after warmup")

    rgb = np.ascontiguousarray(rgba[:, :, :3].astype(np.uint8))
    depth = _depth_to_uint16_mm(camera.get_depth())

    intrinsics = default_intrinsics()
    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        gt_pose=Pose3D(x=center_optical[0], y=center_optical[1], z=center_optical[2]),
        metadata={
            "source": "isaac_sim",
            "cube_center_m": list(center_optical),
            "cube_size_m": CUBE_SIZE_M,
            "hsv_range": HSV_RANGE,
            "isaac_cube_prim": "/World/colored_cube",
            "isaac_camera_prim": "/World/rgbd_camera",
        },
        source="isaac_sim",
    )


def _run_export(out_dir, num_frames, center_fn):
    simulation_app, kit_label = _create_sim_app(headless=True)
    print("Isaac kit: %s" % kit_label)

    try:
        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.sensor import Camera
        import omni.isaac.core.utils.numpy.rotations as rot_utils

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        camera = Camera(
            prim_path="/World/rgbd_camera",
            position=np.array([0.0, 0.0, 0.0]),
            frequency=15,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0.0, 90.0, 90.0]), degrees=True
            ),
        )
        world.scene.add(camera)

        cube = world.scene.add(
            VisualCuboid(
                prim_path="/World/colored_cube",
                name="colored_cube",
                position=np.array(optical_to_world(center_fn(0.0))),
                scale=np.array([0.08, 0.08, 0.08]),
                color=np.array([0.31, 0.31, 0.86]),
            )
        )

        world.reset()
        camera.initialize()
        _warmup_world(world, camera)

        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            center_optical = center_fn(t)
            frame = _capture_frame(world, camera, cube, center_optical)
            save_rgbd_frame(frame, out_dir, stem="frame_%03d" % i)
            world_xyz = optical_to_world(center_optical)
            print("  frame_%03d: gt_optical=%s world=%s" % (i, center_optical, world_xyz))
    finally:
        simulation_app.close()


def export_isaac(out_dir, num_frames=5, center_fn=None, force=False):
    out_dir = os.path.abspath(out_dir)
    ok, msg = isaac_available()
    if not ok:
        raise RuntimeError(msg)

    if not force and has_real_sim_exports(out_dir, min_frames=num_frames):
        print("Reusing existing Isaac exports in %s" % display_path(out_dir))
        return out_dir

    center_fn = center_fn or default_center_fn
    os.makedirs(out_dir, exist_ok=True)

    _run_export(out_dir, num_frames, center_fn)

    manifest = {
        "exporter": "data/isaac.py",
        "backend": "isaac",
        "num_frames": num_frames,
        "source": "isaac_sim",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2))
    print("Exported %d Isaac frames to %s" % (num_frames, display_path(out_dir)))
    return out_dir
