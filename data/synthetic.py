# data/synthetic.py

import os

import numpy as np

from data.frame import (
    CUBE_COLOR_RGB,
    CUBE_SIZE_M,
    HSV_RANGE,
    RGBDFrame,
    default_intrinsics,
    display_path,
    save_rgbd_frame,
)
from data.geometry import Pose3D


def render_frame(
    cube_center_m=(0.0, 0.0, 0.6),
    add_noise=False,
    rgb_noise_std=8.0,
    depth_noise_mm=5.0,
    occlusion_fraction=0.0,
    width=640,
    height=480,
    fx=525.0,
    fy=525.0,
    seed=42,
):
    intrinsics = default_intrinsics(width=width, height=height, fx=fx, fy=fy)
    rng = np.random.default_rng(seed)

    rgb = np.full((height, width, 3), 45, dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint16)

    for v in range(height):
        shade = int(30 + 40 * v / height)
        rgb[v, :, 1] = shade
        rgb[v, :, 2] = shade // 2

    cx, cy, cz = cube_center_m
    half = CUBE_SIZE_M / 2.0
    gt_pose = Pose3D(cx, cy, cz)

    u_c = int(round(fx * cx / cz + intrinsics.cx)) if cz > 0.01 else -1
    v_c = int(round(fy * cy / cz + intrinsics.cy)) if cz > 0.01 else -1
    if u_c >= 0 and v_c >= 0 and cz > 0.01:
        half_u = max(6, int(round(fx * half / cz)))
        half_v = max(6, int(round(fy * half / cz)))
        x0 = max(0, u_c - half_u)
        x1 = min(width, u_c + half_u + 1)
        y0 = max(0, v_c - half_v)
        y1 = min(height, v_c + half_v + 1)

        depth_mm = int(round(cz * 1000))
        rgb[y0:y1, x0:x1] = CUBE_COLOR_RGB
        depth[y0:y1, x0:x1] = depth_mm

    if occlusion_fraction > 0 and v_c >= 0:
        occ_h = max(8, int(height * occlusion_fraction))
        v0 = max(0, v_c - occ_h // 2)
        v1 = min(height, v0 + occ_h)
        rgb[v0:v1, :] = (20, 20, 20)
        depth[v0:v1, :] = 0

    if add_noise:
        noise = rng.normal(0, rgb_noise_std, rgb.shape)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        valid = depth > 0
        d_noise = rng.normal(0, depth_noise_mm, size=valid.sum())
        depth_f = depth.astype(np.float32)
        depth_f[valid] = np.clip(depth_f[valid] + d_noise, 1, 65535)
        depth = depth_f.astype(np.uint16)

    metadata = {
        "source": "synthetic_fallback",
        "cube_center_m": [cx, cy, cz],
        "cube_size_m": CUBE_SIZE_M,
        "hsv_range": HSV_RANGE,
    }

    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        gt_pose=gt_pose,
        metadata=metadata,
        source="synthetic_fallback",
    )


def generate_sequence(num_frames=10, scenario="clean", seed=42):
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        center = (
            0.02 * np.sin(2 * np.pi * t),
            0.015 * np.cos(2 * np.pi * t),
            0.58 + 0.04 * t,
        )

        kwargs = {"seed": seed + i}
        if scenario == "rgb_noise":
            kwargs.update({"add_noise": True, "rgb_noise_std": 15.0})
        elif scenario == "depth_noise":
            kwargs.update({"add_noise": True, "depth_noise_mm": 20.0})
        elif scenario == "occlusion":
            kwargs.update({"occlusion_fraction": 0.25})

        frames.append(render_frame(cube_center_m=center, **kwargs))
    return frames


def export_frames(out_dir, num_frames=5, source="synthetic_export"):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_frames):
        frame = render_frame()
        frame.metadata["source"] = source
        frame.source = source
        save_rgbd_frame(frame, out_dir, stem=f"frame_{i:03d}")
    print(f"Wrote {num_frames} synthetic frames to {display_path(out_dir)}")
    return out_dir
