# data/frame.py

import json
import os

import cv2
import numpy as np

from data.geometry import CameraIntrinsics, Pose3D

SYNTHETIC_SOURCES = frozenset(
    {
        "synthetic_fallback",
        "synthetic_export",
        "isaac_sim_stub",
        "gazebo_stub",
    }
)

CUBE_SIZE_M = 0.08
CUBE_COLOR_RGB = (80, 80, 220)
HSV_RANGE = ((100, 80, 80), (130, 255, 255))

_OPTICAL_TO_WORLD = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

ISAAC_EXPORTS = "isaac_exports"
GAZEBO_EXPORTS = "gazebo_exports"
SYNTHETIC_EXPORTS = "synthetic_exports"


class RGBDFrame:

    def __init__(self, rgb, depth, intrinsics, gt_pose, metadata, source="", stem=""):
        self.rgb = rgb
        self.depth = depth
        self.intrinsics = intrinsics
        self.gt_pose = gt_pose
        self.metadata = metadata
        self.source = source or str(metadata.get("source", ""))
        self.stem = stem


def default_intrinsics(width=640, height=480, fx=525.0, fy=525.0):
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


def optical_to_world(center_m):
    p = _OPTICAL_TO_WORLD @ np.asarray(center_m, dtype=np.float64)
    return float(p[0]), float(p[1]), float(p[2])


def default_center_fn(t):
    return (0.03 * t, 0.02 * (1.0 - t), 0.57 + 0.05 * t)


def is_real_sim_metadata(metadata):
    source = str(metadata.get("source", ""))
    return bool(source) and source not in SYNTHETIC_SOURCES


def export_dir(project_root, name):
    return os.path.join(os.path.abspath(project_root), "data", name)


def export_search_dirs(project_root):
    project_root = os.path.abspath(project_root)
    return (
        export_dir(project_root, ISAAC_EXPORTS),
        export_dir(project_root, GAZEBO_EXPORTS),
        export_dir(project_root, SYNTHETIC_EXPORTS),
    )


def has_sim_exports(frame_dir):
    if not os.path.isdir(frame_dir):
        return False
    names = os.listdir(frame_dir)
    has_rgb = any(n.endswith("_rgb.png") for n in names)
    has_meta = any(n.endswith("_meta.json") for n in names)
    return has_rgb and has_meta


def count_sim_exports(frame_dir):
    if not os.path.isdir(frame_dir):
        return 0
    return sum(1 for name in os.listdir(frame_dir) if name.endswith("_rgb.png"))


def has_real_sim_exports(frame_dir, min_frames=1):
    if not os.path.isdir(frame_dir):
        return False

    stems = set()
    for name in os.listdir(frame_dir):
        if name.endswith("_rgb.png"):
            stems.add(name.replace("_rgb.png", ""))

    real = 0
    for stem in sorted(stems):
        meta_path = os.path.join(frame_dir, f"{stem}_meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            continue
        if is_real_sim_metadata(meta):
            real += 1
    return real >= min_frames


def _load_meta(meta_path):
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_rgbd_frame(frame_dir, stem="frame_000"):
    rgb_path = os.path.join(frame_dir, f"{stem}_rgb.png")
    depth_path = os.path.join(frame_dir, f"{stem}_depth.png")
    meta_path = os.path.join(frame_dir, f"{stem}_meta.json")

    if not all(os.path.isfile(p) for p in (rgb_path, depth_path, meta_path)):
        return None

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    meta = _load_meta(meta_path)

    intr = meta.get("intrinsics", {})
    intrinsics = CameraIntrinsics(
        fx=float(intr.get("fx", 525.0)),
        fy=float(intr.get("fy", 525.0)),
        cx=float(intr.get("cx", rgb.shape[1] / 2)),
        cy=float(intr.get("cy", rgb.shape[0] / 2)),
        width=int(intr.get("width", rgb.shape[1])),
        height=int(intr.get("height", rgb.shape[0])),
    )

    gt = meta.get("gt_pose_m", [0.0, 0.0, 0.6])
    gt_pose = Pose3D(float(gt[0]), float(gt[1]), float(gt[2]))
    source = str(meta.get("source", os.path.basename(frame_dir)))

    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        gt_pose=gt_pose,
        metadata=meta,
        source=source,
        stem=stem,
    )


def load_rgbd_sequence(frame_dir, max_frames=10):
    if not os.path.isdir(frame_dir):
        return []

    stems = sorted(
        name.replace("_rgb.png", "")
        for name in os.listdir(frame_dir)
        if name.endswith("_rgb.png")
    )
    frames = []
    for stem in stems[:max_frames]:
        frame = load_rgbd_frame(frame_dir, stem=stem)
        if frame is not None:
            frames.append(frame)
    return frames


def save_rgbd_frame(frame, out_dir, stem="frame_000"):
    os.makedirs(out_dir, exist_ok=True)

    rgb_path = os.path.join(out_dir, f"{stem}_rgb.png")
    depth_path = os.path.join(out_dir, f"{stem}_depth.png")
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")

    cv2.imwrite(rgb_path, cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, frame.depth)

    meta = {
        **frame.metadata,
        "gt_pose_m": frame.gt_pose.as_array().tolist(),
        "intrinsics": {
            "fx": frame.intrinsics.fx,
            "fy": frame.intrinsics.fy,
            "cx": frame.intrinsics.cx,
            "cy": frame.intrinsics.cy,
            "width": frame.intrinsics.width,
            "height": frame.intrinsics.height,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def display_path(path, project_root=None):
    path = os.path.abspath(path)
    project_root = os.path.abspath(project_root or _ROOT)
    try:
        rel = os.path.relpath(path, project_root)
        label = os.path.basename(project_root)
        if rel == ".":
            return label
        return f"{label}/{rel.replace(os.sep, '/')}"
    except ValueError:
        return os.path.basename(path)


def _synthetic_exports_dir(out_dir):
    data_dir = os.path.dirname(os.path.abspath(out_dir))
    return os.path.join(data_dir, SYNTHETIC_EXPORTS)


def _describe_export_dir(label, frame_dir, project_root, *, sim_slot=False):
    path = display_path(frame_dir, project_root)
    count = count_sim_exports(frame_dir)
    if count == 0:
        if sim_slot:
            backend = label.split("_", 1)[0]
            return f"{label}: empty (export via notebooks/data.ipynb when {backend} runtime is available)"
        return f"{label}: empty (run notebooks/data.ipynb)"
    if sim_slot and has_real_sim_exports(frame_dir, min_frames=1):
        return f"{label}: {count} real frames ({path})"
    if sim_slot:
        return f"{label}: {count} frames ({path})"
    return f"{label}: {count} bundled demo frames ({path})"


def describe_frame_sources(project_root):
    from data.gazebo import gazebo_available
    from data.isaac import isaac_available

    project_root = os.path.abspath(project_root)
    status = []

    gz_ok, gz_msg = gazebo_available()
    isaac_ok, isaac_msg = isaac_available()

    status.append(
        f"isaac_runtime: {'available' if isaac_ok else 'not available'} ({isaac_msg})"
    )
    status.append(f"gazebo_runtime: {'available' if gz_ok else 'not available'} ({gz_msg})")

    isaac_dir, gazebo_dir, synthetic_dir = export_search_dirs(project_root)
    status.append(_describe_export_dir(ISAAC_EXPORTS, isaac_dir, project_root, sim_slot=True))
    status.append(_describe_export_dir(GAZEBO_EXPORTS, gazebo_dir, project_root, sim_slot=True))
    status.append(_describe_export_dir(SYNTHETIC_EXPORTS, synthetic_dir, project_root))

    return status


def load_frame(project_root, stem="frame_000"):
    project_root = os.path.abspath(project_root)

    for frame_dir in export_search_dirs(project_root):
        frame = load_rgbd_frame(frame_dir, stem=stem)
        if frame is not None:
            return frame

    searched = ", ".join(display_path(d, project_root) for d in export_search_dirs(project_root))
    raise FileNotFoundError(
        f"No frame '{stem}' found. Searched: {searched}. "
        "Run notebooks/data.ipynb to export sim frames or regenerate data/synthetic_exports/."
    )


def export_sim_frames(backend, out_dir, num_frames=5, force=False):
    from data import synthetic as synthetic_mod
    from data.gazebo import export_gazebo, gazebo_available
    from data.isaac import export_isaac, isaac_available

    out_dir = os.path.abspath(out_dir)
    synthetic_dir = _synthetic_exports_dir(out_dir)

    if backend == "synthetic":
        print(f"Export backend: synthetic ({display_path(out_dir)})")
        return synthetic_mod.export_frames(out_dir, num_frames, "synthetic_export")

    if backend == "gazebo":
        ok, msg = gazebo_available()
        print(f"Gazebo check: {msg}")
        if ok:
            try:
                return export_gazebo(out_dir, num_frames=num_frames, force=force)
            except Exception as exc:
                print(f"Gazebo export failed: {exc}")
                print(f"Falling back to synthetic in {display_path(synthetic_dir)}.")
                return synthetic_mod.export_frames(synthetic_dir, num_frames, "synthetic_export")
        print(f"Gazebo not available. Writing synthetic to {display_path(synthetic_dir)}.")
        return synthetic_mod.export_frames(synthetic_dir, num_frames, "synthetic_export")

    from notebooks.colab_utils import check_gpu, try_isaac_import

    print(f"GPU check: {check_gpu()}")
    ok, msg = try_isaac_import()
    print(f"Isaac check: {msg}")
    if ok or isaac_available()[0]:
        try:
            return export_isaac(out_dir, num_frames=num_frames, force=force)
        except Exception as exc:
            print(f"Isaac export failed: {exc}")
            print(f"Falling back to synthetic in {display_path(synthetic_dir)}.")
            return synthetic_mod.export_frames(synthetic_dir, num_frames, "synthetic_export")
    print(f"Isaac not available. Writing synthetic to {display_path(synthetic_dir)}.")
    return synthetic_mod.export_frames(synthetic_dir, num_frames, "synthetic_export")
