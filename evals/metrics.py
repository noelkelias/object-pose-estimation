# evals/metrics.py

import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rgbd_pose.detection import DetectionResult, ObjectDetector
from data.geometry import Pose3D
from rgbd_pose.pose import PoseEstimator


# =========================
# METRICS
# =========================

class FrameResult:

    def __init__(self, detected, error_m, latency_s):
        self.detected = detected
        self.error_m = error_m
        self.latency_s = latency_s


class ScenarioMetrics:

    def __init__(
        self,
        name,
        detection_rate,
        mean_error_m,
        median_error_m,
        fps,
        num_frames,
        frame_results=None,
    ):
        self.name = name
        self.detection_rate = detection_rate
        self.mean_error_m = mean_error_m
        self.median_error_m = median_error_m
        self.fps = fps
        self.num_frames = num_frames
        self.frame_results = frame_results or []

    def as_dict(self):
        return {
            "scenario": self.name,
            "detection_rate": self.detection_rate,
            "mean_error_m": self.mean_error_m,
            "median_error_m": self.median_error_m,
            "fps": self.fps,
            "num_frames": self.num_frames,
        }


def compute_metrics(results, scenario_name="default"):
    n = len(results)
    if n == 0:
        return ScenarioMetrics(
            name=scenario_name,
            detection_rate=0.0,
            mean_error_m=float("nan"),
            median_error_m=float("nan"),
            fps=0.0,
            num_frames=0,
        )

    detected = [r for r in results if r.detected]
    detection_rate = len(detected) / n

    errors = [r.error_m for r in detected if r.error_m is not None]
    mean_error = float(np.mean(errors)) if errors else float("nan")
    median_error = float(np.median(errors)) if errors else float("nan")

    total_time = sum(r.latency_s for r in results)
    fps = n / total_time if total_time > 0 else 0.0

    return ScenarioMetrics(
        name=scenario_name,
        detection_rate=detection_rate,
        mean_error_m=mean_error,
        median_error_m=median_error,
        fps=fps,
        num_frames=n,
        frame_results=results,
    )


def evaluate_scenario(name, frames, detector, pose_estimator, perturb_fn=None):
    results = []

    for frame in frames:
        if perturb_fn is not None:
            frame = perturb_fn(frame)

        t0 = time.perf_counter()
        detection = detector.detect(frame["rgb"])
        pose = None
        if detection is not None:
            pose = pose_estimator.estimate(
                frame["depth"], detection.center_px, mask=detection.mask
            )

        latency = time.perf_counter() - t0

        if detection is None or pose is None:
            results.append(FrameResult(detected=False, error_m=None, latency_s=latency))
            continue

        gt = frame["gt_pose"]
        error_m = pose.distance_to(gt)
        results.append(FrameResult(detected=True, error_m=error_m, latency_s=latency))

    return compute_metrics(results, scenario_name=name)


# =========================
# VISUALIZATION
# =========================

def draw_detection(rgb, detection, label="object"):
    vis = rgb.copy()
    if detection is None:
        cv2.putText(
            vis,
            "NO DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
        return vis

    x, y, w, h = detection.bbox
    cx, cy = detection.center_px

    overlay = vis.copy()
    overlay[detection.mask > 0] = (
        overlay[detection.mask > 0] * 0.5 + np.array([0, 180, 255]) * 0.5
    ).astype(np.uint8)
    vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawMarker(vis, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        vis,
        f"{label} ({cx},{cy})",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def draw_3d_error(rgb, estimated, ground_truth, error_m=None):
    vis = rgb.copy()
    gt_text = f"GT:  ({ground_truth.x:.3f}, {ground_truth.y:.3f}, {ground_truth.z:.3f}) m"

    if estimated is None:
        est_text = "Est: (---, ---, ---) m"
        err_text = "Error: N/A"
        color = (255, 80, 80)
    else:
        est_text = f"Est: ({estimated.x:.3f}, {estimated.y:.3f}, {estimated.z:.3f}) m"
        if error_m is None:
            error_m = estimated.distance_to(ground_truth)
        err_text = f"Error: {error_m * 1000:.1f} mm"
        color = (80, 255, 80) if error_m < 0.05 else (255, 200, 80)

    lines = [est_text, gt_text, err_text]
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (20, vis.shape[0] - 80 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color if i == 2 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def show_rgb_depth(rgb, depth, title="RGB-D Frame", figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    depth_vis = depth.astype(np.float32)
    depth_vis[depth_vis == 0] = np.nan
    im = axes[1].imshow(depth_vis, cmap="turbo")
    axes[1].set_title("Depth (raw units)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_mask_point_cloud(
    rgb,
    detection,
    pose_estimator,
    depth,
    estimated,
    ground_truth,
    error_m=None,
    path=None,
    figsize=(12, 4.5),
):
    """Side-by-side mask overlay and zoomed 3D point cloud."""
    if detection is None or estimated is None:
        raise ValueError("detection and estimated pose are required")

    points = pose_estimator.point_cloud_points(depth, detection.mask)
    if points is None:
        raise ValueError("could not build point cloud from mask")

    if error_m is None:
        error_m = estimated.distance_to(ground_truth)

    bg = "#1a1a2e"
    fig = plt.figure(figsize=figsize, facecolor=bg)
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d", facecolor=bg)

    vis = draw_detection(rgb, detection)
    ax2d.imshow(vis)
    ax2d.set_title("HSV mask overlay", color="white", fontsize=10)
    ax2d.axis("off")

    z = points[:, 2]
    ax3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=z,
        cmap="turbo",
        s=3,
        alpha=0.9,
        depthshade=True,
    )

    ref = np.vstack([points, estimated.as_array(), ground_truth.as_array()])
    center = ref.mean(axis=0)
    span = max((ref.max(axis=0) - ref.min(axis=0)).max() / 2.0, 0.02)
    ax3d.set_xlim(center[0] - span, center[0] + span)
    ax3d.set_ylim(center[1] - span, center[1] + span)
    ax3d.set_zlim(center[2] - span, center[2] + span)

    ax3d.scatter(
        [estimated.x],
        [estimated.y],
        [estimated.z],
        c="#39ff14",
        s=18,
        depthshade=False,
        label="estimated",
    )
    ax3d.scatter(
        [ground_truth.x],
        [ground_truth.y],
        [ground_truth.z],
        c="#ff4d4d",
        s=18,
        marker="x",
        linewidths=1.2,
        depthshade=False,
        label="ground truth",
    )
    ax3d.text(
        estimated.x,
        estimated.y,
        estimated.z,
        f"  {error_m * 1000:.2f} mm",
        color="white",
        fontsize=8,
    )

    ax3d.set_title("Masked Open3D point cloud", color="white", fontsize=10)
    ax3d.set_xlabel("X (m)", color="white", fontsize=8)
    ax3d.set_ylabel("Y (m)", color="white", fontsize=8)
    ax3d.set_zlabel("Z (m)", color="white", fontsize=8)
    ax3d.tick_params(colors="white", labelsize=7)
    ax3d.view_init(elev=24, azim=-56)
    ax3d.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="white", labelcolor="white")

    try:
        ax3d.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    fig.suptitle(
        "OpenCV segmentation + Open3D centroid",
        color="white",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg)
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()
