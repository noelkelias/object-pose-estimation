# evals/run_evals.py

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from evals.metrics import evaluate_scenario
from rgbd_pose.detection import ObjectDetector
from rgbd_pose.pose import PoseEstimator
from data.synthetic import generate_sequence
from data.frame import display_path

SCENARIOS = ["clean", "rgb_noise", "depth_noise", "occlusion"]


def run_all_evals(num_frames=20, scenarios=None):
    scenarios = scenarios or SCENARIOS
    detector = ObjectDetector(hsv_lower=(100, 80, 80), hsv_upper=(130, 255, 255))

    rows = []
    for scenario in scenarios:
        frames_raw = generate_sequence(num_frames=num_frames, scenario=scenario)
        frames = [
            {"rgb": f.rgb, "depth": f.depth, "gt_pose": f.gt_pose}
            for f in frames_raw
        ]
        pose_est = PoseEstimator(frames_raw[0].intrinsics, depth_scale=0.001)
        metrics = evaluate_scenario(scenario, frames, detector, pose_est)
        rows.append(metrics.as_dict())

    df = pd.DataFrame(rows)
    df["mean_error_mm"] = df["mean_error_m"] * 1000
    df["detection_rate_pct"] = (df["detection_rate"] * 100).round(1)
    df["fps"] = df["fps"].round(1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run RGB-D pose evaluations")

    parser.add_argument(
        "--num_frames",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--out",
        type=str,
        default="evals/results/eval_results.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    df = run_all_evals(num_frames=args.num_frames)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df[["scenario", "detection_rate_pct", "mean_error_mm", "fps"]].to_string(index=False))

    print(f"\nSaved: {display_path(os.path.abspath(args.out), PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
