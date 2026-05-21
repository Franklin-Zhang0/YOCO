#!/usr/bin/env python3
"""One-time AVP collector for the four glove calibration GT poses.

For each pose, press Enter after matching the requested hand shape. The script
captures a short burst of valid AVP hand keypoints, averages them, and saves:

* ``avp_calib_gt_poses.npz``: numeric arrays for downstream scripts;
* ``avp_calib_gt_poses.json``: human-readable metadata and keypoints;
* ``avp_calib_gt_presets.py``: ready-to-copy preset dictionaries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration_net.utils import transform_joint3d as joint_3d_world_to_local
from scripts.collect_aligned_avp_glove import (
    _sample_avp,
    patch_avp_stream_empty_hands,
    wait_for_valid_avp,
)


POSE_ORDER = ("pinch", "fully_open", "fist", "little_pinch")


@dataclass(frozen=True)
class AvpCalibPoseConfig:
    avp_ip: str
    side: str
    output_dir: Path
    samples_per_pose: int
    sample_interval_s: float
    pose_timeout_s: float
    avp_valid_timeout_s: float
    avp_record: bool


def parse_args() -> AvpCalibPoseConfig:
    parser = argparse.ArgumentParser(
        description="Collect four AVP GT keypoint poses for glove calibration."
    )
    parser.add_argument("--avp-ip", required=True, help="Vision Pro IP address.")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="right",
        help="AVP hand side to snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/avp_calib_gt_poses",
        help="Root directory for timestamped output.",
    )
    parser.add_argument(
        "--samples-per-pose",
        type=int,
        default=5,
        help="Valid AVP frames to average after each Enter press.",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=0.03,
        help="Delay between frames in the per-pose burst.",
    )
    parser.add_argument(
        "--pose-timeout-s",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for the per-pose burst.",
    )
    parser.add_argument(
        "--avp-valid-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for initially valid AVP joints.",
    )
    parser.add_argument(
        "--no-avp-record",
        action="store_true",
        help="Pass record=False to VisionProStreamer.",
    )
    args = parser.parse_args()

    if args.samples_per_pose <= 0:
        parser.error("--samples-per-pose must be positive.")
    if args.sample_interval_s < 0:
        parser.error("--sample-interval-s must be non-negative.")
    if args.pose_timeout_s <= 0:
        parser.error("--pose-timeout-s must be positive.")
    if args.avp_valid_timeout_s <= 0:
        parser.error("--avp-valid-timeout-s must be positive.")

    return AvpCalibPoseConfig(
        avp_ip=args.avp_ip,
        side=args.side,
        output_dir=Path(args.output_dir).expanduser(),
        samples_per_pose=args.samples_per_pose,
        sample_interval_s=args.sample_interval_s,
        pose_timeout_s=args.pose_timeout_s,
        avp_valid_timeout_s=args.avp_valid_timeout_s,
        avp_record=not args.no_avp_record,
    )


def create_session_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / timestamp
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _finite_hand_joints(joints: np.ndarray) -> bool:
    return joints.shape == (21, 3) and bool(np.isfinite(joints).all())


def _to_local(joints: np.ndarray) -> np.ndarray:
    local = joint_3d_world_to_local(np.asarray(joints, dtype=np.float32))
    return np.asarray(local, dtype=np.float32)


def capture_pose(
    streamer: Any,
    pose_name: str,
    config: AvpCalibPoseConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    input(
        f"[{pose_name}] Match this pose with your {config.side} hand, "
        "then press Enter to snapshot..."
    )

    samples = []
    timestamps = []
    deadline = time.monotonic() + config.pose_timeout_s
    while len(samples) < config.samples_per_pose and time.monotonic() < deadline:
        sample = _sample_avp(streamer, (config.side,))
        side_payload = sample[config.side]
        joints = np.asarray(side_payload["world_hand_joints"], dtype=np.float32)
        if bool(side_payload["valid"]) and _finite_hand_joints(joints):
            samples.append(joints)
            timestamps.append(time.time())
        if len(samples) < config.samples_per_pose:
            time.sleep(config.sample_interval_s)

    if len(samples) < config.samples_per_pose:
        raise RuntimeError(
            f"Only collected {len(samples)} valid frame(s) for pose "
            f"'{pose_name}', expected {config.samples_per_pose}. Make sure "
            "the selected hand is visible to the headset and tracking is valid."
        )

    burst = np.stack(samples).astype(np.float32)
    averaged = burst.mean(axis=0).astype(np.float32)
    metadata = {
        "num_frames": int(burst.shape[0]),
        "timestamp_unix_s_first": float(timestamps[0]),
        "timestamp_unix_s_last": float(timestamps[-1]),
        "per_frame_std_m": burst.std(axis=0).astype(np.float32).tolist(),
        "mean_per_joint_std_m": float(
            np.linalg.norm(burst.std(axis=0), axis=-1).mean()
        ),
    }
    return averaged, metadata


def _json_ready_config(config: AvpCalibPoseConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    return data


def _format_numpy_dict(name: str, pose_names: tuple[str, ...], data: np.ndarray) -> str:
    lines = [f"{name} = {{"]
    for pose_name, joints in zip(pose_names, data):
        array_text = np.array2string(
            joints,
            precision=6,
            separator=", ",
            suppress_small=False,
        )
        lines.append(f'    "{pose_name}": np.array(')
        lines.append(f"{array_text},")
        lines.append("        dtype=np.float32,")
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def write_outputs(
    session_dir: Path,
    config: AvpCalibPoseConfig,
    world_joints: np.ndarray,
    local_joints: np.ndarray,
    pose_metadata: dict[str, Any],
) -> None:
    pose_names = np.asarray(POSE_ORDER)
    npz_path = session_dir / "avp_calib_gt_poses.npz"
    json_path = session_dir / "avp_calib_gt_poses.json"
    presets_path = session_dir / "avp_calib_gt_presets.py"

    np.savez(
        npz_path,
        pose_names=pose_names,
        world_hand_joints=world_joints,
        local_hand_joints=local_joints,
        side=np.asarray([config.side]),
    )

    payload = {
        "schema_version": 1,
        "collector": "avp_calib_gt_poses",
        "pose_order": list(POSE_ORDER),
        "side": config.side,
        "config": _json_ready_config(config),
        "pose_metadata": pose_metadata,
        "world_hand_joints": {
            pose_name: joints.tolist()
            for pose_name, joints in zip(POSE_ORDER, world_joints)
        },
        "local_hand_joints": {
            pose_name: joints.tolist()
            for pose_name, joints in zip(POSE_ORDER, local_joints)
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    presets_text = "\n\n".join(
        [
            "import numpy as np",
            f"POSE_ORDER = {POSE_ORDER!r}",
            _format_numpy_dict(
                "PRESET_CALIBRATION_KEYPOINTS_WORLD",
                POSE_ORDER,
                world_joints,
            ),
            _format_numpy_dict(
                "PRESET_CALIBRATION_KEYPOINTS_LOCAL",
                POSE_ORDER,
                local_joints,
            ),
        ]
    )
    presets_path.write_text(presets_text + "\n", encoding="utf-8")

    print(f"Wrote numeric arrays: {npz_path}")
    print(f"Wrote JSON metadata:  {json_path}")
    print(f"Wrote Python presets: {presets_path}")


def _close_device(device: Any) -> None:
    for method_name in ("stop", "close", "shutdown"):
        method = getattr(device, method_name, None)
        if callable(method):
            method()
            break


def main() -> None:
    config = parse_args()

    try:
        patch_avp_stream_empty_hands()
        from avp_stream import VisionProStreamer
    except ImportError as exc:
        raise SystemExit(
            "avp_stream is not installed in the current Python environment. "
            "Install it in the teleop environment before collecting."
        ) from exc

    session_dir = create_session_dir(config.output_dir)
    print(f"Session directory: {session_dir}")
    print("Initializing AVP streamer...")
    streamer = VisionProStreamer(ip=config.avp_ip, record=config.avp_record)

    try:
        wait_for_valid_avp(
            streamer,
            (config.side,),
            timeout_s=config.avp_valid_timeout_s,
        )
        print(
            f"AVP {config.side} hand is valid. "
            f"Collecting poses in order: {', '.join(POSE_ORDER)}"
        )

        world_by_pose = []
        pose_metadata = {}
        for pose_name in POSE_ORDER:
            joints, metadata = capture_pose(streamer, pose_name, config)
            world_by_pose.append(joints)
            pose_metadata[pose_name] = metadata
            print(
                f"Captured {pose_name}: "
                f"mean per-joint std = {metadata['mean_per_joint_std_m']:.6f} m"
            )

        world_joints = np.stack(world_by_pose).astype(np.float32)
        local_joints = _to_local(world_joints).astype(np.float32)
        write_outputs(
            session_dir=session_dir,
            config=config,
            world_joints=world_joints,
            local_joints=local_joints,
            pose_metadata=pose_metadata,
        )
    finally:
        _close_device(streamer)


if __name__ == "__main__":
    main()
