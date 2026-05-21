#!/usr/bin/env python3
"""Collect aligned AVP and CyberGlove samples into visualization-ready HDF5.

The output follows the key structure used by
``data/InterHand_retargeted_30fps_biased.hdf5``:

* ``world_hand_joints``: AVP hand joints, treated as the clean target.
* ``world_hand_joints_true``: same AVP joints, for paired-pose consumers.
* ``world_hand_joints_biased``: glove-derived hand joints, treated as input.
* ``mano_angles``, ``betas``, and ``mano_side``: compatibility fields needed by
  ``visualization/visualize_keypoint_recon.py``.

Press Enter once to start recording, then press Enter again to stop and quit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pickle
import signal
import sys
import threading
import time
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WEBXR_TO_SMPLX = [
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
]


@dataclass(frozen=True)
class CollectorConfig:
    avp_ip: str
    output_dir: Path
    hand_model_path: Path
    glove_model_path: Path
    mano_model_dir: Path
    left_glove_port: Optional[str]
    right_glove_port: Optional[str]
    sides: tuple[str, ...]
    frame_rate: int = 30
    avp_record: bool = True
    calibrate_glove: bool = True
    glove_inference_method: str = "model"
    mano_fit_iters: int = 200
    mano_device: Optional[str] = None
    avp_valid_timeout_s: float = 10.0


class FrameRateLimiter:
    """Sleep-based fixed-rate loop using monotonic time."""

    def __init__(self, frame_rate: float) -> None:
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive.")
        self.period = 1.0 / frame_rate
        self.next_tick = time.perf_counter()

    def wait(self) -> None:
        now = time.perf_counter()
        if now < self.next_tick:
            time.sleep(self.next_tick - now)
            now = time.perf_counter()

        if now - self.next_tick > self.period:
            skipped = math.floor((now - self.next_tick) / self.period)
            self.next_tick += skipped * self.period

        self.next_tick += self.period


class GloveJointProjector:
    """Project one CyberGlove raw packet to 21 hand joints without opening a UI."""

    def __init__(self, hand_model_path: Path) -> None:
        from open_cyber_glove.utils import DEFAULT_GT_ORDER, forward_kinematics

        self.default_gt_order = DEFAULT_GT_ORDER
        self.forward_kinematics = forward_kinematics
        with hand_model_path.open("rb") as f:
            self.hand_model = pickle.load(f)

    def angles_to_joints(self, angles: np.ndarray, side: str) -> np.ndarray:
        angle_dict: dict[str, dict[str, dict[str, float]]] = {}
        for i, angle_name in enumerate(self.default_gt_order):
            finger, joint, dof = angle_name.split("_")
            angle_dict.setdefault(finger, {}).setdefault(joint, {})[dof] = angles[i]

        joints, _ = self.forward_kinematics(
            self.hand_model[side],
            angle_dict,
            side,
            np,
        )
        return (np.asarray(joints, dtype=np.float32) / 1000.0).astype(np.float32)


def _nan_hand_joints() -> np.ndarray:
    return np.full((21, 3), np.nan, dtype=np.float32)


def _nan_angles() -> np.ndarray:
    return np.full((22,), np.nan, dtype=np.float32)


def _nan_vec(length: int) -> np.ndarray:
    return np.full((length,), np.nan, dtype=np.float32)


def _nan_transform() -> np.ndarray:
    return np.full((1, 4, 4), np.nan, dtype=np.float32)


def _to_float_scalar(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_transform(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return _nan_transform()

    if arr.shape != (1, 4, 4):
        return _nan_transform()
    return arr


def _extract_avp_world_hand_joints(
    data: Mapping[str, Any], side: str
) -> tuple[np.ndarray, bool]:
    finger_key = f"{side}_fingers"
    wrist_key = f"{side}_wrist"

    try:
        finger_data = np.asarray(data[finger_key], dtype=np.float32)
        wrist_data = np.asarray(data[wrist_key], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return _nan_hand_joints(), False

    if finger_data.shape != (25, 4, 4) or wrist_data.shape != (1, 4, 4):
        return _nan_hand_joints(), False
    if not np.isfinite(finger_data).all() or not np.isfinite(wrist_data).all():
        return _nan_hand_joints(), False

    local_finger_joints_xyz = finger_data[:, :3, 3]
    global_wrist_xyz = wrist_data[0, :3, 3]
    global_joints_xyz = local_finger_joints_xyz + global_wrist_xyz
    return global_joints_xyz[WEBXR_TO_SMPLX].astype(np.float32), True


def patch_avp_stream_empty_hands() -> None:
    """Let avp_stream keep frames where the unneeded hand has no skeleton."""
    import avp_stream.streamer as streamer_module

    original_process_matrix = streamer_module.process_matrix

    def safe_process_matrices(
        skeleton: Any, matrix: np.ndarray = np.eye(4)
    ) -> np.ndarray:
        matrices = list(skeleton)
        if not matrices:
            return np.full((25, 4, 4), np.nan, dtype=np.float32)
        return np.concatenate(
            [matrix @ original_process_matrix(joint) for joint in matrices],
            axis=0,
        )

    def safe_get_pinch_distance(finger_messages: Any) -> float:
        fingers = safe_process_matrices(finger_messages)
        if fingers.shape[0] <= 9 or not np.isfinite(fingers[[4, 9]]).all():
            return float("nan")
        thumb = fingers[4, :3, 3]
        index = fingers[9, :3, 3]
        return float(np.linalg.norm(thumb - index))

    streamer_module.process_matrices = safe_process_matrices
    streamer_module.get_pinch_distance = safe_get_pinch_distance


def wait_for_valid_avp(
    streamer: Any, sides: tuple[str, ...], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    last_valid: dict[str, bool] = {side: False for side in sides}
    while time.monotonic() < deadline:
        sample = _sample_avp(streamer, sides)
        last_valid = {side: bool(sample[side]["valid"]) for side in sides}
        if all(last_valid.values()):
            return
        time.sleep(0.1)

    raise RuntimeError(
        "Timed out waiting for valid AVP hand joints for selected side(s) "
        f"{list(sides)}. Last valid flags: {last_valid}. "
        "Make sure the VisionOS app is started, hand tracking permission is "
        "enabled, and the selected hand is visible to the headset."
    )


def _sample_avp(streamer: Any, sides: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {"head": _nan_transform(), "error": ""}
    for side in sides:
        payload[side] = {
            "world_hand_joints": _nan_hand_joints(),
            "valid": False,
            "wrist": _nan_transform(),
            "pinch_distance": float("nan"),
            "wrist_roll": float("nan"),
        }

    try:
        data = streamer.latest
    except Exception as exc:
        payload["error"] = repr(exc)
        return payload

    if not isinstance(data, Mapping) or not data:
        payload["error"] = "empty_or_non_mapping_latest"
        return payload

    payload["head"] = _to_transform(data.get("head"))
    for side in sides:
        joints, valid = _extract_avp_world_hand_joints(data, side)
        payload[side] = {
            "world_hand_joints": joints,
            "valid": valid,
            "wrist": _to_transform(data.get(f"{side}_wrist")),
            "pinch_distance": _to_float_scalar(data.get(f"{side}_pinch_distance")),
            "wrist_roll": _to_float_scalar(data.get(f"{side}_wrist_roll")),
        }
    return payload


def _as_float_array(value: Any, length: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return _nan_vec(length)
    if arr.shape != (length,):
        return _nan_vec(length)
    return arr


def _sample_glove_side(
    glove_sdk: Any,
    projector: GloveJointProjector,
    side: str,
    method: str,
) -> dict[str, Any]:
    glove = getattr(glove_sdk, f"{side}_glove", None)
    if glove is None:
        return {
            "world_hand_joints": _nan_hand_joints(),
            "angles": _nan_angles(),
            "tensile_data": _nan_vec(19),
            "acc_data": _nan_vec(3),
            "gyro_data": _nan_vec(3),
            "mag_data": _nan_vec(3),
            "temperature": float("nan"),
            "device_timestamp": -1,
            "valid": False,
            "error": "not_configured",
        }

    try:
        raw = glove_sdk.get_data(hand_type=side)
        angles = np.asarray(
            glove.inference(raw, method=method, model=glove_sdk.model),
            dtype=np.float32,
        )
        joints = projector.angles_to_joints(angles, side)
    except Exception as exc:
        return {
            "world_hand_joints": _nan_hand_joints(),
            "angles": _nan_angles(),
            "tensile_data": _nan_vec(19),
            "acc_data": _nan_vec(3),
            "gyro_data": _nan_vec(3),
            "mag_data": _nan_vec(3),
            "temperature": float("nan"),
            "device_timestamp": -1,
            "valid": False,
            "error": repr(exc),
        }

    return {
        "world_hand_joints": joints,
        "angles": angles.astype(np.float32),
        "tensile_data": _as_float_array(raw.tensile_data, 19),
        "acc_data": _as_float_array(raw.acc_data, 3),
        "gyro_data": _as_float_array(raw.gyro_data, 3),
        "mag_data": _as_float_array(raw.mag_data, 3),
        "temperature": float(raw.temperature),
        "device_timestamp": int(raw.timestamp),
        "valid": True,
        "error": "",
    }


def _sample_glove(
    glove_sdk: Any,
    projector: GloveJointProjector,
    sides: tuple[str, ...],
    method: str,
) -> dict[str, Any]:
    return {
        side: _sample_glove_side(glove_sdk, projector, side, method)
        for side in sides
    }


class HDF5AlignedWriter:
    """Write aligned frames after MANO fitting completes."""

    def __init__(self, session_dir: Path, config: CollectorConfig) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise SystemExit(
                "h5py is required to write visualization-ready HDF5 output. "
                "Install project dependencies before collecting."
            ) from exc

        self.h5py = h5py
        self.path = session_dir / "aligned_avp_glove.hdf5"
        self.config = config
        self.buffer: list[dict[str, Any]] = []
        self.group_id_by_side = {
            side: str(index) for index, side in enumerate(config.sides)
        }

    def append(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)

    @staticmethod
    def _finite_joints(joints: np.ndarray) -> np.ndarray:
        return np.isfinite(joints).all(axis=(1, 2))

    def _write_side_group(
        self,
        h5_file: Any,
        side: str,
        records: list[dict[str, Any]],
    ) -> None:
        from calibration_net.teleop.mano_fitter import ManoSequenceFitter

        group = h5_file.create_group(self.group_id_by_side[side])
        string_dtype = self.h5py.string_dtype(encoding="utf-8")
        group.create_dataset("mano_side", data=np.asarray([side], dtype=string_dtype))
        group.attrs["side"] = side
        group.attrs["world_hand_joints_source"] = "avp"
        group.attrs["world_hand_joints_biased_source"] = "cyber_glove_fk"
        group.attrs["mano_angles_source"] = "mano_sequence_fitter_avp"
        group.attrs["glove_model_path"] = str(self.config.glove_model_path)
        group.attrs["hand_model_path"] = str(self.config.hand_model_path)
        group.attrs["mano_model_dir"] = str(self.config.mano_model_dir)
        group.attrs["mano_fit_iters"] = self.config.mano_fit_iters

        avp = [record["avp"][side] for record in records]
        glove = [record["glove"][side] for record in records]
        avp_joints = np.stack([item["world_hand_joints"] for item in avp]).astype(
            np.float32
        )
        glove_joints = np.stack(
            [item["world_hand_joints"] for item in glove]
        ).astype(np.float32)
        avp_valid = np.asarray([item["valid"] for item in avp], dtype=bool)
        glove_valid = np.asarray([item["valid"] for item in glove], dtype=bool)
        keep = (
            avp_valid
            & glove_valid
            & self._finite_joints(avp_joints)
            & self._finite_joints(glove_joints)
        )

        if not np.any(keep):
            raise RuntimeError(
                f"No valid aligned AVP/glove frames for side '{side}'. "
                "Check AVP tracking and glove inference before saving."
            )

        dropped = int((~keep).sum())
        if dropped:
            print(f"Dropping {dropped} invalid frames for {side} before MANO fitting.")

        kept_records = [record for record, valid in zip(records, keep) if valid]
        avp = [item for item, valid in zip(avp, keep) if valid]
        glove = [item for item, valid in zip(glove, keep) if valid]
        avp_joints = avp_joints[keep]
        glove_joints = glove_joints[keep]

        fitter_kwargs: dict[str, Any] = {}
        if self.config.mano_device is not None:
            fitter_kwargs["device"] = self.config.mano_device

        print(
            f"Fitting MANO for {side}: {avp_joints.shape[0]} frames, "
            f"{self.config.mano_fit_iters} iterations."
        )
        fitter = ManoSequenceFitter(
            mano_model_dir=str(self.config.mano_model_dir),
            mano_side=side,
            use_pca=False,
            ncomps=45,
            iters=self.config.mano_fit_iters,
            w_joint=1.0,
            w_pose_prior=0.001,
            w_shape_prior=0.01,
            w_trans_smooth=0.1,
            w_pose_smooth=0.01,
            **fitter_kwargs,
        )
        fit = fitter.fit_sequence(avp_joints, optimize_scale=False)
        mano_angles = np.asarray(fit["hand_pose"], dtype=np.float32)
        betas = np.asarray(fit["betas"], dtype=np.float32)
        group.create_dataset("betas", data=betas)

        batch = {
            "world_hand_joints": avp_joints,
            "world_hand_joints_true": avp_joints,
            "world_hand_joints_biased": glove_joints,
            "mano_angles": mano_angles,
            "timestamp_unix_s": np.asarray(
                [record["timestamp_unix_s"] for record in kept_records],
                dtype=np.float64,
            ),
            "timestamp_perf_s": np.asarray(
                [record["timestamp_perf_s"] for record in kept_records],
                dtype=np.float64,
            ),
            "sample_start_perf_s": np.asarray(
                [record["sample_start_perf_s"] for record in kept_records],
                dtype=np.float64,
            ),
            "sample_end_perf_s": np.asarray(
                [record["sample_end_perf_s"] for record in kept_records],
                dtype=np.float64,
            ),
            "glove_angles": np.stack([item["angles"] for item in glove]).astype(
                np.float32
            ),
            "glove_tensile_data": np.stack(
                [item["tensile_data"] for item in glove]
            ).astype(np.float32),
            "glove_acc_data": np.stack([item["acc_data"] for item in glove]).astype(
                np.float32
            ),
            "glove_gyro_data": np.stack([item["gyro_data"] for item in glove]).astype(
                np.float32
            ),
            "glove_mag_data": np.stack([item["mag_data"] for item in glove]).astype(
                np.float32
            ),
            "glove_temperature": np.asarray(
                [item["temperature"] for item in glove], dtype=np.float32
            ),
            "glove_device_timestamp": np.asarray(
                [item["device_timestamp"] for item in glove], dtype=np.int64
            ),
        }

        for name, data in batch.items():
            group.create_dataset(name, data=data)

    def save(self) -> Optional[Path]:
        if not self.buffer:
            return None

        with self.h5py.File(self.path, "w") as h5_file:
            h5_file.attrs["collector"] = "aligned_avp_glove"
            h5_file.attrs["schema_version"] = 1
            h5_file.attrs["frame_rate"] = self.config.frame_rate
            h5_file.attrs["avp_ip"] = self.config.avp_ip
            h5_file.attrs["glove_data_role"] = "world_hand_joints_biased"
            h5_file.attrs["avp_data_role"] = "world_hand_joints"
            for side in self.config.sides:
                self._write_side_group(h5_file, side, self.buffer)
        return self.path


def _parse_sides(value: str, ports: Mapping[str, Optional[str]]) -> tuple[str, ...]:
    if value == "auto":
        sides = tuple(side for side in ("left", "right") if ports.get(side))
        if not sides:
            raise ValueError("At least one glove port is required.")
        return sides
    if value == "both":
        missing = [side for side in ("left", "right") if not ports.get(side)]
        if missing:
            raise ValueError("--sides both requires both glove ports.")
        return ("left", "right")
    if not ports.get(value):
        raise ValueError(f"--sides {value} requires --{value}-glove-port.")
    return (value,)


def parse_args() -> CollectorConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Collect aligned AVP and CyberGlove data into an HDF5 file "
            "compatible with visualization/visualize_keypoint_recon.py."
        )
    )
    parser.add_argument("--avp-ip", required=True, help="Vision Pro IP address.")
    parser.add_argument(
        "--output-dir",
        default="data/aligned_avp_glove",
        help="Root directory for timestamped session output.",
    )
    parser.add_argument(
        "--hand-model-path",
        required=True,
        help="OpenCyberGlove hand_model.pkl used to convert glove angles to joints.",
    )
    parser.add_argument(
        "--glove-model-path",
        required=True,
        help="OpenCyberGlove ONNX model used for tensile-to-angle inference.",
    )
    parser.add_argument(
        "--mano-model-dir",
        default=os.environ.get("MANO_PATH", ""),
        help="MANO model directory. Defaults to MANO_PATH.",
    )
    parser.add_argument(
        "--left-glove-port", default=None, help="Left glove serial port."
    )
    parser.add_argument(
        "--right-glove-port", default=None, help="Right glove serial port."
    )
    parser.add_argument(
        "--sides",
        choices=("auto", "left", "right", "both"),
        default="auto",
        help="Hand side(s) to save. Default auto uses the provided glove ports.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="Target aligned sampling rate in Hz.",
    )
    parser.add_argument(
        "--no-avp-record",
        action="store_true",
        help="Pass record=False to VisionProStreamer.",
    )
    glove_calibration = parser.add_mutually_exclusive_group()
    glove_calibration.add_argument(
        "--calibrate-glove",
        dest="calibrate_glove",
        action="store_true",
        default=True,
        help="Run OpenCyberGlove calibration before recording. Enabled by default.",
    )
    glove_calibration.add_argument(
        "--no-calibrate-glove",
        dest="calibrate_glove",
        action="store_false",
        help="Skip OpenCyberGlove calibration. Use only if avg_val was set elsewhere.",
    )
    parser.add_argument(
        "--mano-fit-iters",
        type=int,
        default=200,
        help="Optimization iterations for final MANO fitting.",
    )
    parser.add_argument(
        "--mano-device",
        default=None,
        help="Torch device for MANO fitting, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--avp-valid-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for valid AVP joints for the selected side(s).",
    )
    args = parser.parse_args()

    if args.frame_rate <= 0:
        parser.error("--frame-rate must be positive.")
    if args.mano_fit_iters <= 0:
        parser.error("--mano-fit-iters must be positive.")
    if args.avp_valid_timeout_s <= 0:
        parser.error("--avp-valid-timeout-s must be positive.")

    hand_model_path = Path(args.hand_model_path).expanduser()
    glove_model_path = Path(args.glove_model_path).expanduser()
    if not args.mano_model_dir:
        parser.error("--mano-model-dir is required unless MANO_PATH is set.")
    mano_model_dir = Path(args.mano_model_dir).expanduser()
    if not hand_model_path.exists():
        parser.error(f"--hand-model-path does not exist: {hand_model_path}")
    if not glove_model_path.exists():
        parser.error(f"--glove-model-path does not exist: {glove_model_path}")
    if not mano_model_dir.exists():
        parser.error(
            "--mano-model-dir does not exist. Pass it explicitly or set MANO_PATH: "
            f"{mano_model_dir}"
        )

    ports = {
        "left": args.left_glove_port,
        "right": args.right_glove_port,
    }
    try:
        sides = _parse_sides(args.sides, ports)
    except ValueError as exc:
        parser.error(str(exc))

    return CollectorConfig(
        avp_ip=args.avp_ip,
        output_dir=Path(args.output_dir).expanduser(),
        hand_model_path=hand_model_path,
        glove_model_path=glove_model_path,
        mano_model_dir=mano_model_dir,
        left_glove_port=args.left_glove_port,
        right_glove_port=args.right_glove_port,
        sides=sides,
        frame_rate=args.frame_rate,
        avp_record=not args.no_avp_record,
        calibrate_glove=args.calibrate_glove,
        mano_fit_iters=args.mano_fit_iters,
        mano_device=args.mano_device,
        avp_valid_timeout_s=args.avp_valid_timeout_s,
    )


def create_session_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / timestamp
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _json_ready_config(config: CollectorConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["hand_model_path"] = str(config.hand_model_path)
    data["glove_model_path"] = str(config.glove_model_path)
    data["mano_model_dir"] = str(config.mano_model_dir)
    data["sides"] = list(config.sides)
    return data


def write_metadata(
    session_dir: Path,
    config: CollectorConfig,
    state: str,
    hdf5_path: Optional[Path],
) -> None:
    metadata = {
        "schema_version": 1,
        "collector": "aligned_avp_glove",
        "state": state,
        "updated_unix_s": time.time(),
        "hdf5_path": str(hdf5_path) if hdf5_path is not None else None,
        "config": _json_ready_config(config),
    }
    with (session_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _request_enter_to_stop(stop_event: threading.Event) -> None:
    try:
        input("Recording. Press Enter to stop and quit...")
    finally:
        stop_event.set()


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
            "Install it with `pip install avp_stream`."
        ) from exc

    try:
        from calibration_net.utils import initilize_glove
    except ImportError as exc:
        raise SystemExit(
            "Cannot import calibration_net.utils.initilize_glove. "
            "Run this script from the Calibration_net environment."
        ) from exc
    matplotlib_cache = Path("/private/tmp/calibration_net_matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        from calibration_net.teleop.mano_fitter import ManoSequenceFitter  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Cannot import the MANO sequence fitter. Install manopth/MANO "
            "dependencies before collecting this dataset."
        ) from exc

    session_dir = create_session_dir(config.output_dir)
    writer = HDF5AlignedWriter(session_dir=session_dir, config=config)
    write_metadata(session_dir, config, state="created", hdf5_path=writer.path)
    print(f"Session directory: {session_dir}")
    print(f"HDF5 output: {writer.path}")

    print("Initializing AVP streamer...")
    streamer = VisionProStreamer(ip=config.avp_ip, record=config.avp_record)
    try:
        wait_for_valid_avp(streamer, config.sides, config.avp_valid_timeout_s)
    except BaseException:
        write_metadata(session_dir, config, state="error", hdf5_path=None)
        _close_device(streamer)
        raise

    print("Initializing glove SDK...")
    try:
        glove_sdk = initilize_glove(
            left_port=config.left_glove_port,
            right_port=config.right_glove_port,
            model_path=str(config.glove_model_path),
            calibrate=config.calibrate_glove,
        )
    except BaseException:
        write_metadata(session_dir, config, state="error", hdf5_path=None)
        _close_device(streamer)
        raise
    projector = GloveJointProjector(config.hand_model_path)

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}. Flushing and exiting...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"Sides: {', '.join(config.sides)}")
    print(f"Target rate: {config.frame_rate} Hz.")
    print("HDF5 will be written after recording, once MANO fitting completes.")
    input("Press Enter to start recording...")
    write_metadata(session_dir, config, state="recording", hdf5_path=writer.path)

    stop_thread = threading.Thread(
        target=_request_enter_to_stop, args=(stop_event,), daemon=True
    )
    stop_thread.start()

    limiter = FrameRateLimiter(config.frame_rate)
    frame_index = 0
    started_unix_s = time.time()
    started_perf_s = time.perf_counter()

    try:
        while not stop_event.is_set():
            limiter.wait()
            sample_start_perf_s = time.perf_counter()
            avp_payload = _sample_avp(streamer, config.sides)
            glove_payload = _sample_glove(
                glove_sdk,
                projector,
                config.sides,
                config.glove_inference_method,
            )
            sample_end_perf_s = time.perf_counter()
            midpoint_perf_s = 0.5 * (sample_start_perf_s + sample_end_perf_s)

            writer.append(
                {
                    "frame_index": frame_index,
                    "timestamp_perf_s": midpoint_perf_s,
                    "timestamp_unix_s": started_unix_s
                    + (midpoint_perf_s - started_perf_s),
                    "sample_start_perf_s": sample_start_perf_s,
                    "sample_end_perf_s": sample_end_perf_s,
                    "avp": avp_payload,
                    "glove": glove_payload,
                }
            )
            frame_index += 1

            if frame_index % max(config.frame_rate, 1) == 0:
                elapsed = time.perf_counter() - started_perf_s
                print(
                    f"Collected {frame_index} aligned frames "
                    f"({elapsed:.1f}s elapsed)",
                    end="\r",
                )
    finally:
        saved_path: Optional[Path] = None
        save_error: Optional[BaseException] = None
        try:
            saved_path = writer.save()
        except BaseException as exc:
            save_error = exc
        final_state = "finished" if save_error is None else "error"
        write_metadata(session_dir, config, state=final_state, hdf5_path=saved_path)
        _close_device(streamer)
        _close_device(glove_sdk)
        print(f"\nFinished. Total aligned frames: {frame_index}")
        if saved_path is not None:
            print(f"Visualization-ready HDF5: {saved_path}")
        else:
            print("No frames were collected; no HDF5 was written.")
        if save_error is not None:
            raise save_error


if __name__ == "__main__":
    main()
