#!/usr/bin/env python3
"""Paper-style calibration evaluation for paired hand keypoint datasets.

This script is a non-visual evaluation counterpart to
`visualization/visualize_keypoint_recon.py`. It evaluates the learned
calibration model and lightweight geometric baselines using N paired
calibration poses, then reports the metrics described in the experiment
section: MPJPE, pinch error, calibration time, and optional downstream task
fields when they are present in the HDF5.
"""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_net.policy.frozen_mano_canonicalizer import (
    decode_mano_side,
)
from calibration_net.policy.refine_inferencer import (
    RefineModelInferencer,
    resolve_torch_device,
)
from calibration_net.utils import (
    mano_recon,
    scale_hand_fingers,
    transform_joint3d,
)


TIP_INDICES = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
TIP_NAME_TO_INDEX = dict(zip(FINGER_NAMES, TIP_INDICES))
DEFAULT_PINCH_PAIRS = [
    "thumb-index",
    "thumb-middle",
    "thumb-ring",
    "thumb-pinky",
]
DEFAULT_METHODS = [
    "ours",
    "no_calib",
    "translation",
    "rigid",
    "similarity",
    "affine",
    "ridge",
]


@dataclass
class Args:
    dataset_path: str
    model_path: Optional[str] = None
    output_dir: str = "eval/results"

    # Paper default: fast deployment calibration with N=4 paired poses.
    num_calib_poses: int = 4

    # MethodName plus simple calibration baselines.
    methods: list[str] = field(default_factory=lambda: DEFAULT_METHODS.copy())

    # HDF5 keys. "auto" uses biased keypoints as input and true keypoints as target
    # when those columns exist.
    input_key: str = "auto"
    target_key: str = "auto"
    # Target key for the paired calibration poses consumed by the hypernetwork.
    # "target" preserves the old behavior and reuses target_key/gt_pose_source.
    conditioning_target_key: str = "target"
    gt_pose_source: str = "keypoint"
    baseline_space: str = "keypoint"
    baseline_theta_source: str = "predicted"
    input_theta_key: str = "mano_angles_biased"
    target_theta_key: str = "mano_angles"
    ridge_alpha: float = 1e-3

    # 0 means no limit.
    max_demos: int = 0
    max_aug_sets: int = 0
    max_frames_per_demo: int = 0

    # Calibration frame selection: uniform, first, or random.
    calib_selection: str = "random"
    # 0 preserves the old behavior. When >0, sample calibration poses only
    # from frame indices [0, calib_pose_index_range) and evaluate on the suffix.
    calib_pose_index_range: int = 0
    seed: int = 42
    device: Optional[str] = None
    side_filter: str = "all"
    fixed_noise: bool = True
    apply_left_shapedirs_fix: bool = True

    # The model predicts MANO hand pose without global pose. Local evaluation is
    # the default used by the visualizer and avoids mixing world-frame drift with
    # hand-pose quality.
    metric_frame: str = "local"

    # Pinching is counted on GT fingertip-pair distance below this threshold.
    pinch_threshold_m: float = 0.005
    pinch_pairs: list[str] = field(default_factory=lambda: DEFAULT_PINCH_PAIRS.copy())
    aucj_max_threshold_m: float = 0.05
    aucj_num_thresholds: int = 101
    # Applies only when the resolved keypoint GT is raw world_hand_joints.
    # Set <= 0 to disable point-level rejection.
    bad_annotation_threshold_m: float = 0.05

    exclude_calib_from_eval: bool = False
    write_predictions: bool = False


@dataclass
class Prediction:
    joints: np.ndarray
    vertices: Optional[np.ndarray] = None


def _sorted_hdf5_keys(data: h5py.File) -> list[str]:
    keys = list(data.keys())
    try:
        return sorted(keys, key=lambda key: int(key))
    except Exception:
        return sorted(keys)


def _read_array(group: h5py.Group, key: str) -> np.ndarray:
    return np.asarray(group[key])


def _decode_side(group: h5py.Group) -> str:
    if "mano_side" in group:
        return decode_mano_side(np.asarray(group["mano_side"]))
    if "mano_side" in group.attrs:
        return decode_mano_side(group.attrs["mano_side"])
    return "right"


def _resolve_key(
    group: h5py.Group,
    requested_key: str,
    candidates: list[str],
    *,
    allow_missing: bool = False,
) -> Optional[str]:
    if requested_key != "auto":
        if requested_key not in group:
            raise KeyError(
                f"Requested key '{requested_key}' is missing from group {group.name}."
            )
        return requested_key

    for key in candidates:
        if key in group:
            return key

    if allow_missing:
        return None
    raise KeyError(
        f"None of the candidate keys {candidates} exist in group {group.name}."
    )


def _ensure_sequence(array: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2 and array.shape == (21, 3):
        array = array[None]
    if array.ndim != 3 or array.shape[-2:] != (21, 3):
        raise ValueError(f"{name} must have shape (T, 21, 3), got {array.shape}.")
    return array


def _ensure_theta_sequence(array: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1 and array.shape[0] == 45:
        array = array[None]
    if array.ndim != 2 or array.shape[-1] != 45:
        raise ValueError(f"{name} must have shape (T, 45), got {array.shape}.")
    return array


def _select_aug_sample(array: np.ndarray, sample_idx: int, *, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4:
        return _ensure_sequence(array[sample_idx], name=name)
    return _ensure_sequence(array, name=name)


def _select_theta_sample(
    array: np.ndarray,
    sample_idx: int,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3:
        return _ensure_theta_sequence(array[sample_idx], name=name)
    return _ensure_theta_sequence(array, name=name)


def _sample_count(*arrays: np.ndarray) -> int:
    count = 1
    for array in arrays:
        if array.ndim == 4:
            count = max(count, int(array.shape[0]))
        elif array.ndim == 3 and array.shape[-2:] != (21, 3):
            count = max(count, int(array.shape[0]))
    return count


def _limit_sequence(
    input_seq: np.ndarray,
    target_seq: np.ndarray,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = min(input_seq.shape[0], target_seq.shape[0])
    if max_frames > 0:
        frame_count = min(frame_count, max_frames)
    return input_seq[:frame_count], target_seq[:frame_count]


def _expand_betas(
    group: h5py.Group,
    sample_idx: int,
    frame_count: int,
    device: torch.device,
) -> torch.Tensor:
    if "betas" not in group:
        return torch.zeros((frame_count, 10), dtype=torch.float32, device=device)

    betas = np.asarray(group["betas"], dtype=np.float32)
    if betas.ndim == 3:
        betas = betas[sample_idx]
    if betas.ndim == 1:
        betas = np.broadcast_to(betas[None], (frame_count, 10))
    elif betas.ndim == 2 and betas.shape[0] == 1:
        betas = np.broadcast_to(betas, (frame_count, 10))
    elif betas.ndim == 2 and betas.shape[0] >= frame_count:
        betas = betas[:frame_count]
    else:
        raise ValueError(f"Unsupported betas shape {betas.shape} in {group.name}.")
    betas = np.array(betas, dtype=np.float32, copy=True)
    return torch.as_tensor(betas, dtype=torch.float32, device=device)


def _load_optional_pose_array(
    group: h5py.Group,
    keys: list[str],
    sample_idx: int,
    frame_count: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    key = next((candidate for candidate in keys if candidate in group), None)
    if key is None:
        return None

    array = np.asarray(group[key], dtype=np.float32)
    if array.ndim == 3:
        array = array[sample_idx]
    if array.ndim == 1:
        array = np.broadcast_to(array[None], (frame_count, 3))
    elif array.ndim == 2 and array.shape[0] >= frame_count:
        array = array[:frame_count]
    else:
        raise ValueError(f"Unsupported {key} shape {array.shape} in {group.name}.")
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _reconstruct_mano_from_group(
    group: h5py.Group,
    side: str,
    sample_idx: int,
    frame_count: int,
    device: torch.device,
    apply_left_shapedirs_fix: bool,
) -> np.ndarray:
    if "mano_angles" not in group:
        raise KeyError(
            f"Group {group.name} has no target keypoint column and no mano_angles."
        )

    theta = np.asarray(group["mano_angles"], dtype=np.float32)
    if theta.ndim == 3:
        theta = theta[sample_idx]
    theta = theta[:frame_count]
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    betas_t = _expand_betas(group, sample_idx, frame_count, device)
    orient_t = _load_optional_pose_array(
        group,
        ["global_orient", "mano_orient"],
        sample_idx,
        frame_count,
        device,
    )
    transl_t = _load_optional_pose_array(
        group,
        ["trans", "transl"],
        sample_idx,
        frame_count,
        device,
    )

    with torch.no_grad():
        vertices, joints = mano_recon(
            theta_t,
            mano_orient=orient_t,
            transl=transl_t,
            side=side,
            betas=betas_t,
            apply_left_shapedirs_fix=apply_left_shapedirs_fix,
        )
    return vertices.detach().cpu().numpy(), joints.detach().cpu().numpy()


def _reconstruct_target_from_mano(
    group: h5py.Group,
    side: str,
    sample_idx: int,
    frame_count: int,
    device: torch.device,
    apply_left_shapedirs_fix: bool,
) -> np.ndarray:
    _, joints = _reconstruct_mano_from_group(
        group,
        side,
        sample_idx,
        frame_count,
        device,
        apply_left_shapedirs_fix,
    )
    return joints


def _metric_frame(joints: np.ndarray, frame: str) -> np.ndarray:
    if frame == "world":
        return np.asarray(joints, dtype=np.float32)
    if frame != "local":
        raise ValueError("metric_frame must be 'local' or 'world'.")

    local = transform_joint3d(np.asarray(joints, dtype=np.float32))
    if torch.is_tensor(local):
        return local.detach().cpu().numpy()
    return np.asarray(local, dtype=np.float32)


def _parse_pinch_pairs(pair_names: list[str]) -> list[tuple[str, int, int]]:
    pairs = []
    for name in pair_names:
        clean = name.lower().replace("_", "-")
        try:
            left, right = clean.split("-", 1)
            left_idx = TIP_NAME_TO_INDEX[left]
            right_idx = TIP_NAME_TO_INDEX[right]
        except Exception as exc:
            valid = ", ".join(
                f"{a}-{b}" for i, a in enumerate(FINGER_NAMES)
                for b in FINGER_NAMES[i + 1 :]
            )
            raise ValueError(
                f"Invalid pinch pair '{name}'. Valid pairs: {valid}"
            ) from exc
        pairs.append((clean, left_idx, right_idx))
    return pairs


def _select_calibration_indices(
    frame_count: int,
    num_calib_poses: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("Cannot select calibration poses from an empty sequence.")
    count = min(max(int(num_calib_poses), 1), frame_count)

    if mode == "first":
        return np.arange(count, dtype=np.int64)
    if mode == "random":
        return np.sort(rng.choice(frame_count, size=count, replace=False))
    if mode != "uniform":
        raise ValueError("calib_selection must be 'uniform', 'first', or 'random'.")

    if count == 1:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, frame_count - 1, num=count).round().astype(np.int64)


def _eval_mask(
    frame_count: int,
    calib_indices: np.ndarray,
    exclude: bool,
    eval_start_index: int = 0,
) -> np.ndarray:
    mask = np.ones(frame_count, dtype=bool)
    if eval_start_index > 0:
        mask[:eval_start_index] = False
    if exclude:
        mask[calib_indices] = False
    if not mask.any():
        raise ValueError("Selected calibration/evaluation split leaves no frames.")
    return mask


def _fit_translation(
    source: np.ndarray,
    target: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    feature_dim = source.shape[-1]
    offset = (target - source).reshape(-1, feature_dim).mean(axis=0)
    return lambda points: points + offset


def _rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    estimate_scale: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    feature_dim = source.shape[-1]
    src = source.reshape(-1, feature_dim).astype(np.float64)
    dst = target.reshape(-1, feature_dim).astype(np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = src_centered.T @ dst_centered
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt_mat[-1] *= -1.0
        singular_values[-1] *= -1.0
        rotation = vt_mat.T @ u_mat.T

    if estimate_scale:
        denom = np.sum(src_centered**2)
        scale = float(np.sum(singular_values) / denom) if denom > 1e-12 else 1.0
    else:
        scale = 1.0

    translation = dst_mean - scale * (src_mean @ rotation.T)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _fit_rigid(
    source: np.ndarray,
    target: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    scale, rotation, translation = _rigid_transform(
        source,
        target,
        estimate_scale=False,
    )
    return lambda points: scale * (points @ rotation.T) + translation


def _fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    scale, rotation, translation = _rigid_transform(
        source,
        target,
        estimate_scale=True,
    )
    return lambda points: scale * (points @ rotation.T) + translation


def _fit_affine(
    source: np.ndarray,
    target: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    feature_dim = source.shape[-1]
    src = source.reshape(-1, feature_dim).astype(np.float64)
    dst = target.reshape(-1, feature_dim).astype(np.float64)
    design = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    matrix, *_ = np.linalg.lstsq(design, dst, rcond=None)

    def apply(points: np.ndarray) -> np.ndarray:
        flat = points.reshape(-1, feature_dim)
        flat_design = np.concatenate(
            [flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)],
            axis=1,
        )
        return (flat_design @ matrix).reshape(points.shape).astype(np.float32)

    return apply


def _fit_ridge(
    source: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> Callable[[np.ndarray], np.ndarray]:
    if alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative.")

    feature_dim = source.shape[-1]
    src = source.reshape(-1, feature_dim).astype(np.float64)
    dst = target.reshape(-1, feature_dim).astype(np.float64)
    design = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    regularizer = alpha * np.eye(design.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    prior = np.zeros((design.shape[1], feature_dim), dtype=np.float64)
    prior[:feature_dim] = np.eye(feature_dim, dtype=np.float64)
    system = design.T @ design + regularizer
    rhs = design.T @ dst + regularizer @ prior
    try:
        matrix = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        matrix = np.linalg.pinv(system) @ rhs

    def apply(points: np.ndarray) -> np.ndarray:
        flat = points.reshape(-1, feature_dim)
        flat_design = np.concatenate(
            [flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)],
            axis=1,
        )
        return (flat_design @ matrix).reshape(points.shape).astype(np.float32)

    return apply


def _baseline_predict(
    method: str,
    input_seq: np.ndarray,
    target_seq: np.ndarray,
    calib_indices: np.ndarray,
    ridge_alpha: float,
) -> tuple[Prediction, float, float]:
    start = time.perf_counter()
    if method == "no_calib":
        predictor = lambda points: points.copy()
    elif method == "translation":
        predictor = _fit_translation(
            input_seq[calib_indices],
            target_seq[calib_indices],
        )
    elif method == "rigid":
        predictor = _fit_rigid(input_seq[calib_indices], target_seq[calib_indices])
    elif method == "similarity":
        predictor = _fit_similarity(input_seq[calib_indices], target_seq[calib_indices])
    elif method == "affine":
        predictor = _fit_affine(input_seq[calib_indices], target_seq[calib_indices])
    elif method == "ridge":
        predictor = _fit_ridge(
            input_seq[calib_indices],
            target_seq[calib_indices],
            ridge_alpha,
        )
    else:
        raise ValueError(f"Unknown baseline method '{method}'.")
    calibration_time = time.perf_counter() - start

    start = time.perf_counter()
    prediction = predictor(input_seq)
    inference_time = time.perf_counter() - start
    return (
        Prediction(joints=prediction.astype(np.float32)),
        calibration_time,
        inference_time,
    )


def _prediction_from_mano_theta(
    theta: np.ndarray,
    group: h5py.Group,
    side: str,
    sample_idx: int,
    device: torch.device,
    apply_left_shapedirs_fix: bool,
) -> Prediction:
    theta = _ensure_theta_sequence(theta, name=f"{group.name}/predicted_theta")
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    betas = _expand_betas(group, sample_idx, theta.shape[0], device)
    with torch.no_grad():
        vertices, joints = mano_recon(
            theta_t,
            side=side,
            betas=betas,
            apply_left_shapedirs_fix=apply_left_shapedirs_fix,
        )
    return Prediction(
        joints=joints.detach().cpu().numpy(),
        vertices=vertices.detach().cpu().numpy(),
    )


def _baseline_predict_theta(
    method: str,
    prediction_input_theta: np.ndarray,
    calibration_input_theta: np.ndarray,
    calibration_target_theta: np.ndarray,
    ridge_alpha: float,
    group: h5py.Group,
    side: str,
    sample_idx: int,
    device: torch.device,
    apply_left_shapedirs_fix: bool,
    extra_calibration_time_s: float = 0.0,
    extra_inference_time_s: float = 0.0,
) -> tuple[Prediction, float, float]:
    start = time.perf_counter()
    if method == "no_calib":
        predictor = lambda theta: theta.copy()
    elif method == "translation":
        predictor = _fit_translation(
            calibration_input_theta,
            calibration_target_theta,
        )
    elif method == "rigid":
        predictor = _fit_rigid(
            calibration_input_theta,
            calibration_target_theta,
        )
    elif method == "similarity":
        predictor = _fit_similarity(
            calibration_input_theta,
            calibration_target_theta,
        )
    elif method == "affine":
        predictor = _fit_affine(
            calibration_input_theta,
            calibration_target_theta,
        )
    elif method == "ridge":
        predictor = _fit_ridge(
            calibration_input_theta,
            calibration_target_theta,
            ridge_alpha,
        )
    else:
        raise ValueError(f"Unknown baseline method '{method}'.")
    calibration_time = time.perf_counter() - start + extra_calibration_time_s

    start = time.perf_counter()
    pred_theta = predictor(prediction_input_theta).astype(np.float32)
    prediction = _prediction_from_mano_theta(
        pred_theta,
        group,
        side,
        sample_idx,
        device,
        apply_left_shapedirs_fix,
    )
    inference_time = time.perf_counter() - start + extra_inference_time_s
    return prediction, calibration_time, inference_time


class OursPredictor:
    def __init__(self, args: Args):
        if args.model_path is None:
            raise ValueError("model_path is required when evaluating method 'ours'.")
        self.args = args
        self.inferencer: Optional[RefineModelInferencer] = None
        self.model_load_time_s = 0.0

    def _ensure_inferencer(self, side: str) -> RefineModelInferencer:
        if self.inferencer is None:
            start = time.perf_counter()
            self.inferencer = RefineModelInferencer(
                model_dir=self.args.model_path,
                side=side,
                device=self.args.device,
                paired_poses=None,
                fixed_noise=self.args.fixed_noise,
            )
            self.model_load_time_s = time.perf_counter() - start
            status = self.inferencer.canonicalizer_status()
            if status is not None:
                print(status)
        else:
            self.inferencer.set_side(side)
        return self.inferencer

    def predict(
        self,
        input_seq: np.ndarray,
        target_seq: np.ndarray,
        conditioning_target_seq: np.ndarray,
        conditioning_target_key: str,
        calib_indices: np.ndarray,
        group: h5py.Group,
        side: str,
        sample_idx: int,
    ) -> tuple[Prediction, float, float]:
        if self.args.metric_frame == "world":
            raise ValueError(
                "method 'ours' predicts MANO-local pose without global pose; "
                "use metric_frame='local' for paper-style pose evaluation."
            )

        inferencer = self._ensure_inferencer(side)
        paired_target_key = (
            "world_hand_joints_true"
            if conditioning_target_key == "world_hand_joints_true"
            else "world_hand_joints"
        )
        paired_poses = {
            "world_hand_joints_biased": input_seq[calib_indices],
            paired_target_key: conditioning_target_seq[calib_indices],
        }

        start = time.perf_counter()
        inferencer.update_condition(paired_poses=paired_poses)
        calibration_time = time.perf_counter() - start

        start = time.perf_counter()
        pred = inferencer.inference(input_seq, batched=True)
        inference_time = time.perf_counter() - start

        prediction = self._prediction_to_output(pred, group, side, sample_idx)
        return prediction, calibration_time, inference_time

    def predict_mano_theta(
        self,
        joints: np.ndarray,
        side: str,
    ) -> tuple[np.ndarray, float]:
        inferencer = self._ensure_inferencer(side)
        if inferencer.input_canonicalizer is None:
            raise ValueError(
                "baseline_theta_source='predicted' requires the model config to "
                "enable dataset.frozen_mano_canonicalizer."
            )

        start = time.perf_counter()
        theta = inferencer.input_canonicalizer.predict_theta(joints, side=side)
        prediction_time = time.perf_counter() - start
        return _ensure_theta_sequence(theta, name="predicted_mano_theta"), prediction_time

    def _prediction_to_output(
        self,
        pred: np.ndarray,
        group: h5py.Group,
        side: str,
        sample_idx: int,
    ) -> Prediction:
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred[None]
        if pred.ndim == 2 and pred.shape == (21, 3):
            pred = pred[None]
        if pred.ndim == 2 and pred.shape[-1] == 63:
            return Prediction(joints=pred.reshape(pred.shape[0], 21, 3))
        if pred.ndim == 3 and pred.shape[-2:] == (21, 3):
            return Prediction(joints=pred)
        if pred.ndim != 2 or pred.shape[-1] != 45:
            raise ValueError(f"Unsupported model output shape {pred.shape}.")

        device = self.inferencer.device
        theta = torch.as_tensor(pred, dtype=torch.float32, device=device)
        betas = _expand_betas(group, sample_idx, theta.shape[0], device)
        with torch.no_grad():
            vertices, joints = mano_recon(
                theta,
                side=side,
                betas=betas,
                apply_left_shapedirs_fix=self.args.apply_left_shapedirs_fix,
            )
            model = self.inferencer.model
            if hasattr(model, "network") and hasattr(model.network, "finger_scaling"):
                finger_scaling = model.network.finger_scaling.detach().to(device)
                if finger_scaling.dim() == 1:
                    finger_scaling = finger_scaling.unsqueeze(0).expand(
                        joints.shape[0],
                        -1,
                    )
                joints = scale_hand_fingers(joints, finger_scaling)
        return Prediction(
            joints=joints.detach().cpu().numpy(),
            vertices=vertices.detach().cpu().numpy(),
        )


def _batch_similarity_align(
    source: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(
            f"Procrustes inputs must have matching shape, got "
            f"{source.shape} and {target.shape}."
        )
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"Expected shape (B, N, 3), got {source.shape}.")

    if valid_mask is None:
        weights = np.ones(source.shape[:2], dtype=np.float64)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != source.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {source.shape[:2]}, "
                f"got {valid_mask.shape}."
            )
        weights = valid_mask.astype(np.float64)
    weight_sum = weights.sum(axis=1, keepdims=True)
    safe_weight_sum = np.maximum(weight_sum, 1.0)
    weights_expanded = weights[..., None]

    src_mean = (source * weights_expanded).sum(axis=1, keepdims=True)
    src_mean /= safe_weight_sum[..., None]
    tgt_mean = (target * weights_expanded).sum(axis=1, keepdims=True)
    tgt_mean /= safe_weight_sum[..., None]
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    covariance = np.einsum(
        "bn,bni,bnj->bij",
        weights,
        src_centered,
        tgt_centered,
    )
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = np.matmul(vt_mat.transpose(0, 2, 1), u_mat.transpose(0, 2, 1))

    det = np.linalg.det(rotation)
    reflection_mask = det < 0.0
    if np.any(reflection_mask):
        vt_mat = vt_mat.copy()
        singular_values = singular_values.copy()
        vt_mat[reflection_mask, -1, :] *= -1.0
        singular_values[reflection_mask, -1] *= -1.0
        rotation = np.matmul(vt_mat.transpose(0, 2, 1), u_mat.transpose(0, 2, 1))

    denom = np.sum(weights_expanded * src_centered**2, axis=(1, 2))
    scale = np.divide(
        singular_values.sum(axis=1),
        denom,
        out=np.ones_like(denom),
        where=denom > 1e-12,
    )
    aligned = scale[:, None, None] * np.matmul(
        src_centered,
        rotation.transpose(0, 2, 1),
    )
    aligned += tgt_mean
    return aligned.astype(np.float32)


def _auc_from_errors(
    errors_m: np.ndarray,
    max_threshold_m: float,
    num_thresholds: int,
) -> float:
    if errors_m.size == 0:
        return math.nan
    thresholds = np.linspace(0.0, max_threshold_m, max(num_thresholds, 2))
    pck = (errors_m[..., None] <= thresholds).mean(axis=tuple(range(errors_m.ndim)))
    try:
        area = np.trapezoid(pck, thresholds)
    except AttributeError:
        area = np.trapz(pck, thresholds)
    return float(area / max_threshold_m) if max_threshold_m > 0 else math.nan


def _f_score_from_errors(errors_m: np.ndarray, threshold_m: float) -> float:
    if errors_m.size == 0:
        return math.nan
    inlier_rate = float((errors_m <= threshold_m).mean())
    precision = inlier_rate
    recall = inlier_rate
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0 else 0.0


def _pinch_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pinch_pairs: list[tuple[str, int, int]],
    threshold_m: float,
    valid_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    l1_values = []
    l2_values = []
    distance_error_values = []
    per_pair_counts = {}
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    for pair_name, left_idx, right_idx in pinch_pairs:
        gt_delta = target[:, left_idx] - target[:, right_idx]
        gt_l2 = np.linalg.norm(gt_delta, axis=-1)
        mask = gt_l2 < threshold_m
        if valid_mask is not None:
            mask &= valid_mask[:, left_idx] & valid_mask[:, right_idx]
        per_pair_counts[f"pinch_count_{pair_name}"] = int(mask.sum())
        if not mask.any():
            continue

        pred_delta = pred[:, left_idx] - pred[:, right_idx]
        pred_l1 = np.abs(pred_delta[mask]).sum(axis=-1)
        pred_l2 = np.linalg.norm(pred_delta[mask], axis=-1)
        l1_values.append(pred_l1)
        l2_values.append(pred_l2)
        distance_error_values.append(np.abs(pred_l2 - gt_l2[mask]))

    if l1_values:
        l1_concat = np.concatenate(l1_values)
        l2_concat = np.concatenate(l2_values)
        err_concat = np.concatenate(distance_error_values)
        pinch_l1 = float(l1_concat.mean())
        pinch_l2 = float(l2_concat.mean())
        pinch_error = float(err_concat.mean())
        pinch_count = int(l1_concat.shape[0])
    else:
        pinch_l1 = math.nan
        pinch_l2 = math.nan
        pinch_error = math.nan
        pinch_count = 0

    return {
        "pinch_error_l1_m": pinch_l1,
        "pinch_error_l1_mm": (
            pinch_l1 * 1000.0 if not math.isnan(pinch_l1) else math.nan
        ),
        "pinch_distance_l2_m": pinch_l2,
        "pinch_distance_l2_mm": (
            pinch_l2 * 1000.0 if not math.isnan(pinch_l2) else math.nan
        ),
        "pinch_distance_error_m": pinch_error,
        "pinch_distance_error_mm": (
            pinch_error * 1000.0 if not math.isnan(pinch_error) else math.nan
        ),
        "pinch_frame_pairs": pinch_count,
        **per_pair_counts,
    }


def _compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pinch_pairs: list[tuple[str, int, int]],
    threshold_m: float,
    aucj_max_threshold_m: float,
    aucj_num_thresholds: int,
    valid_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    if valid_mask is None:
        valid_mask = np.ones(pred.shape[:2], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != pred.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {pred.shape[:2]}, "
                f"got {valid_mask.shape}."
            )

    diff = pred - target
    joint_l2 = np.linalg.norm(diff, axis=-1)
    valid_joint_l2 = joint_l2[valid_mask]
    valid_diff = diff[valid_mask]
    tip_l2 = joint_l2[:, TIP_INDICES]
    tip_valid_mask = valid_mask[:, TIP_INDICES]
    valid_tip_l2 = tip_l2[tip_valid_mask]
    pa_pred = _batch_similarity_align(pred, target, valid_mask)
    pa_joint_l2 = np.linalg.norm(pa_pred - target, axis=-1)
    valid_pa_joint_l2 = pa_joint_l2[valid_mask]
    pa_tip_pred = _batch_similarity_align(
        pred[:, TIP_INDICES],
        target[:, TIP_INDICES],
        tip_valid_mask,
    )
    pa_tip_l2 = np.linalg.norm(pa_tip_pred - target[:, TIP_INDICES], axis=-1)
    valid_pa_tip_l2 = pa_tip_l2[tip_valid_mask]

    if valid_joint_l2.size == 0:
        joint_mean = math.nan
        joint_mae = math.nan
        joint_mse = math.nan
        pa_joint_mean = math.nan
    else:
        joint_mean = float(valid_joint_l2.mean())
        joint_mae = float(np.abs(valid_diff).mean())
        joint_mse = float((valid_diff**2).mean())
        pa_joint_mean = float(valid_pa_joint_l2.mean())

    tip_mean = float(valid_tip_l2.mean()) if valid_tip_l2.size else math.nan
    pa_tip_mean = (
        float(valid_pa_tip_l2.mean()) if valid_pa_tip_l2.size else math.nan
    )

    metrics = {
        "mpjpe_m": joint_mean,
        "mpjpe_mm": joint_mean * 1000.0 if not math.isnan(joint_mean) else math.nan,
        "pa_mpjpe_m": pa_joint_mean,
        "pa_mpjpe_mm": (
            pa_joint_mean * 1000.0 if not math.isnan(pa_joint_mean) else math.nan
        ),
        "aucj": _auc_from_errors(
            valid_joint_l2,
            aucj_max_threshold_m,
            aucj_num_thresholds,
        ),
        "joint_mae_m": joint_mae,
        "joint_mse_m2": joint_mse,
        "tip_mpjpe_m": tip_mean,
        "tip_mpjpe_mm": tip_mean * 1000.0 if not math.isnan(tip_mean) else math.nan,
        "tip_pa_mpjpe_m": pa_tip_mean,
        "tip_pa_mpjpe_mm": (
            pa_tip_mean * 1000.0 if not math.isnan(pa_tip_mean) else math.nan
        ),
        "annotation_valid_ratio": float(valid_mask.mean()),
        "num_eval_points": int(valid_mask.sum()),
        "num_eval_tip_points": int(tip_valid_mask.sum()),
    }
    metrics.update(_pinch_metrics(pred, target, pinch_pairs, threshold_m, valid_mask))
    return metrics


def _compute_mesh_metrics(
    pred_vertices: Optional[np.ndarray],
    target_vertices: Optional[np.ndarray],
) -> dict[str, float]:
    metrics = {
        "pa_mpvpe_m": math.nan,
        "pa_mpvpe_mm": math.nan,
        "f_score_5mm": math.nan,
        "f_score_15mm": math.nan,
    }
    if pred_vertices is None or target_vertices is None:
        return metrics

    pa_pred_vertices = _batch_similarity_align(pred_vertices, target_vertices)
    vertex_errors = np.linalg.norm(pa_pred_vertices - target_vertices, axis=-1)
    metrics.update(
        {
            "pa_mpvpe_m": float(vertex_errors.mean()),
            "pa_mpvpe_mm": float(vertex_errors.mean() * 1000.0),
            "f_score_5mm": _f_score_from_errors(vertex_errors, 0.005),
            "f_score_15mm": _f_score_from_errors(vertex_errors, 0.015),
        }
    )
    return metrics


def _downstream_fields(group: h5py.Group) -> dict[str, float]:
    success_key = next(
        (key for key in ["success", "task_success", "success_rate"] if key in group),
        None,
    )
    time_key = next(
        (
            key
            for key in [
                "completion_time",
                "completion_time_s",
                "task_completion_time",
                "task_completion_time_s",
            ]
            if key in group
        ),
        None,
    )

    fields: dict[str, float] = {}
    if success_key is not None:
        fields["downstream_success_rate"] = float(
            np.asarray(group[success_key], dtype=np.float32).mean()
        )
    if time_key is not None:
        fields["downstream_completion_time_s"] = float(
            np.asarray(group[time_key], dtype=np.float32).mean()
        )
    return fields


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_mean(
    rows: list[dict[str, Any]],
    value_key: str,
    weight_key: str,
) -> float:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        value = row.get(value_key, math.nan)
        weight = float(row.get(weight_key, 0.0))
        if value is None or math.isnan(float(value)) or weight <= 0:
            continue
        total += float(value) * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else math.nan


def _simple_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] is not None and not math.isnan(float(row[key]))
    ]
    return float(np.mean(values)) if values else math.nan


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in rows})
    summaries = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        total_frames = int(sum(int(row["num_eval_frames"]) for row in method_rows))
        total_points = int(
            sum(
                int(row.get("num_eval_points", int(row["num_eval_frames"]) * 21))
                for row in method_rows
            )
        )
        total_tip_points = int(
            sum(
                int(row.get("num_eval_tip_points", int(row["num_eval_frames"]) * 5))
                for row in method_rows
            )
        )
        total_pinch_pairs = int(
            sum(int(row.get("pinch_frame_pairs", 0)) for row in method_rows)
        )
        inference_time_s = sum(float(row["inference_time_s"]) for row in method_rows)

        summary = {
            "method": method,
            "num_units": len(method_rows),
            "num_eval_frames": total_frames,
            "num_eval_points": total_points,
            "num_eval_tip_points": total_tip_points,
            "annotation_valid_ratio": (
                total_points / (total_frames * 21.0) if total_frames else math.nan
            ),
            "pinch_frame_pairs": total_pinch_pairs,
            "mpjpe_m": _weighted_mean(method_rows, "mpjpe_m", "num_eval_points"),
            "mpjpe_mm": _weighted_mean(method_rows, "mpjpe_mm", "num_eval_points"),
            "pa_mpjpe_mm": _weighted_mean(
                method_rows,
                "pa_mpjpe_mm",
                "num_eval_points",
            ),
            "aucj": _weighted_mean(method_rows, "aucj", "num_eval_points"),
            "tip_mpjpe_mm": _weighted_mean(
                method_rows,
                "tip_mpjpe_mm",
                "num_eval_tip_points",
            ),
            "tip_pa_mpjpe_mm": _weighted_mean(
                method_rows,
                "tip_pa_mpjpe_mm",
                "num_eval_tip_points",
            ),
            "pa_mpvpe_mm": _weighted_mean(
                method_rows,
                "pa_mpvpe_mm",
                "num_eval_frames",
            ),
            "f_score_5mm": _weighted_mean(
                method_rows,
                "f_score_5mm",
                "num_eval_frames",
            ),
            "f_score_15mm": _weighted_mean(
                method_rows,
                "f_score_15mm",
                "num_eval_frames",
            ),
            "pinch_error_l1_mm": _weighted_mean(
                method_rows,
                "pinch_error_l1_mm",
                "pinch_frame_pairs",
            ),
            "pinch_distance_l2_mm": _weighted_mean(
                method_rows,
                "pinch_distance_l2_mm",
                "pinch_frame_pairs",
            ),
            "pinch_distance_error_mm": _weighted_mean(
                method_rows,
                "pinch_distance_error_mm",
                "pinch_frame_pairs",
            ),
            "calibration_time_s": _simple_mean(method_rows, "calibration_time_s"),
            "inference_time_s": inference_time_s,
            "inference_ms_per_frame": (
                inference_time_s / total_frames * 1000.0 if total_frames else math.nan
            ),
        }

        if any("downstream_success_rate" in row for row in method_rows):
            summary["downstream_success_rate"] = _weighted_mean(
                method_rows,
                "downstream_success_rate",
                "num_eval_frames",
            )
        if any("downstream_completion_time_s" in row for row in method_rows):
            summary["downstream_completion_time_s"] = _weighted_mean(
                method_rows,
                "downstream_completion_time_s",
                "num_eval_frames",
            )
        summaries.append(summary)
    return summaries


def _print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nEvaluation summary")
    print(
        f"{'method':<14} {'MPJPE':>8} {'PA-MPJPE':>9} {'Tip-PA':>8} "
        f"{'AUCJ':>7} {'PA-MPVPE':>9} {'F@5':>7} {'F@15':>7} "
        f"{'Pinch-L2':>9} {'calib':>8}"
    )
    print("-" * 104)
    for row in summary_rows:
        print(
            f"{row['method']:<14} "
            f"{row['mpjpe_mm']:>8.3f} "
            f"{row['pa_mpjpe_mm']:>9.3f} "
            f"{row['tip_pa_mpjpe_mm']:>8.3f} "
            f"{row['aucj']:>7.3f} "
            f"{row['pa_mpvpe_mm']:>9.3f} "
            f"{row['f_score_5mm']:>7.3f} "
            f"{row['f_score_15mm']:>7.3f} "
            f"{row['pinch_distance_l2_mm']:>9.3f} "
            f"{row['calibration_time_s']:>8.4f}"
        )


def _save_predictions(
    output_dir: Path,
    method: str,
    demo_id: str,
    sample_idx: int,
    pred: np.ndarray,
    target: np.ndarray,
    input_seq: np.ndarray,
) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{method}_demo-{demo_id}_sample-{sample_idx}.npz"
    np.savez_compressed(path, pred=pred, target=target, input=input_seq)


def evaluate(args: Args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unknown_methods = sorted(set(args.methods) - set(DEFAULT_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {unknown_methods}")
    baseline_methods = [method for method in args.methods if method != "ours"]
    if "ours" in args.methods and args.model_path is None:
        raise ValueError("model_path is required when methods includes 'ours'.")
    if args.baseline_space not in {"keypoint", "mano_theta"}:
        raise ValueError("baseline_space must be 'keypoint' or 'mano_theta'.")
    if args.gt_pose_source not in {"keypoint", "mano"}:
        raise ValueError("gt_pose_source must be 'keypoint' or 'mano'.")
    if args.side_filter not in {"all", "left", "right"}:
        raise ValueError("side_filter must be 'all', 'left', or 'right'.")
    if args.baseline_theta_source not in {"predicted", "hdf5"}:
        raise ValueError("baseline_theta_source must be 'predicted' or 'hdf5'.")
    if args.bad_annotation_threshold_m < 0:
        raise ValueError("bad_annotation_threshold_m must be non-negative.")
    if args.calib_pose_index_range < 0:
        raise ValueError("calib_pose_index_range must be non-negative.")
    if (
        args.baseline_space == "mano_theta"
        and baseline_methods
        and args.baseline_theta_source == "predicted"
        and args.model_path is None
    ):
        raise ValueError(
            "model_path is required for predicted MANO-theta baselines."
        )

    pinch_pairs = _parse_pinch_pairs(args.pinch_pairs)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)

    ours_predictor = OursPredictor(args) if "ours" in args.methods else None
    theta_predictor = None
    if (
        args.baseline_space == "mano_theta"
        and baseline_methods
        and args.baseline_theta_source == "predicted"
    ):
        theta_predictor = ours_predictor if ours_predictor is not None else OursPredictor(args)
    rows: list[dict[str, Any]] = []

    with h5py.File(args.dataset_path, "r") as data:
        demo_keys = _sorted_hdf5_keys(data)
        if args.max_demos > 0:
            demo_keys = demo_keys[: args.max_demos]
        if not demo_keys:
            raise ValueError(f"No HDF5 groups found in {args.dataset_path}.")

        for demo_id in tqdm(demo_keys, desc="demos"):
            group = data[demo_id]
            side = _decode_side(group)
            if args.side_filter != "all" and side != args.side_filter:
                continue
            input_key = _resolve_key(
                group,
                args.input_key,
                ["world_hand_joints_biased", "world_hand_joints"],
            )
            target_key = _resolve_key(
                group,
                args.target_key,
                ["world_hand_joints_true", "world_hand_joints"],
                allow_missing=True,
            )
            conditioning_target_key: Optional[str]
            if "ours" in args.methods and args.conditioning_target_key != "target":
                conditioning_target_key = _resolve_key(
                    group,
                    args.conditioning_target_key,
                    ["world_hand_joints_true", "world_hand_joints"],
                )
            else:
                conditioning_target_key = None

            input_raw = _read_array(group, input_key)
            if args.gt_pose_source == "mano":
                target_raw = np.asarray(group["mano_angles"])
            elif target_key is not None:
                target_raw = _read_array(group, target_key)
            else:
                # Shape is needed only for sample counting; exact reconstruction
                # happens per sample after max-frame truncation is known.
                target_raw = np.asarray(group["mano_angles"])
            conditioning_target_raw = (
                _read_array(group, conditioning_target_key)
                if conditioning_target_key is not None
                else None
            )

            input_theta_raw = None
            target_theta_raw = None
            if (
                args.baseline_space == "mano_theta"
                and args.baseline_theta_source == "hdf5"
            ):
                if args.input_theta_key not in group:
                    raise KeyError(
                        f"baseline_space='mano_theta' requires "
                        f"'{args.input_theta_key}' in group {group.name}."
                    )
                if args.target_theta_key not in group:
                    raise KeyError(
                        f"baseline_space='mano_theta' requires "
                        f"'{args.target_theta_key}' in group {group.name}."
                    )
                input_theta_raw = _read_array(group, args.input_theta_key)
                target_theta_raw = _read_array(group, args.target_theta_key)

            sample_arrays = [input_raw, target_raw]
            if conditioning_target_raw is not None:
                sample_arrays.append(conditioning_target_raw)
            if input_theta_raw is not None:
                sample_arrays.append(input_theta_raw)
            if target_theta_raw is not None:
                sample_arrays.append(target_theta_raw)
            num_samples = _sample_count(*sample_arrays)
            if args.max_aug_sets > 0:
                num_samples = min(num_samples, args.max_aug_sets)

            for sample_idx in range(num_samples):
                input_seq = _select_aug_sample(
                    input_raw,
                    sample_idx,
                    name=f"{demo_id}/{input_key}",
                )
                target_vertices = None
                if args.gt_pose_source == "mano":
                    target_theta = _select_theta_sample(
                        target_raw,
                        sample_idx,
                        name=f"{demo_id}/mano_angles",
                    )
                    frame_count = min(input_seq.shape[0], target_theta.shape[0])
                    if args.max_frames_per_demo > 0:
                        frame_count = min(frame_count, args.max_frames_per_demo)
                    input_seq = input_seq[:frame_count]
                    target_vertices, target_seq = _reconstruct_mano_from_group(
                        group,
                        side,
                        sample_idx,
                        frame_count,
                        device,
                        args.apply_left_shapedirs_fix,
                    )
                elif target_key is not None:
                    target_seq = _select_aug_sample(
                        target_raw,
                        sample_idx,
                        name=f"{demo_id}/{target_key}",
                    )
                    input_seq, target_seq = _limit_sequence(
                        input_seq,
                        target_seq,
                        args.max_frames_per_demo,
                    )
                else:
                    frame_count = input_seq.shape[0]
                    if args.max_frames_per_demo > 0:
                        frame_count = min(frame_count, args.max_frames_per_demo)
                    input_seq = input_seq[:frame_count]
                    target_seq = _reconstruct_target_from_mano(
                        group,
                        side,
                        sample_idx,
                        frame_count,
                        device,
                        args.apply_left_shapedirs_fix,
                    )

                frame_count = input_seq.shape[0]
                if conditioning_target_raw is None:
                    conditioning_target_seq = target_seq
                    effective_conditioning_target_key = (
                        "mano_angles_reconstructed"
                        if args.gt_pose_source == "mano"
                        else target_key or "mano_angles"
                    )
                else:
                    conditioning_target_seq = _select_aug_sample(
                        conditioning_target_raw,
                        sample_idx,
                        name=f"{demo_id}/{conditioning_target_key}",
                    )
                    frame_count = min(frame_count, conditioning_target_seq.shape[0])
                    input_seq = input_seq[:frame_count]
                    target_seq = target_seq[:frame_count]
                    conditioning_target_seq = conditioning_target_seq[:frame_count]
                    if target_vertices is not None:
                        target_vertices = target_vertices[:frame_count]
                    effective_conditioning_target_key = conditioning_target_key

                input_theta = None
                target_theta = None
                if (
                    args.baseline_space == "mano_theta"
                    and args.baseline_theta_source == "hdf5"
                ):
                    input_theta = _select_theta_sample(
                        input_theta_raw,
                        sample_idx,
                        name=f"{demo_id}/{args.input_theta_key}",
                    )
                    target_theta = _select_theta_sample(
                        target_theta_raw,
                        sample_idx,
                        name=f"{demo_id}/{args.target_theta_key}",
                    )
                    input_theta = input_theta[:frame_count]
                    target_theta = target_theta[:frame_count]

                if (
                    args.calib_pose_index_range > 0
                    and args.calib_pose_index_range >= frame_count
                ):
                    raise ValueError(
                        f"calib_pose_index_range={args.calib_pose_index_range} "
                        f"leaves no evaluation frames for {demo_id} sample "
                        f"{sample_idx} with {frame_count} frames."
                    )
                calib_candidate_frame_count = (
                    args.calib_pose_index_range
                    if args.calib_pose_index_range > 0
                    else frame_count
                )
                eval_start_index = (
                    args.calib_pose_index_range
                    if args.calib_pose_index_range > 0
                    else 0
                )

                calib_indices = _select_calibration_indices(
                    calib_candidate_frame_count,
                    args.num_calib_poses,
                    args.calib_selection,
                    rng,
                )
                eval_mask = _eval_mask(
                    frame_count,
                    calib_indices,
                    args.exclude_calib_from_eval,
                    eval_start_index,
                )
                annotation_valid_mask = np.ones(target_seq.shape[:2], dtype=bool)
                annotation_rejection_applied = False
                annotation_reference_key = ""
                if (
                    args.gt_pose_source == "keypoint"
                    and target_key == "world_hand_joints"
                    and args.bad_annotation_threshold_m > 0
                ):
                    if target_vertices is None:
                        target_vertices, annotation_reference_seq = (
                            _reconstruct_mano_from_group(
                                group,
                                side,
                                sample_idx,
                                frame_count,
                                device,
                                args.apply_left_shapedirs_fix,
                            )
                        )
                    else:
                        annotation_reference_seq = target_seq
                    target_for_mask = _metric_frame(target_seq, args.metric_frame)
                    reference_for_mask = _metric_frame(
                        annotation_reference_seq,
                        args.metric_frame,
                    )
                    annotation_error_sq = (
                        (target_for_mask - reference_for_mask) ** 2
                    ).sum(axis=-1)
                    annotation_valid_mask = (
                        annotation_error_sq
                        <= args.bad_annotation_threshold_m**2
                    )
                    annotation_rejection_applied = True
                    annotation_reference_key = "mano_angles_reconstructed"

                downstream = _downstream_fields(group)
                theta_baseline_inputs = None
                if args.baseline_space == "mano_theta" and baseline_methods:
                    if args.baseline_theta_source == "hdf5":
                        theta_baseline_inputs = {
                            "prediction_input_theta": input_theta,
                            "calibration_input_theta": input_theta[calib_indices],
                            "calibration_target_theta": target_theta[calib_indices],
                            "extra_calibration_time_s": 0.0,
                            "extra_inference_time_s": 0.0,
                        }
                    else:
                        assert theta_predictor is not None
                        calibration_input_theta, input_calib_time = (
                            theta_predictor.predict_mano_theta(
                                input_seq[calib_indices],
                                side,
                            )
                        )
                        calibration_target_theta, target_calib_time = (
                            theta_predictor.predict_mano_theta(
                                target_seq[calib_indices],
                                side,
                            )
                        )
                        prediction_input_theta, input_prediction_time = (
                            theta_predictor.predict_mano_theta(input_seq, side)
                        )
                        theta_baseline_inputs = {
                            "prediction_input_theta": prediction_input_theta,
                            "calibration_input_theta": calibration_input_theta,
                            "calibration_target_theta": calibration_target_theta,
                            "extra_calibration_time_s": (
                                input_calib_time + target_calib_time
                            ),
                            "extra_inference_time_s": input_prediction_time,
                        }

                for method in args.methods:
                    if method == "ours":
                        assert ours_predictor is not None
                        pred_output, calibration_time, inference_time = (
                            ours_predictor.predict(
                                input_seq,
                                target_seq,
                                conditioning_target_seq,
                                effective_conditioning_target_key,
                                calib_indices,
                                group,
                                side,
                                sample_idx,
                            )
                        )
                    else:
                        if args.baseline_space == "mano_theta":
                            assert theta_baseline_inputs is not None
                            pred_output, calibration_time, inference_time = (
                                _baseline_predict_theta(
                                    method,
                                    theta_baseline_inputs[
                                        "prediction_input_theta"
                                    ],
                                    theta_baseline_inputs[
                                        "calibration_input_theta"
                                    ],
                                    theta_baseline_inputs[
                                        "calibration_target_theta"
                                    ],
                                    args.ridge_alpha,
                                    group,
                                    side,
                                    sample_idx,
                                    device,
                                    args.apply_left_shapedirs_fix,
                                    theta_baseline_inputs[
                                        "extra_calibration_time_s"
                                    ],
                                    theta_baseline_inputs[
                                        "extra_inference_time_s"
                                    ],
                                )
                            )
                        else:
                            pred_output, calibration_time, inference_time = (
                                _baseline_predict(
                                    method,
                                    input_seq,
                                    target_seq,
                                    calib_indices,
                                    args.ridge_alpha,
                                )
                            )

                    pred_metric = _metric_frame(
                        pred_output.joints,
                        args.metric_frame,
                    )[eval_mask]
                    target_metric = _metric_frame(target_seq, args.metric_frame)[
                        eval_mask
                    ]
                    input_metric = _metric_frame(input_seq, args.metric_frame)[
                        eval_mask
                    ]
                    metrics = _compute_metrics(
                        pred_metric,
                        target_metric,
                        pinch_pairs,
                        args.pinch_threshold_m,
                        args.aucj_max_threshold_m,
                        args.aucj_num_thresholds,
                        annotation_valid_mask[eval_mask],
                    )
                    if pred_output.vertices is not None:
                        if target_vertices is None:
                            target_vertices, _ = _reconstruct_mano_from_group(
                                group,
                                side,
                                sample_idx,
                                frame_count,
                                device,
                                args.apply_left_shapedirs_fix,
                            )
                        mesh_metrics = _compute_mesh_metrics(
                            pred_output.vertices[eval_mask],
                            target_vertices[eval_mask],
                        )
                    else:
                        mesh_metrics = _compute_mesh_metrics(None, None)

                    row: dict[str, Any] = {
                        "dataset_path": args.dataset_path,
                        "demo_id": demo_id,
                        "sample_idx": sample_idx,
                        "method": method,
                        "side": side,
                        "side_filter": args.side_filter,
                        "input_key": input_key,
                        "target_key": (
                            "mano_angles_reconstructed"
                            if args.gt_pose_source == "mano"
                            else target_key or "mano_angles"
                        ),
                        "conditioning_target_key": (
                            effective_conditioning_target_key
                        ),
                        "gt_pose_source": args.gt_pose_source,
                        "baseline_space": args.baseline_space,
                        "baseline_theta_source": args.baseline_theta_source,
                        "input_theta_key": args.input_theta_key,
                        "target_theta_key": args.target_theta_key,
                        "ridge_alpha": args.ridge_alpha,
                        "metric_frame": args.metric_frame,
                        "bad_annotation_threshold_m": (
                            args.bad_annotation_threshold_m
                        ),
                        "bad_annotation_rejection": annotation_rejection_applied,
                        "bad_annotation_reference_key": annotation_reference_key,
                        "num_calib_poses": int(len(calib_indices)),
                        "calib_indices": " ".join(str(int(i)) for i in calib_indices),
                        "calib_pose_index_range": args.calib_pose_index_range,
                        "calib_candidate_frame_count": int(
                            calib_candidate_frame_count
                        ),
                        "eval_start_index": int(eval_start_index),
                        "num_eval_frames": int(eval_mask.sum()),
                        "calibration_time_s": float(calibration_time),
                        "inference_time_s": float(inference_time),
                        **metrics,
                        **mesh_metrics,
                        **downstream,
                    }
                    rows.append(row)

                    if args.write_predictions:
                        _save_predictions(
                            output_dir,
                            method,
                            demo_id,
                            sample_idx,
                            pred_metric,
                            target_metric,
                            input_metric,
                        )

    if not rows:
        raise ValueError(
            f"No evaluation rows produced for side_filter='{args.side_filter}'."
        )

    summary_rows = _aggregate(rows)
    metadata = {
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "methods": args.methods,
        "num_calib_poses": args.num_calib_poses,
        "side_filter": args.side_filter,
        "conditioning_target_key": args.conditioning_target_key,
        "gt_pose_source": args.gt_pose_source,
        "baseline_space": args.baseline_space,
        "baseline_theta_source": args.baseline_theta_source,
        "input_theta_key": args.input_theta_key,
        "target_theta_key": args.target_theta_key,
        "ridge_alpha": args.ridge_alpha,
        "metric_frame": args.metric_frame,
        "pinch_threshold_m": args.pinch_threshold_m,
        "pinch_pairs": args.pinch_pairs,
        "aucj_max_threshold_m": args.aucj_max_threshold_m,
        "aucj_num_thresholds": args.aucj_num_thresholds,
        "bad_annotation_threshold_m": args.bad_annotation_threshold_m,
        "calib_selection": args.calib_selection,
        "calib_pose_index_range": args.calib_pose_index_range,
        "exclude_calib_from_eval": args.exclude_calib_from_eval,
    }
    if ours_predictor is not None:
        metadata["model_load_time_s"] = ours_predictor.model_load_time_s

    _write_csv(output_dir / "summary.csv", summary_rows)
    with (output_dir / "summary.json").open("w") as file:
        json.dump({"metadata": metadata, "summary": summary_rows}, file, indent=2)
    try:
        _write_csv(output_dir / "per_sequence.csv", rows)
    except OSError as exc:
        print(
            "Warning: failed to write per_sequence.csv after summary files were "
            f"saved: {exc}"
        )

    return rows, summary_rows


def main(args: Args) -> None:
    rows, summary_rows = evaluate(args)
    _print_summary(summary_rows)
    print(f"\nWrote {len(rows)} per-sequence rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
