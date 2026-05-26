#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from calibration_net.policy.baseline_finetune import (
    BaselineConfig,
    adapt_model,
    configure_baseline_model,
    predict,
    trainable_parameter_count,
)
from calibration_net.policy.loss import (
    TIP_INDICIES,
    compute_relative_distances,
    relative_distance_loss,
)
from calibration_net.policy.model import load_model
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.utils import mano_recon, scale_hand_fingers, transform_joint3d


TIP_JOINT_INDICES = [4, 8, 12, 16, 20]


def make_calibration_rng(
    seed: int,
    session_name: str,
    sample_idx: int,
    num_calib_actions: int,
) -> np.random.Generator:
    payload = f"calibration|{int(seed)}|{session_name}|{int(sample_idx)}|{int(num_calib_actions)}".encode(
        "utf-8"
    )
    child_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng(child_seed)


def _similarity_align_one(
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Canonical Umeyama similarity alignment for row-vector joints."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if fit_mask is None:
        src_fit = src
        tgt_fit = tgt
    else:
        valid = np.asarray(fit_mask, dtype=bool)
        if int(valid.sum()) < 3:
            return src.copy()
        src_fit = src[valid]
        tgt_fit = tgt[valid]

    src_mean = src_fit.mean(axis=0, keepdims=True)
    tgt_mean = tgt_fit.mean(axis=0, keepdims=True)
    src_centered = src_fit - src_mean
    tgt_centered = tgt_fit - tgt_mean
    src_var = float(np.sum(src_centered ** 2))
    if src_var < 1e-12:
        return np.broadcast_to(tgt_mean, src.shape).copy()

    cov = src_centered.T @ tgt_centered
    u, singular_values, vh = np.linalg.svd(cov)
    z = np.eye(cov.shape[0], dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        z[-1, -1] = -1.0
    rot = u @ z @ vh
    scale = float(np.sum(singular_values * np.diag(z)) / src_var)
    return scale * ((src - src_mean) @ rot) + tgt_mean


def batch_similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align source joints to target joints per frame with similarity Procrustes."""
    return np.stack(
        [_similarity_align_one(src, tgt) for src, tgt in zip(source, target)],
        axis=0,
    )


def batch_similarity_align_masked(
    source: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if valid_mask is None:
        return batch_similarity_align(source, target)

    return np.stack(
        [
            _similarity_align_one(src, tgt, valid)
            for src, tgt, valid in zip(source, target, valid_mask)
        ],
        axis=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-session LoRA/full/head/linear finetune baselines for "
            "joint->MANO calibration."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Pretrained checkpoint dir")
    parser.add_argument("--data-path", required=True, help="Calibration/eval HDF5")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics")
    parser.add_argument(
        "--method",
        required=True,
        choices=["lora", "full_ft", "residual_linear", "last_layer"],
    )
    parser.add_argument(
        "--num-calib-actions",
        type=int,
        nargs="+",
        default=[4],
        help=(
            "Calibration pair/frame counts to sweep. Example: "
            "--num-calib-actions 4 8 16"
        ),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--input-key",
        default="world_hand_joints_biased",
        help="Observed/biased joint key. Falls back to world_hand_joints.",
    )
    parser.add_argument("--target-key", default="mano_angles")
    parser.add_argument(
        "--clean-input-key",
        default="world_hand_joints",
        help="Clean/non-biased joints used to build pseudo GT with the original model.",
    )
    parser.add_argument(
        "--target-source",
        choices=["clean_model", "dataset"],
        default="dataset",
        help=(
            "clean_model: use original model(clean joints) as GT MANO pose. "
            "dataset: use --target-key from HDF5."
        ),
    )
    parser.add_argument(
        "--true-joints-key",
        default="world_hand_joints",
        help="GT joint key used for MPJPE/PA-MPJPE/tip MPJPE metrics.",
    )
    parser.add_argument(
        "--pred-beta-source",
        default="auto",
        choices=["auto", "group", "neutral"],
        help=(
            "MANO beta used for predicted-theta FK. auto uses neutral beta for "
            "world_hand_joints_true targets and group betas otherwise."
        ),
    )
    parser.add_argument(
        "--side-filter",
        default="right",
        choices=["all", "left", "right"],
        help="Only run one MANO side. Default: right, matching Franklin eval.",
    )
    parser.add_argument(
        "--exclude-calib-from-eval",
        action="store_true",
        help="Exclude calibration frames from metrics. Default matches Franklin eval: include them.",
    )
    parser.add_argument(
        "--calibration-num",
        "--calibration_num",
        type=int,
        default=0,
        help=(
            "If >0, use frames [0,N) only for calibration selection and "
            "frames [N,T) only for evaluation. Example: --calibration-num 300."
        ),
    )
    parser.add_argument(
        "--split-key",
        default="",
        help="Optional per-frame action/pose label key. If set, K means K unique labels.",
    )
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=-1,
        help=(
            "Biased sample index for keys shaped (S,T,...). "
            "Use -1 to evaluate all biased samples as separate sessions."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-use-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sigmoid gates on static LoRA outputs, matching Franklin HyperLoRA.",
    )
    parser.add_argument("--lora-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=[],
        help=(
            "Name filters for static LoRA target Linear modules. "
            "Default is empty, matching Franklin HyperLoRA name_filter=[] "
            "and applying LoRA to all Linear layers."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--no-reconstruction",
        action="store_true",
        help="Skip MANO reconstruction metrics.",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full-sequence finetuned predictions to predictions.hdf5.",
    )
    return parser.parse_args()


def decode_side(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "right"
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value).lower()


def read_array(group: h5py.Group, key: str, sample_idx: int = 0) -> np.ndarray:
    array = np.asarray(group[key])
    if array.ndim >= 4 and key.endswith("_biased"):
        if sample_idx < 0:
            raise ValueError("sample_idx must be >= 0 when reading a biased sample")
        sample_idx = min(sample_idx, array.shape[0] - 1)
        array = array[sample_idx]
    return array.astype(np.float32, copy=False) if np.issubdtype(array.dtype, np.number) else array


def read_bad_frame_mask(group: h5py.Group, length: int) -> np.ndarray:
    if "bad_frame" not in group:
        return np.zeros(length, dtype=bool)
    mask = np.asarray(group["bad_frame"], dtype=bool).reshape(-1)
    if len(mask) != length:
        raise ValueError(f"bad_frame length {len(mask)} does not match sequence length {length}")
    return mask


def read_valid_joint_mask(group: h5py.Group, length: int) -> np.ndarray | None:
    if "gt_joint_mask" not in group:
        return None
    bad = np.asarray(group["gt_joint_mask"], dtype=bool)
    if bad.shape != (length, 21):
        raise ValueError(f"gt_joint_mask shape {bad.shape} does not match expected {(length, 21)}")
    return ~bad


def biased_sample_indices(group: h5py.Group, input_key: str, sample_idx: int) -> list[int]:
    if input_key not in group:
        return [0]
    shape = group[input_key].shape
    if len(shape) >= 4 and input_key.endswith("_biased"):
        if sample_idx < 0:
            return list(range(shape[0]))
        return [min(sample_idx, shape[0] - 1)]
    return [0]


def resolve_input_key(group: h5py.Group, requested_key: str) -> str:
    if requested_key in group:
        return requested_key
    if "world_hand_joints" in group:
        return "world_hand_joints"
    raise KeyError(
        f"Input key '{requested_key}' not found and no 'world_hand_joints' fallback in "
        f"group keys={list(group.keys())}"
    )


def resolve_true_joints_key(group: h5py.Group, requested_key: str) -> str | None:
    if requested_key and requested_key in group:
        return requested_key
    for key in ("world_hand_joints_true", "world_hand_joints"):
        if key in group:
            return key
    return None


def select_mano_betas(
    group: h5py.Group,
    true_joints_key: str | None,
    pred_beta_source: str,
) -> tuple[np.ndarray | None, str]:
    source = pred_beta_source
    if source == "auto":
        source = "neutral" if true_joints_key == "world_hand_joints_true" else "group"
    if source == "neutral":
        return None, source
    if "betas" not in group:
        return None, "missing_group"
    return np.asarray(group["betas"], dtype=np.float32), source


def split_indices(
    group: h5py.Group,
    length: int,
    num_calib: int,
    rng: np.random.Generator,
    split_key: str = "",
    bad_frame_mask: np.ndarray | None = None,
    exclude_calib_from_eval: bool = False,
    calibration_num: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    all_indices = np.arange(length)
    good_indices = all_indices if bad_frame_mask is None else all_indices[~bad_frame_mask]
    if len(good_indices) == 0:
        raise ValueError("No non-bad frames available for calibration")

    if calibration_num > 0:
        split_at = int(calibration_num)
        calib_candidates = good_indices[good_indices < split_at]
        eval_indices = all_indices[all_indices >= split_at]
        if len(calib_candidates) == 0:
            raise ValueError(f"No non-bad frames in [0,{split_at}) for calibration")
        if len(eval_indices) == 0:
            raise ValueError(f"No frames in [{split_at},T) for evaluation")
        count = min(int(num_calib), len(calib_candidates))
        calib_indices = np.sort(rng.choice(calib_candidates, size=count, replace=False))
        return calib_indices, eval_indices, f"calibration_num_{split_at}"

    if split_key and split_key in group:
        labels = np.asarray(group[split_key])
        labels = labels.reshape(-1)
        if len(labels) != length:
            raise ValueError(
                f"split_key '{split_key}' has length {len(labels)}, expected {length}"
            )
        unique_labels = np.array(sorted(set(labels[good_indices].tolist())))
        if len(unique_labels) <= num_calib:
            calib_labels = unique_labels
        else:
            calib_labels = rng.choice(unique_labels, size=num_calib, replace=False)
        calib_mask = np.isin(labels, calib_labels)
        if bad_frame_mask is not None:
            calib_mask &= ~bad_frame_mask
        eval_mask = np.ones_like(calib_mask, dtype=bool)
        if exclude_calib_from_eval:
            eval_mask = ~calib_mask
            if not eval_mask.any():
                eval_mask = np.ones_like(calib_mask, dtype=bool)
        return all_indices[calib_mask], all_indices[eval_mask], "label"

    if len(good_indices) <= num_calib:
        calib_indices = good_indices
        return calib_indices, all_indices, "frame_all_good"
    calib_indices = rng.choice(good_indices, size=num_calib, replace=False)
    eval_mask = np.ones(length, dtype=bool)
    if exclude_calib_from_eval:
        eval_mask[calib_indices] = False
    return np.sort(calib_indices), all_indices[eval_mask], "frame"


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def process_arrays(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    process_x,
    process_y_train,
) -> tuple[np.ndarray, np.ndarray]:
    x = process_x(x_raw)
    y = process_y_train(y_raw)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


@torch.no_grad()
def theta_metrics(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> dict[str, float]:
    return {
        "theta_mse": float(F.mse_loss(pred_theta, gt_theta).cpu()),
        "theta_mae": float(torch.mean(torch.abs(pred_theta - gt_theta)).cpu()),
    }


def reconstruction_metrics(
    pred_theta: torch.Tensor,
    gt_theta: torch.Tensor,
    true_joints: np.ndarray | None,
    side: str,
    model: torch.nn.Module | None = None,
    betas: np.ndarray | None = None,
    valid_joint_mask: np.ndarray | None = None,
) -> dict[str, float]:
    with torch.no_grad():
        try:
            betas_tensor = None
            if betas is not None:
                betas_tensor = torch.as_tensor(
                    betas,
                    dtype=torch.float32,
                    device=pred_theta.device,
                )
                if betas_tensor.ndim == 1:
                    betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)

            _, pred_joints = mano_recon(
                pred_theta,
                side=side,
                betas=betas_tensor,
                apply_left_shapedirs_fix=True,
            )
            network = getattr(model, "network", None)
            if network is None and hasattr(model, "base_model"):
                network = getattr(model.base_model, "network", None)
            if network is not None and hasattr(network, "finger_scaling"):
                pred_joints = scale_hand_fingers(pred_joints, network.finger_scaling)

            _, gt_joints = mano_recon(
                gt_theta,
                side=side,
                betas=betas_tensor,
                apply_left_shapedirs_fix=True,
            )
            if true_joints is None:
                raw_true_local = None
                target_joints = gt_joints
            else:
                raw_true_joints = torch.as_tensor(
                    true_joints,
                    dtype=torch.float32,
                    device=pred_theta.device,
                )
                raw_true_local = transform_joint3d(raw_true_joints)
                target_joints = raw_true_local

            pred_local = transform_joint3d(pred_joints)
            if torch.is_tensor(target_joints):
                target_local = target_joints
            else:
                target_local = transform_joint3d(target_joints)
            pred_local = torch.as_tensor(
                pred_local, dtype=torch.float32, device=pred_theta.device
            )
            target_local = torch.as_tensor(
                target_local, dtype=torch.float32, device=pred_theta.device
            )

            pred_dists = compute_relative_distances(pred_local)
            gt_dists = compute_relative_distances(target_local)
            pred_np = pred_local.detach().cpu().numpy()
            target_np = target_local.detach().cpu().numpy()
            if valid_joint_mask is not None:
                valid_joint_mask = np.asarray(valid_joint_mask, dtype=bool)
                if valid_joint_mask.shape != pred_np.shape[:2]:
                    raise ValueError(
                        f"valid_joint_mask shape {valid_joint_mask.shape} "
                        f"does not match joints shape {pred_np.shape[:2]}"
                    )
            pa_pred_np = batch_similarity_align_masked(pred_np, target_np, valid_joint_mask)
            valid = (
                np.ones(pred_np.shape[:2], dtype=bool)
                if valid_joint_mask is None
                else valid_joint_mask
            )
            if not np.any(valid):
                raise ValueError("No valid joints left for reconstruction metrics")
            tip_valid = valid[:, TIP_JOINT_INDICES]
            if not np.any(tip_valid):
                tip_valid = valid

            joint_err = np.linalg.norm(pred_np - target_np, axis=-1)
            pa_joint_err = np.linalg.norm(pa_pred_np - target_np, axis=-1)
            recon_mse = float(np.mean(((pred_np - target_np) ** 2)[valid[..., None].repeat(3, axis=-1)]))
            recon_mae = float(np.mean(np.abs(pred_np - target_np)[valid[..., None].repeat(3, axis=-1)]))
            mpjpe = float(np.mean(joint_err[valid]))
            pa_mpjpe = float(np.mean(pa_joint_err[valid]))
            tip_err = np.linalg.norm(
                pred_np[:, TIP_JOINT_INDICES] - target_np[:, TIP_JOINT_INDICES],
                axis=-1,
            )
            tip_mpjpe = float(np.mean(tip_err[tip_valid]))

            metrics = {
                "mse": recon_mse,
                "mae": recon_mae,
                "mpjpe": mpjpe,
                "pa_mpjpe": pa_mpjpe,
                "tip_mpjpe": tip_mpjpe,
                "mpjpe_mm": mpjpe * 1000.0,
                "pa_mpjpe_mm": pa_mpjpe * 1000.0,
                "tip_mpjpe_mm": tip_mpjpe * 1000.0,
                "joint_mse": recon_mse,
                "joint_mae": recon_mae,
                "tip_mse": float(
                    F.mse_loss(pred_local[:, TIP_INDICIES], target_local[:, TIP_INDICIES]).cpu()
                ),
                "relative_distance_loss": float(
                    relative_distance_loss(pred_local, target_local).cpu()
                ),
                "mean_abs_tip_distance_error": float(
                    torch.mean(torch.abs(pred_dists - gt_dists)).cpu()
                ),
                "num_metric_joints": int(np.sum(valid)),
                "num_masked_bad_joints": int(valid.size - np.sum(valid)),
            }
            if raw_true_local is not None:
                metrics["hdf5_true_mpjpe"] = mpjpe
                metrics["hdf5_true_mpjpe_mm"] = mpjpe * 1000.0
            return metrics
        except Exception as exc:
            return {"reconstruction_error": str(exc)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"sessions": rows, "summary": summary}, f, indent=2, sort_keys=True)

    if rows:
        keys = sorted({key for row in rows for key in row.keys()})
        with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def reconstruct_prediction_joints(
    pred_theta: torch.Tensor,
    side: str,
    betas: np.ndarray | None,
    model: torch.nn.Module | None = None,
) -> np.ndarray:
    with torch.no_grad():
        betas_tensor = None
        if betas is not None:
            betas_tensor = torch.as_tensor(
                betas,
                dtype=torch.float32,
                device=pred_theta.device,
            )
            if betas_tensor.ndim == 1:
                betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)

        _, pred_joints = mano_recon(
            pred_theta,
            side=side,
            betas=betas_tensor,
            apply_left_shapedirs_fix=True,
        )
        network = getattr(model, "network", None)
        if network is None and hasattr(model, "base_model"):
            network = getattr(model.base_model, "network", None)
        if network is not None and hasattr(network, "finger_scaling"):
            pred_joints = scale_hand_fingers(pred_joints, network.finger_scaling)

        pred_local = transform_joint3d(pred_joints)
        pred_local = torch.as_tensor(
            pred_local,
            dtype=torch.float32,
            device=pred_theta.device,
        )
        return pred_local.detach().cpu().numpy().astype(np.float32)


def save_session_predictions(
    output_dir: Path,
    session_name: str,
    sample_idx: int,
    method: str,
    num_calib_actions: int,
    pred_theta: np.ndarray,
    pred_joints: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_dir / "predictions.hdf5", "a") as f:
        group = f.require_group(session_name.split("/")[0])
        sample_group = group.require_group(f"sample_{sample_idx}")
        for key in ("pred_theta", "pred_joints", "calib_idx", "eval_idx"):
            if key in sample_group:
                del sample_group[key]
        sample_group.create_dataset("pred_theta", data=pred_theta, compression="gzip")
        sample_group.create_dataset("pred_joints", data=pred_joints, compression="gzip")
        sample_group.create_dataset("calib_idx", data=calib_idx.astype(np.int64))
        sample_group.create_dataset("eval_idx", data=eval_idx.astype(np.int64))
        sample_group.attrs["method"] = method
        sample_group.attrs["num_calib_actions"] = int(num_calib_actions)


def run_session(
    group: h5py.Group,
    session_name: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    norm_stats: dict[str, Any],
    device: torch.device,
    num_calib_actions: int,
    sample_idx: int,
    output_dir: Path,
) -> dict[str, Any]:
    model, _, _ = load_model(args.model_dir, device=str(device))
    model.to(device)
    model.eval()

    input_key = resolve_input_key(group, args.input_key)
    if args.target_source == "dataset" and args.target_key not in group:
        raise KeyError(f"Target key '{args.target_key}' not found in session {session_name}")
    if args.target_source == "clean_model" and args.clean_input_key not in group:
        raise KeyError(
            f"Clean input key '{args.clean_input_key}' not found in session {session_name}"
        )

    x_raw = read_array(group, input_key, sample_idx=sample_idx)
    process_x = build_preprocessing(cfg["dataset"]["preprocessing_x"], "x", norm_stats)
    process_y_train = build_preprocessing(cfg["dataset"]["preprocessing_y"], "y", norm_stats)
    process_y_eval = build_preprocessing(
        [
            process if process != "normalize" else "denormalize"
            for process in cfg["dataset"]["preprocessing_y"]
        ],
        "y",
        norm_stats,
    )
    x_proc = np.asarray(process_x(x_raw), dtype=np.float32)
    bad_frame_mask = read_bad_frame_mask(group, len(x_proc))
    valid_joint_mask_all = read_valid_joint_mask(group, len(x_proc))

    if args.target_source == "clean_model":
        clean_raw = read_array(group, args.clean_input_key, sample_idx=sample_idx)
        if len(x_raw) != len(clean_raw):
            raise ValueError(
                f"Session {session_name} input/clean length mismatch: "
                f"{len(x_raw)} vs {len(clean_raw)}"
            )
        clean_proc = np.asarray(process_x(clean_raw), dtype=np.float32)
        with torch.no_grad():
            clean_tensor = to_tensor(clean_proc, device)
            y_proc_tensor = predict(model, clean_tensor)
        y_proc = y_proc_tensor.detach().cpu().numpy().astype(np.float32)
        y_raw = np.asarray(process_y_eval(y_proc), dtype=np.float32)
    else:
        y_raw = read_array(group, args.target_key, sample_idx=sample_idx)
        y_proc = np.asarray(process_y_train(y_raw), dtype=np.float32)

    if len(x_proc) != len(y_proc):
        raise ValueError(
            f"Session {session_name} input/target length mismatch: "
            f"{len(x_proc)} vs {len(y_proc)}"
        )

    calib_rng = make_calibration_rng(
        args.seed,
        session_name.split("/", 1)[0],
        sample_idx,
        num_calib_actions,
    )
    calib_idx, eval_idx, split_mode = split_indices(
        group,
        len(x_proc),
        num_calib_actions,
        calib_rng,
        split_key=args.split_key,
        bad_frame_mask=bad_frame_mask,
        exclude_calib_from_eval=args.exclude_calib_from_eval,
        calibration_num=args.calibration_num,
    )

    baseline = BaselineConfig(
        method=args.method,
        num_calib_actions=num_calib_actions,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        target_modules=args.target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_use_gates=args.lora_use_gates,
        lora_gate_init=args.lora_gate_init,
    )

    torch.manual_seed(args.seed)
    adapted_model = configure_baseline_model(
        model,
        method=baseline.method,
        input_shape=tuple(x_proc.shape[1:]),
        output_size=y_proc.shape[-1],
        target_modules=baseline.target_modules,
        lora_rank=baseline.lora_rank,
        lora_alpha=baseline.lora_alpha,
        lora_dropout=baseline.lora_dropout,
        lora_use_gates=baseline.lora_use_gates,
        lora_gate_init=baseline.lora_gate_init,
    ).to(device)

    x_calib = to_tensor(x_proc[calib_idx], device)
    y_calib = to_tensor(y_proc[calib_idx], device)
    losses, adaptation_time = adapt_model(
        adapted_model,
        x_calib,
        y_calib,
        steps=baseline.steps,
        lr=baseline.lr,
        weight_decay=baseline.weight_decay,
        grad_clip=args.grad_clip,
        show_progress=not args.no_progress,
        progress_desc=f"{args.method} k={num_calib_actions} {session_name}",
    )

    x_eval = to_tensor(x_proc[eval_idx], device)
    inference_start = perf_counter()
    pred_proc = predict(adapted_model, x_eval)
    inference_time = perf_counter() - inference_start

    pred_np = pred_proc.detach().cpu().numpy()
    pred_theta_np = process_y_eval(pred_np)
    gt_theta_np = y_raw[eval_idx]
    pred_theta = to_tensor(np.asarray(pred_theta_np, dtype=np.float32), device)
    gt_theta = to_tensor(np.asarray(gt_theta_np, dtype=np.float32), device)

    metrics: dict[str, Any] = {
        "session": session_name,
        "group": session_name.split("/")[0],
        "sample_idx": int(sample_idx),
        "method": args.method,
        "target_source": args.target_source,
        "split_mode": split_mode,
        "calibration_num": int(args.calibration_num),
        "calib_idx": " ".join(str(int(i)) for i in calib_idx),
        "num_calib_requested": num_calib_actions,
        "num_calib_samples": int(len(calib_idx)),
        "num_eval_samples": int(len(eval_idx)),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_use_gates": bool(args.lora_use_gates),
        "lora_gate_init": float(args.lora_gate_init),
        "target_modules": " ".join(args.target_modules),
        "trainable_params": int(trainable_parameter_count(adapted_model)),
        "adaptation_time_sec": float(adaptation_time),
        "inference_time_sec": float(inference_time),
        "final_adapt_loss": float(losses[-1]) if losses else None,
        "initial_adapt_loss": float(losses[0]) if losses else None,
    }
    metrics.update(theta_metrics(pred_theta, gt_theta))

    side = "right"
    if "mano_side" in group:
        side = decode_side(np.asarray(group["mano_side"]))
    metrics["side"] = side

    if not args.no_reconstruction:
        true_joints = None
        true_key = None
        if args.target_source == "dataset":
            true_key = resolve_true_joints_key(group, args.true_joints_key)
            if true_key is not None:
                true_joints = read_array(group, true_key, sample_idx=sample_idx)[eval_idx]
        betas, beta_source = select_mano_betas(group, true_key, args.pred_beta_source)
        metrics["true_joints_key"] = true_key
        metrics["pred_beta_source"] = beta_source
        metrics.update(
            reconstruction_metrics(
                pred_theta,
                gt_theta,
                true_joints,
                side,
                model=adapted_model,
                betas=betas,
                valid_joint_mask=(
                    valid_joint_mask_all[eval_idx]
                    if valid_joint_mask_all is not None and true_joints is not None
                    else None
                ),
            )
        )

    if args.save_predictions:
        side = "right"
        if "mano_side" in group:
            side = decode_side(np.asarray(group["mano_side"]))
        true_key = resolve_true_joints_key(group, args.true_joints_key)
        betas, beta_source = select_mano_betas(group, true_key, args.pred_beta_source)
        with torch.no_grad():
            x_all = to_tensor(x_proc, device)
            pred_all_proc = predict(adapted_model, x_all)
            pred_all_np = pred_all_proc.detach().cpu().numpy()
            pred_all_theta_np = np.asarray(process_y_eval(pred_all_np), dtype=np.float32)
            pred_all_theta = to_tensor(pred_all_theta_np, device)
            pred_all_joints = reconstruct_prediction_joints(
                pred_all_theta,
                side=side,
                betas=betas,
                model=adapted_model,
            )
        save_session_predictions(
            output_dir,
            session_name=session_name,
            sample_idx=sample_idx,
            method=args.method,
            num_calib_actions=num_calib_actions,
            pred_theta=pred_all_theta_np,
            pred_joints=pred_all_joints,
            calib_idx=calib_idx,
            eval_idx=eval_idx,
        )

    if hasattr(adapted_model, "remove_hooks"):
        adapted_model.remove_hooks()
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    _, norm_stats, cfg = load_model(args.model_dir, device="cpu")
    input_key_for_samples = args.input_key
    with h5py.File(os.path.expanduser(args.data_path), "r") as f:
        session_names = list(f.keys())
        try:
            session_names = sorted(session_names, key=lambda item: int(item))
        except Exception:
            session_names = sorted(session_names)
        if args.max_sessions > 0:
            session_names = session_names[: args.max_sessions]

        for num_calib_actions in args.num_calib_actions:
            output_dir = Path(args.output_dir) / args.method / f"k{num_calib_actions}"
            run_start = perf_counter()
            rows: list[dict[str, Any]] = []
            total_units = 0
            for session_name in session_names:
                group = f[session_name]
                side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
                if args.side_filter != "all" and side != args.side_filter:
                    continue
                resolved_input_key = (
                    input_key_for_samples
                    if input_key_for_samples in group
                    else "world_hand_joints"
                )
                total_units += len(
                    biased_sample_indices(group, resolved_input_key, args.sample_idx)
                )

            unit_idx = 0
            session_iter = tqdm(
                session_names,
                desc=f"{args.method} k={num_calib_actions} sessions",
                disable=args.no_progress,
            )
            for session_name in session_iter:
                group = f[session_name]
                side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
                if args.side_filter != "all" and side != args.side_filter:
                    continue
                resolved_input_key = (
                    input_key_for_samples
                    if input_key_for_samples in group
                    else "world_hand_joints"
                )
                sample_indices = biased_sample_indices(
                    group, resolved_input_key, args.sample_idx
                )
                sample_iter = tqdm(
                    sample_indices,
                    desc=f"{session_name} biased samples",
                    leave=False,
                    disable=args.no_progress or len(sample_indices) <= 1,
                )
                for sample_idx in sample_iter:
                    unit_idx += 1
                    label = f"{session_name}/sample_{sample_idx}"
                    if not args.no_progress:
                        session_iter.set_postfix(unit=f"{unit_idx}/{total_units}")
                    print(
                        f"[{unit_idx}/{total_units}] {args.method} "
                        f"k={num_calib_actions} session={label}"
                    )
                    row = run_session(
                        group,
                        label,
                        args,
                        cfg,
                        norm_stats,
                        device,
                        num_calib_actions=num_calib_actions,
                        sample_idx=sample_idx,
                        output_dir=output_dir,
                    )
                    rows.append(row)
                    write_outputs(output_dir, rows, summarize(rows))

            summary = summarize(rows)
            summary["wall_time_sec"] = float(perf_counter() - run_start)
            write_outputs(output_dir, rows, summary)
            print(f"Summary for {args.method} k={num_calib_actions}")
            print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
