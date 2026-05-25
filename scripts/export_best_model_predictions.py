#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, ".")

from calibration_net.policy.baseline_finetune import predict
from calibration_net.policy.model import load_model
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.utils import mano_recon, scale_hand_fingers, transform_joint3d


TIP_JOINT_INDICES = [4, 8, 12, 16, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export no-finetune best-model predictions for HDF5 joint keys."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-calib-actions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-idx", type=int, default=-1)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--gt-key", default="world_hand_joints")
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
        help="Only export/evaluate one MANO side. Default: right, matching Franklin eval.",
    )
    parser.add_argument(
        "--exclude-calib-from-eval",
        action="store_true",
        help="Exclude calibration frames from metrics. Default matches Franklin eval: include them.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "best_world_hand_joints_biased",
            "best_world_hand_joints",
            "best_world_hand_joints_true",
            "gt_mano_fk",
        ],
        choices=[
            "best_world_hand_joints_biased",
            "best_world_hand_joints",
            "best_world_hand_joints_true",
            "best_biased",
            "best_original",
            "best_true",
            "gt_mano_fk",
        ],
    )
    return parser.parse_args()


def decode_side(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "right"
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value).lower()


def read_array(group: h5py.Group, key: str, sample_idx: int = 0) -> np.ndarray:
    array = np.asarray(group[key])
    if array.ndim >= 4 and key.endswith("_biased"):
        sample_idx = min(max(sample_idx, 0), array.shape[0] - 1)
        array = array[sample_idx]
    return array.astype(np.float32, copy=False)


def biased_sample_indices(group: h5py.Group, sample_idx: int) -> list[int]:
    if "world_hand_joints_biased" not in group:
        return [0]
    shape = group["world_hand_joints_biased"].shape
    if len(shape) >= 4:
        if sample_idx < 0:
            return list(range(shape[0]))
        return [min(sample_idx, shape[0] - 1)]
    return [0]


def choose_calib_indices(
    length: int,
    num_calib: int,
    rng: np.random.Generator,
    exclude_calib_from_eval: bool,
) -> tuple[np.ndarray, np.ndarray]:
    return choose_calib_indices_with_mask(length, num_calib, rng, None, exclude_calib_from_eval)


def choose_calib_indices_with_mask(
    length: int,
    num_calib: int,
    rng: np.random.Generator,
    bad_frame_mask: np.ndarray | None,
    exclude_calib_from_eval: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(length)
    good_indices = all_indices if bad_frame_mask is None else all_indices[~bad_frame_mask]
    if len(good_indices) == 0:
        raise ValueError("No non-bad frames available for calibration")
    if len(good_indices) <= num_calib:
        return good_indices, all_indices
    calib_idx = np.sort(rng.choice(good_indices, size=num_calib, replace=False))
    eval_mask = np.ones(length, dtype=bool)
    if exclude_calib_from_eval:
        eval_mask[calib_idx] = False
    return calib_idx, all_indices[eval_mask]


def read_bad_frame_mask(group: h5py.Group, length: int) -> np.ndarray | None:
    if "bad_frame" not in group:
        return None
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


def select_mano_betas(
    group: h5py.Group,
    gt_key: str,
    pred_beta_source: str,
) -> tuple[np.ndarray | None, str]:
    source = pred_beta_source
    if source == "auto":
        source = "neutral" if gt_key == "world_hand_joints_true" else "group"
    if source == "neutral":
        return None, source
    if "betas" not in group:
        return None, "missing_group"
    return np.asarray(group["betas"], dtype=np.float32), source


def _similarity_align_one(
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
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
    return np.stack(
        [_similarity_align_one(src, tgt) for src, tgt in zip(source, target)],
        axis=0,
    ).astype(np.float32)


def batch_similarity_align_masked(
    source: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    if valid_mask is None:
        return batch_similarity_align(source, target)
    return np.stack(
        [
            _similarity_align_one(src, tgt, valid)
            for src, tgt, valid in zip(source, target, valid_mask)
        ],
        axis=0,
    ).astype(np.float32)


def compute_joint_metrics(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    valid_joint_mask: np.ndarray | None = None,
) -> dict[str, float]:
    pred = np.asarray(pred_joints, dtype=np.float32)
    gt = np.asarray(gt_joints, dtype=np.float32)
    if valid_joint_mask is not None:
        valid_joint_mask = np.asarray(valid_joint_mask, dtype=bool)
        if valid_joint_mask.shape != pred.shape[:2]:
            raise ValueError(
                f"valid_joint_mask shape {valid_joint_mask.shape} does not match joints shape {pred.shape[:2]}"
            )
    valid = np.ones(pred.shape[:2], dtype=bool) if valid_joint_mask is None else valid_joint_mask
    if not np.any(valid):
        raise ValueError("No valid joints left for metrics")
    pa_pred = batch_similarity_align_masked(pred, gt, valid_joint_mask)
    err = np.linalg.norm(pred - gt, axis=-1)
    pa_err = np.linalg.norm(pa_pred - gt, axis=-1)
    tip_valid = valid[:, TIP_JOINT_INDICES]
    tip_err = np.linalg.norm(pred[:, TIP_JOINT_INDICES] - gt[:, TIP_JOINT_INDICES], axis=-1)
    if not np.any(tip_valid):
        tip_valid = valid
        tip_err = err
    mpjpe = float(np.mean(err[valid]))
    pa_mpjpe = float(np.mean(pa_err[valid]))
    tip_mpjpe = float(np.mean(tip_err[tip_valid]))
    return {
        "mpjpe": mpjpe,
        "mpjpe_mm": mpjpe * 1000.0,
        "pa_mpjpe": pa_mpjpe,
        "pa_mpjpe_mm": pa_mpjpe * 1000.0,
        "tip_mpjpe": tip_mpjpe,
        "tip_mpjpe_mm": tip_mpjpe * 1000.0,
        "num_metric_joints": int(np.sum(valid)),
        "num_masked_bad_joints": int(valid.size - np.sum(valid)),
    }


def compute_theta_metrics(pred_theta: np.ndarray, gt_theta: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_theta, dtype=np.float32)
    gt = np.asarray(gt_theta, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred theta shape {pred.shape} does not match gt theta shape {gt.shape}")
    return {
        "theta_mse": float(np.mean((pred - gt) ** 2)),
    }


def reconstruct_joints(
    pred_theta_np: np.ndarray,
    model: torch.nn.Module,
    side: str,
    betas: np.ndarray | None,
    device: torch.device,
    global_orient: np.ndarray | None = None,
    trans: np.ndarray | None = None,
    apply_finger_scaling: bool = True,
) -> np.ndarray:
    pred_theta = torch.as_tensor(pred_theta_np, dtype=torch.float32, device=device)
    betas_tensor = None
    if betas is not None:
        betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device)
        if betas_tensor.ndim == 1:
            betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)
    orient_tensor = None
    if global_orient is not None:
        orient_tensor = torch.as_tensor(global_orient, dtype=torch.float32, device=device)
    trans_tensor = None
    if trans is not None:
        trans_tensor = torch.as_tensor(trans, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, pred_joints = mano_recon(
            pred_theta,
            mano_orient=orient_tensor,
            transl=trans_tensor,
            side=side,
            betas=betas_tensor,
            apply_left_shapedirs_fix=True,
        )
        if apply_finger_scaling and hasattr(model, "network") and hasattr(model.network, "finger_scaling"):
            pred_joints = scale_hand_fingers(pred_joints, model.network.finger_scaling)
        pred_local = transform_joint3d(pred_joints)
        pred_local = torch.as_tensor(pred_local, dtype=torch.float32, device=device)
    return pred_local.detach().cpu().numpy().astype(np.float32)


def write_predictions(
    predictions_path: Path,
    session_name: str,
    sample_idx: int,
    pred_theta: np.ndarray,
    pred_joints: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    attrs: dict[str, Any],
) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(predictions_path, "a") as f:
        group = f.require_group(session_name)
        sample_group = group.require_group(f"sample_{sample_idx}")
        for key in ("pred_theta", "pred_joints", "calib_idx", "eval_idx"):
            if key in sample_group:
                del sample_group[key]
        sample_group.create_dataset("pred_theta", data=pred_theta, compression="gzip")
        sample_group.create_dataset("pred_joints", data=pred_joints, compression="gzip")
        sample_group.create_dataset("calib_idx", data=calib_idx.astype(np.int64))
        sample_group.create_dataset("eval_idx", data=eval_idx.astype(np.int64))
        for key, value in attrs.items():
            sample_group.attrs[key] = value


def summarize(rows: list[dict[str, Any]], wall_time_sec: float) -> dict[str, float]:
    summary: dict[str, float] = {
        "num_units": len(rows),
        "wall_time_sec": float(wall_time_sec),
    }
    for key in (
        "conditioning_time_sec",
        "adaptation_time_sec",
        "inference_time_sec",
        "reconstruction_time_sec",
        "mpjpe",
        "mpjpe_mm",
        "pa_mpjpe",
        "pa_mpjpe_mm",
        "tip_mpjpe",
        "tip_mpjpe_mm",
        "theta_mse",
        "num_metric_joints",
        "num_masked_bad_joints",
    ):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            summary[f"{key}_total"] = float(np.sum(values))
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def write_metrics(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"sessions": rows, "summary": summary}, f, indent=2, sort_keys=True)
    if rows:
        keys = sorted({key for row in rows for key in row.keys()})
        with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm_stats, cfg = load_model(args.model_dir, device=str(device))
    model.to(device)
    model.eval()
    process_x = build_preprocessing(cfg["dataset"]["preprocessing_x"], "x", norm_stats)
    process_y_eval = build_preprocessing(
        [process if process != "normalize" else "denormalize" for process in cfg["dataset"]["preprocessing_y"]],
        "y",
        norm_stats,
    )

    rows_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in args.variants}
    run_start_by_variant = {variant: perf_counter() for variant in args.variants}
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.data_path, "r") as f:
        session_names = sorted(f.keys(), key=lambda item: int(item) if item.isdigit() else item)
        if args.max_sessions > 0:
            session_names = session_names[: args.max_sessions]
        for session_name in tqdm(session_names, desc="best model sessions"):
            group = f[session_name]
            side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
            if args.side_filter != "all" and side != args.side_filter:
                continue
            betas, beta_source = select_mano_betas(group, args.gt_key, args.pred_beta_source)
            gt_all = np.asarray(transform_joint3d(read_array(group, args.gt_key)), dtype=np.float32)
            bad_frame_mask = read_bad_frame_mask(group, len(gt_all))
            valid_joint_mask_all = read_valid_joint_mask(group, len(gt_all))
            global_orient = np.asarray(group["global_orient"], dtype=np.float32) if "global_orient" in group else None
            trans = np.asarray(group["trans"], dtype=np.float32) if "trans" in group else None
            if "mano_angles" not in group:
                raise KeyError(f"'mano_angles' not found in session {session_name}")
            gt_theta_all = read_array(group, "mano_angles", sample_idx=0)
            sample_indices = biased_sample_indices(group, args.sample_idx)
            for sample_idx in tqdm(sample_indices, desc=f"{session_name} samples", leave=False, disable=len(sample_indices) <= 1):
                calib_idx, eval_idx = choose_calib_indices_with_mask(
                    len(gt_all),
                    args.num_calib_actions,
                    rng,
                    bad_frame_mask,
                    args.exclude_calib_from_eval,
                )
                for variant in args.variants:
                    if variant in {"best_world_hand_joints_biased", "best_biased"}:
                        input_key = "world_hand_joints_biased"
                    elif variant in {"best_world_hand_joints", "best_original"}:
                        input_key = "world_hand_joints"
                    elif variant in {"best_world_hand_joints_true", "best_true"}:
                        input_key = "world_hand_joints_true"
                    else:
                        input_key = "mano_angles"

                    effective_sample_idx = sample_idx if input_key.endswith("_biased") else 0
                    if variant == "gt_mano_fk":
                        pred_theta = read_array(group, "mano_angles", sample_idx=0)
                        inference_time = 0.0
                    else:
                        x_raw = read_array(group, input_key, sample_idx=effective_sample_idx)
                        if len(x_raw) != len(gt_all):
                            raise ValueError(
                                f"Length mismatch for {session_name}/{variant}: "
                                f"input={len(x_raw)} gt={len(gt_all)}"
                            )
                        x_proc = np.asarray(process_x(x_raw), dtype=np.float32)
                        start = perf_counter()
                        with torch.no_grad():
                            pred_proc = predict(model, torch.as_tensor(x_proc, dtype=torch.float32, device=device))
                        inference_time = perf_counter() - start
                        pred_theta = np.asarray(process_y_eval(pred_proc.detach().cpu().numpy()), dtype=np.float32)

                    start = perf_counter()
                    pred_joints = reconstruct_joints(
                        pred_theta,
                        model,
                        side=side,
                        betas=betas,
                        device=device,
                        global_orient=global_orient if variant == "gt_mano_fk" else None,
                        trans=trans if variant == "gt_mano_fk" else None,
                        apply_finger_scaling=variant != "gt_mano_fk",
                    )
                    reconstruction_time = perf_counter() - start
                    metrics = compute_joint_metrics(
                        pred_joints[eval_idx],
                        gt_all[eval_idx],
                        valid_joint_mask_all[eval_idx] if valid_joint_mask_all is not None else None,
                    )
                    metrics.update(
                        compute_theta_metrics(
                            pred_theta[eval_idx],
                            gt_theta_all[eval_idx],
                        )
                    )

                    output_dir = Path(args.output_dir) / variant / f"k{args.num_calib_actions}"
                    write_predictions(
                        output_dir / "predictions.hdf5",
                        session_name=session_name,
                        sample_idx=sample_idx,
                        pred_theta=pred_theta,
                        pred_joints=pred_joints,
                        calib_idx=calib_idx,
                        eval_idx=eval_idx,
                        attrs={
                            "method": variant,
                            "input_key": input_key,
                            "gt_key": args.gt_key,
                            "pred_beta_source": beta_source,
                            "side_filter": args.side_filter,
                            "exclude_calib_from_eval": bool(args.exclude_calib_from_eval),
                            "num_calib_actions": int(args.num_calib_actions),
                            "model_dir": str(args.model_dir),
                        },
                    )
                    rows_by_variant[variant].append(
                        {
                            "session": f"{session_name}/sample_{sample_idx}",
                            "group": session_name,
                            "side": side,
                            "sample_idx": int(sample_idx),
                            "method": variant,
                            "input_key": input_key,
                            "gt_key": args.gt_key,
                            "pred_beta_source": beta_source,
                            "side_filter": args.side_filter,
                            "exclude_calib_from_eval": bool(args.exclude_calib_from_eval),
                            "num_calib_actions": int(args.num_calib_actions),
                            "num_calib_samples": int(len(calib_idx)),
                            "num_eval_samples": int(len(eval_idx)),
                            "conditioning_time_sec": 0.0,
                            "adaptation_time_sec": 0.0,
                            "inference_time_sec": float(inference_time),
                            "reconstruction_time_sec": float(reconstruction_time),
                            **metrics,
                        }
                    )

    for variant, rows in rows_by_variant.items():
        output_dir = Path(args.output_dir) / variant / f"k{args.num_calib_actions}"
        summary = summarize(rows, perf_counter() - run_start_by_variant[variant])
        write_metrics(output_dir, rows, summary)
        print(f"Saved {variant} to {output_dir}")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
