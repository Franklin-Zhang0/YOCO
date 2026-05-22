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
        description="Export no-finetune best-model predictions for biased/original joints."
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
        "--variants",
        nargs="*",
        default=["best_biased", "best_original", "gt_mano_fk"],
        choices=["best_biased", "best_original", "gt_mano_fk"],
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


def choose_calib_indices(length: int, num_calib: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(length)
    if length <= num_calib:
        return all_indices, all_indices
    rng = np.random.default_rng(seed)
    calib_idx = np.sort(rng.choice(all_indices, size=num_calib, replace=False))
    eval_mask = np.ones(length, dtype=bool)
    eval_mask[calib_idx] = False
    return calib_idx, all_indices[eval_mask]


def batch_similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(source, dtype=np.float64)
    for i, (src, tgt) in enumerate(zip(source, target)):
        src_mean = src.mean(axis=0, keepdims=True)
        tgt_mean = tgt.mean(axis=0, keepdims=True)
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean
        src_var = float(np.sum(src_centered ** 2))
        if src_var < 1e-12:
            aligned[i] = np.broadcast_to(tgt_mean, src.shape)
            continue
        u, _, vh = np.linalg.svd(src_centered.T @ tgt_centered)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            vh[-1] *= -1
            rot = u @ vh
        scale = np.trace(rot.T @ src_centered.T @ tgt_centered) / src_var
        aligned[i] = scale * (src_centered @ rot) + tgt_mean
    return aligned.astype(np.float32)


def compute_joint_metrics(pred_joints: np.ndarray, gt_joints: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_joints, dtype=np.float32)
    gt = np.asarray(gt_joints, dtype=np.float32)
    pa_pred = batch_similarity_align(pred, gt)
    mpjpe = float(np.mean(np.linalg.norm(pred - gt, axis=-1)))
    pa_mpjpe = float(np.mean(np.linalg.norm(pa_pred - gt, axis=-1)))
    tip_mpjpe = float(
        np.mean(np.linalg.norm(pred[:, TIP_JOINT_INDICES] - gt[:, TIP_JOINT_INDICES], axis=-1))
    )
    return {
        "mpjpe": mpjpe,
        "mpjpe_mm": mpjpe * 1000.0,
        "pa_mpjpe": pa_mpjpe,
        "pa_mpjpe_mm": pa_mpjpe * 1000.0,
        "tip_mpjpe": tip_mpjpe,
        "tip_mpjpe_mm": tip_mpjpe * 1000.0,
    }


def reconstruct_joints(
    pred_theta_np: np.ndarray,
    model: torch.nn.Module,
    side: str,
    betas: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    pred_theta = torch.as_tensor(pred_theta_np, dtype=torch.float32, device=device)
    betas_tensor = None
    if betas is not None:
        betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device)
        if betas_tensor.ndim == 1:
            betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)
    with torch.no_grad():
        _, pred_joints = mano_recon(
            pred_theta,
            side=side,
            betas=betas_tensor,
            apply_left_shapedirs_fix=True,
        )
        if hasattr(model, "network") and hasattr(model.network, "finger_scaling"):
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

    with h5py.File(args.data_path, "r") as f:
        session_names = sorted(f.keys(), key=lambda item: int(item) if item.isdigit() else item)
        if args.max_sessions > 0:
            session_names = session_names[: args.max_sessions]
        for session_name in tqdm(session_names, desc="best model sessions"):
            group = f[session_name]
            side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
            betas = np.asarray(group["betas"], dtype=np.float32) if "betas" in group else None
            gt_all = np.asarray(transform_joint3d(read_array(group, args.gt_key)), dtype=np.float32)
            sample_indices = biased_sample_indices(group, args.sample_idx)
            for sample_idx in tqdm(sample_indices, desc=f"{session_name} samples", leave=False, disable=len(sample_indices) <= 1):
                for variant in args.variants:
                    if variant == "best_biased":
                        input_key = "world_hand_joints_biased"
                    elif variant == "best_original":
                        input_key = "world_hand_joints"
                    else:
                        input_key = "mano_angles"

                    effective_sample_idx = sample_idx if input_key.endswith("_biased") else 0
                    if variant == "gt_mano_fk":
                        pred_theta = read_array(group, "mano_angles", sample_idx=0)
                        calib_idx, eval_idx = choose_calib_indices(len(pred_theta), args.num_calib_actions, args.seed)
                        inference_time = 0.0
                    else:
                        x_raw = read_array(group, input_key, sample_idx=effective_sample_idx)
                        calib_idx, eval_idx = choose_calib_indices(len(x_raw), args.num_calib_actions, args.seed)
                        x_proc = np.asarray(process_x(x_raw), dtype=np.float32)
                        start = perf_counter()
                        with torch.no_grad():
                            pred_proc = predict(model, torch.as_tensor(x_proc, dtype=torch.float32, device=device))
                        inference_time = perf_counter() - start
                        pred_theta = np.asarray(process_y_eval(pred_proc.detach().cpu().numpy()), dtype=np.float32)

                    start = perf_counter()
                    pred_joints = reconstruct_joints(pred_theta, model, side=side, betas=betas, device=device)
                    reconstruction_time = perf_counter() - start
                    metrics = compute_joint_metrics(pred_joints[eval_idx], gt_all[eval_idx])

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
