#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py


DEFAULT_DATA_PATH = (
    "hand_keypoint_data/Interhand_val/"
    "InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize finetune/hypernet timing and MPJPE metrics.")
    parser.add_argument(
        "--output-dir",
        default="expdata_annie/finetune_baselines_mac_bone50_filtered",
        help="Finetune output root",
    )
    parser.add_argument("--k", type=int, default=4, help="Calibration K to summarize")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="HDF5 dataset path used to infer left/right side for older metrics files.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "best_world_hand_joints_biased",
            "best_world_hand_joints",
            "best_world_hand_joints_true",
            "gt_mano_fk",
            "lora",
            "full_ft",
            "residual_linear",
            "last_layer",
            "hypernet",
            "hypernet_lora_ft",
        ],
    )
    return parser.parse_args()


def _decode_side(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_group_side_map(data_path: str | None) -> dict[str, str]:
    if not data_path:
        return {}
    path = Path(data_path)
    if not path.exists():
        return {}
    side_by_group: dict[str, str] = {}
    with h5py.File(path, "r") as f:
        for group_name in f.keys():
            group = f[group_name]
            if "mano_side" in group:
                side_by_group[str(group_name)] = _decode_side(group["mano_side"][()])
    return side_by_group


def _row_side(row: dict[str, Any], side_by_group: dict[str, str]) -> str:
    side = row.get("side")
    if side:
        return str(side)
    group = row.get("group")
    if group is not None:
        return side_by_group.get(str(group), "unknown")
    session = str(row.get("session", ""))
    if session:
        return side_by_group.get(session.split("/")[0], "unknown")
    return "unknown"


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows))


def _mean_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def _fmt_float(value: Any, width: int, precision: int) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):>{width}.{precision}f}"
    return f"{'NA':>{width}}"


def _comparison_row(
    method: str,
    k: int,
    side: str,
    rows: list[dict[str, Any]],
    wall_time_sec: float | None,
    metrics_path: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "side": side,
        "k": k,
        "status": "ok",
        "num_units": len(rows),
        "wall_time_sec": wall_time_sec,
        "total_conditioning_time_sec": _sum(rows, "conditioning_time_sec"),
        "mean_conditioning_time_sec": _mean_from_rows(rows, "conditioning_time_sec"),
        "total_adaptation_time_sec": _sum(rows, "adaptation_time_sec"),
        "mean_adaptation_time_sec": _mean_from_rows(rows, "adaptation_time_sec"),
        "total_inference_time_sec": _sum(rows, "inference_time_sec"),
        "mean_inference_time_sec": _mean_from_rows(rows, "inference_time_sec"),
        "total_reconstruction_time_sec": _sum(rows, "reconstruction_time_sec"),
        "mean_reconstruction_time_sec": _mean_from_rows(rows, "reconstruction_time_sec"),
        "mean_trainable_params": _mean_from_rows(rows, "trainable_params"),
        "mpjpe_mm_mean": _mean_from_rows(rows, "mpjpe_mm"),
        "pa_mpjpe_mm_mean": _mean_from_rows(rows, "pa_mpjpe_mm"),
        "tip_mpjpe_mm_mean": _mean_from_rows(rows, "tip_mpjpe_mm"),
        "theta_mse_mean": _mean_from_rows(rows, "theta_mse"),
        "mpjpe_mean": _mean_from_rows(rows, "mpjpe"),
        "pa_mpjpe_mean": _mean_from_rows(rows, "pa_mpjpe"),
        "tip_mpjpe_mean": _mean_from_rows(rows, "tip_mpjpe"),
        "metrics_path": metrics_path,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    side_by_group = _load_group_side_map(args.data_path)
    rows: list[dict[str, Any]] = []

    for method in args.methods:
        metrics_path = root / method / f"k{args.k}" / "metrics.json"
        if not metrics_path.exists():
            rows.append(
                {
                    "method": method,
                    "side": "overall",
                    "k": args.k,
                    "status": "missing",
                    "metrics_path": str(metrics_path),
                }
            )
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sessions = payload.get("sessions", [])
        summary = payload.get("summary", {})

        for session in sessions:
            session["side"] = _row_side(session, side_by_group)

        rows.append(
            _comparison_row(
                method,
                args.k,
                "overall",
                sessions,
                float(summary.get("wall_time_sec", 0.0) or 0.0),
                str(metrics_path),
            )
        )
        for side in ("left", "right", "unknown"):
            side_rows = [session for session in sessions if session.get("side") == side]
            if side_rows:
                rows.append(_comparison_row(method, args.k, side, side_rows, None, str(metrics_path)))

    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"time_comparison_k{args.k}_by_side.json"
    csv_path = root / f"time_comparison_k{args.k}_by_side.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    keys = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Timing comparison saved to {csv_path}")
    print()
    print("method              side      units  wall_s    cond_s   adapt_s   infer_s   recon_s   mpjpe   pa      tip     theta_mse")
    print("------------------  -------  -----  --------  -------  --------  --------  --------  ------  ------  ------  ----------")
    for row in rows:
        if row.get("status") != "ok":
            print(f"{row['method']:<18}  missing ({row['metrics_path']})")
            continue
        print(
            f"{row['method']:<18}  "
            f"{row['side']:<7}  "
            f"{int(row['num_units']):>5}  "
            f"{(row['wall_time_sec'] or 0.0):>8.2f}  "
            f"{row['total_conditioning_time_sec']:>7.2f}  "
            f"{row['total_adaptation_time_sec']:>8.2f}  "
            f"{row['total_inference_time_sec']:>8.2f}  "
            f"{row['total_reconstruction_time_sec']:>8.2f}  "
            f"{_fmt_float(row['mpjpe_mm_mean'], 6, 2)}  "
            f"{_fmt_float(row['pa_mpjpe_mm_mean'], 6, 2)}  "
            f"{_fmt_float(row['tip_mpjpe_mm_mean'], 6, 2)}  "
            f"{_fmt_float(row['theta_mse_mean'], 10, 6)}"
        )


if __name__ == "__main__":
    main()
