#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

sys.path.insert(0, ".")
from calibration_net.utils import mano_recon

HAND_BONES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an HDF5 file and add bad_frame + gt_joint_mask annotations. "
            "Bad GT joints are detected by distance to MANO FK joints."
        )
    )
    parser.add_argument("--input", required=True, help="Input HDF5 path")
    parser.add_argument(
        "--output",
        default="",
        help="Output HDF5 path. Default: *_bad_joint_marked.hdf5",
    )
    parser.add_argument("--report-dir", default="", help="Directory for CSV/JSON reports")
    parser.add_argument(
        "--compare-summary-json",
        default="",
        help="Optional filter_bone_outliers_hdf5.py summary.json used to save report-vs-direct difference masks.",
    )
    parser.add_argument("--gt-key", default="world_hand_joints")
    parser.add_argument("--mano-angles-key", default="mano_angles")
    parser.add_argument("--global-orient-key", default="global_orient")
    parser.add_argument("--trans-key", default="trans")
    parser.add_argument("--betas-key", default="betas")
    parser.add_argument("--side-key", default="mano_side")
    parser.add_argument(
        "--point-distance-thr-mm",
        type=float,
        default=200.0,
        help="Joint distance threshold to MANO FK, in millimeters. Default 200mm = 20cm.",
    )
    parser.add_argument(
        "--bone-long-thr",
        type=float,
        default=20.0,
        help="Bone length ratio threshold for the comparison bone-endpoint mask.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def decode_hdf5_string(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def copy_attrs(src: Any, dst: Any) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group(src_group: h5py.Group, dst_group: h5py.Group) -> None:
    copy_attrs(src_group, dst_group)
    for key, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            kwargs = {}
            if item.compression is not None and item.shape != ():
                kwargs["compression"] = item.compression
            dst_data = dst_group.create_dataset(
                key,
                data=item[()],
                dtype=item.dtype if item.dtype.kind == "O" else None,
                **kwargs,
            )
            copy_attrs(item, dst_data)
        elif isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            copy_group(item, child)


def reconstruct_mano_fk_joints(
    group: h5py.Group,
    *,
    mano_angles_key: str,
    global_orient_key: str,
    trans_key: str,
    betas_key: str,
    side_key: str,
    device: torch.device,
) -> np.ndarray:
    mano_angles = torch.as_tensor(
        np.asarray(group[mano_angles_key], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    global_orient = torch.as_tensor(
        np.asarray(group[global_orient_key], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    trans = torch.as_tensor(
        np.asarray(group[trans_key], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    betas = torch.as_tensor(
        np.asarray(group[betas_key], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    if betas.ndim == 1:
        betas = betas.unsqueeze(0).expand(mano_angles.shape[0], -1)
    side = decode_hdf5_string(group[side_key][0] if len(group[side_key]) > 0 else group[side_key][()])

    with torch.no_grad():
        _, joints = mano_recon(
            mano_angles,
            mano_orient=global_orient,
            transl=trans,
            side=side,
            betas=betas,
            apply_left_shapedirs_fix=True,
        )
    return joints.detach().cpu().numpy().astype(np.float32)


def mark_group(
    gt_joints: np.ndarray,
    mano_fk_joints: np.ndarray,
    *,
    point_distance_thr_m: float,
    bone_long_thr: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    if gt_joints.shape != mano_fk_joints.shape:
        raise ValueError(f"GT/FK shape mismatch: {gt_joints.shape} vs {mano_fk_joints.shape}")
    if gt_joints.ndim != 3 or gt_joints.shape[1:] != (21, 3):
        raise ValueError(f"expected GT joints shape (T,21,3), got {gt_joints.shape}")

    finite = np.isfinite(gt_joints).all(axis=-1) & np.isfinite(mano_fk_joints).all(axis=-1)
    dist = np.linalg.norm(gt_joints - mano_fk_joints, axis=-1)
    gt_joint_mask = (~finite) | (dist > point_distance_thr_m)
    bad_frame = gt_joint_mask.any(axis=1)
    gt_joint_mask_bone_endpoint = mark_bone_endpoint_outliers(
        gt_joints,
        dist,
        finite,
        point_distance_thr_m=point_distance_thr_m,
        bone_long_thr=bone_long_thr,
    )
    bad_frame_bone_endpoint = gt_joint_mask_bone_endpoint.any(axis=1)
    gt_joint_mask_direct_only = gt_joint_mask & ~gt_joint_mask_bone_endpoint
    gt_joint_mask_bone_endpoint_only = gt_joint_mask_bone_endpoint & ~gt_joint_mask
    bad_frame_direct_only = bad_frame & ~bad_frame_bone_endpoint
    bad_frame_bone_endpoint_only = bad_frame_bone_endpoint & ~bad_frame

    rows: list[dict[str, Any]] = []
    bad_t, bad_j = np.nonzero(gt_joint_mask | gt_joint_mask_bone_endpoint)
    for t, j in zip(bad_t.tolist(), bad_j.tolist()):
        rows.append(
            {
                "frame": int(t),
                "joint": int(j),
                "distance_mm": float(dist[t, j] * 1000.0) if np.isfinite(dist[t, j]) else float("inf"),
                "finite": bool(finite[t, j]),
                "direct": bool(gt_joint_mask[t, j]),
                "bone_endpoint": bool(gt_joint_mask_bone_endpoint[t, j]),
                "direct_only": bool(gt_joint_mask_direct_only[t, j]),
                "bone_endpoint_only": bool(gt_joint_mask_bone_endpoint_only[t, j]),
            }
        )

    summary = {
        "original_frames": int(gt_joints.shape[0]),
        "bad_frames": int(np.sum(bad_frame)),
        "bad_joint_entries": int(np.sum(gt_joint_mask)),
        "bone_endpoint_bad_frames": int(np.sum(bad_frame_bone_endpoint)),
        "bone_endpoint_bad_joint_entries": int(np.sum(gt_joint_mask_bone_endpoint)),
        "direct_only_bad_frames": int(np.sum(bad_frame_direct_only)),
        "direct_only_bad_joint_entries": int(np.sum(gt_joint_mask_direct_only)),
        "bone_endpoint_only_bad_frames": int(np.sum(bad_frame_bone_endpoint_only)),
        "bone_endpoint_only_bad_joint_entries": int(np.sum(gt_joint_mask_bone_endpoint_only)),
        "max_distance_mm": float(np.nanmax(dist) * 1000.0),
        "p95_distance_mm": float(np.nanpercentile(dist, 95) * 1000.0),
        "mean_distance_mm": float(np.nanmean(dist) * 1000.0),
    }
    masks = {
        "bad_frame": bad_frame,
        "gt_joint_mask": gt_joint_mask,
        "bad_frame_bone_endpoint": bad_frame_bone_endpoint,
        "gt_joint_mask_bone_endpoint": gt_joint_mask_bone_endpoint,
        "bad_frame_direct_only": bad_frame_direct_only,
        "gt_joint_mask_direct_only": gt_joint_mask_direct_only,
        "bad_frame_bone_endpoint_only": bad_frame_bone_endpoint_only,
        "gt_joint_mask_bone_endpoint_only": gt_joint_mask_bone_endpoint_only,
    }
    return masks, summary, rows


def mark_bone_endpoint_outliers(
    gt_joints: np.ndarray,
    dist_to_fk: np.ndarray,
    finite_to_fk: np.ndarray,
    *,
    point_distance_thr_m: float,
    bone_long_thr: float,
) -> np.ndarray:
    finite_gt = np.isfinite(gt_joints).all(axis=-1)
    lengths = np.full((gt_joints.shape[0], len(HAND_BONES_21)), np.nan, dtype=np.float64)
    for bi, (a, b) in enumerate(HAND_BONES_21):
        ok = finite_gt[:, a] & finite_gt[:, b]
        if np.any(ok):
            lengths[ok, bi] = np.linalg.norm(gt_joints[ok, b] - gt_joints[ok, a], axis=-1)

    refs = np.full(len(HAND_BONES_21), np.nan, dtype=np.float64)
    for bi in range(len(HAND_BONES_21)):
        vals = lengths[:, bi]
        vals = vals[np.isfinite(vals) & (vals > 1e-8)]
        if vals.size:
            refs[bi] = float(np.median(vals))

    mask = np.zeros(gt_joints.shape[:2], dtype=bool)
    far_from_fk = (~finite_to_fk) | (dist_to_fk > point_distance_thr_m)
    for bi, (a, b) in enumerate(HAND_BONES_21):
        ref = refs[bi]
        if not np.isfinite(ref) or ref <= 1e-8:
            continue
        hit = np.isfinite(lengths[:, bi]) & (lengths[:, bi] > 1e-8) & ((lengths[:, bi] / ref) > bone_long_thr)
        if np.any(hit):
            mask[hit, a] |= far_from_fk[hit, a]
            mask[hit, b] |= far_from_fk[hit, b]
    return mask


def write_marked_hdf5(
    input_path: Path,
    output_path: Path,
    group_masks: dict[str, dict[str, np.ndarray]],
    attrs: dict[str, Any],
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output exists, refusing to overwrite: {output_path}")
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src, dst)
        for key, value in attrs.items():
            dst.attrs[key] = value
        for group_name in src.keys():
            src_group = src[group_name]
            dst_group = dst.create_group(group_name)
            copy_group(src_group, dst_group)
            for mask_name, mask in group_masks[group_name].items():
                ds = dst_group.create_dataset(mask_name, data=mask.astype(np.bool_))
                ds.attrs["description"] = mask_description(mask_name)


def mask_description(mask_name: str) -> str:
    descriptions = {
        "bad_frame": "True means this frame has at least one GT joint farther than the threshold from MANO FK.",
        "gt_joint_mask": "Boolean mask over world_hand_joints. True means this joint is farther than the threshold from MANO FK.",
        "bad_frame_bone_endpoint": "Frame-level version of gt_joint_mask_bone_endpoint.",
        "gt_joint_mask_bone_endpoint": "Bone-endpoint comparison mask: endpoint belongs to an extreme-length bone and is farther than the threshold from MANO FK.",
        "bad_frame_direct_only": "True for frames detected by direct joint-to-FK mask but not by the bone-endpoint mask.",
        "gt_joint_mask_direct_only": "Point-level difference: direct joint-to-FK mask only.",
        "bad_frame_bone_endpoint_only": "True for frames detected by bone-endpoint mask but not by the direct joint-to-FK mask.",
        "gt_joint_mask_bone_endpoint_only": "Point-level difference: bone-endpoint mask only.",
        "bad_frame_filter_report": "Frames removed by the optional compare-summary-json report.",
        "gt_joint_mask_filter_report": "All joints are marked on frames removed by the optional compare-summary-json report.",
        "bad_frame_direct_only_vs_filter_report": "Frames detected by direct joint-to-FK mask but not removed by the compare-summary-json report.",
        "gt_joint_mask_direct_only_vs_filter_report": "Direct bad joints on frames detected only by direct joint-to-FK mask compared with the report.",
        "bad_frame_filter_report_only_vs_direct": "Frames removed by the compare-summary-json report but not detected by direct joint-to-FK mask.",
        "gt_joint_mask_filter_report_only_vs_direct": "All joints are marked on frames removed only by the compare-summary-json report.",
    }
    return descriptions.get(mask_name, "Bad joint/frame mask generated by mark_gt_joint_outliers_hdf5.py.")


def read_removed_frames(summary_json: str) -> dict[str, set[int]]:
    if not summary_json:
        return {}
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        str(group_name): set(int(x) for x in payload.get("removed_frame_indices", []))
        for group_name, payload in summary.get("groups", {}).items()
    }


def add_filter_report_comparison_masks(
    masks: dict[str, np.ndarray],
    removed_frames: set[int],
) -> dict[str, int]:
    bad_frame = masks["bad_frame"]
    gt_joint_mask = masks["gt_joint_mask"]
    report_bad_frame = np.zeros_like(bad_frame, dtype=bool)
    valid_removed = [idx for idx in removed_frames if 0 <= idx < len(report_bad_frame)]
    report_bad_frame[valid_removed] = True

    direct_only = bad_frame & ~report_bad_frame
    report_only = report_bad_frame & ~bad_frame

    masks["bad_frame_filter_report"] = report_bad_frame
    masks["gt_joint_mask_filter_report"] = np.broadcast_to(report_bad_frame[:, None], gt_joint_mask.shape).copy()
    masks["bad_frame_direct_only_vs_filter_report"] = direct_only
    masks["gt_joint_mask_direct_only_vs_filter_report"] = gt_joint_mask & direct_only[:, None]
    masks["bad_frame_filter_report_only_vs_direct"] = report_only
    masks["gt_joint_mask_filter_report_only_vs_direct"] = np.broadcast_to(report_only[:, None], gt_joint_mask.shape).copy()

    return {
        "filter_report_bad_frames": int(np.sum(report_bad_frame)),
        "direct_only_vs_filter_report_bad_frames": int(np.sum(direct_only)),
        "direct_only_vs_filter_report_bad_joint_entries": int(np.sum(masks["gt_joint_mask_direct_only_vs_filter_report"])),
        "filter_report_only_vs_direct_bad_frames": int(np.sum(report_only)),
        "filter_report_only_vs_direct_bad_joint_entries": int(np.sum(masks["gt_joint_mask_filter_report_only_vs_direct"])),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_bad_joint_marked.hdf5")
    )
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else output_path.with_suffix("").with_name(f"{output_path.stem}_report")
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    point_distance_thr_m = float(args.point_distance_thr_mm) / 1000.0
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    removed_by_group = read_removed_frames(args.compare_summary_json)

    group_masks: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "gt_key": args.gt_key,
        "reference": "mano_fk",
        "point_distance_thr_mm": float(args.point_distance_thr_mm),
        "bone_long_thr": float(args.bone_long_thr),
        "compare_summary_json": args.compare_summary_json,
        "total_original_frames": 0,
        "total_bad_frames": 0,
        "total_bad_joint_entries": 0,
        "total_bone_endpoint_bad_frames": 0,
        "total_bone_endpoint_bad_joint_entries": 0,
        "total_direct_only_bad_frames": 0,
        "total_direct_only_bad_joint_entries": 0,
        "total_bone_endpoint_only_bad_frames": 0,
        "total_bone_endpoint_only_bad_joint_entries": 0,
        "total_filter_report_bad_frames": 0,
        "total_direct_only_vs_filter_report_bad_frames": 0,
        "total_direct_only_vs_filter_report_bad_joint_entries": 0,
        "total_filter_report_only_vs_direct_bad_frames": 0,
        "total_filter_report_only_vs_direct_bad_joint_entries": 0,
        "groups": {},
    }
    detail_rows: list[dict[str, Any]] = []

    with h5py.File(input_path, "r") as f:
        group_names = list(f.keys())
        iterator = tqdm(group_names, desc="mark groups") if tqdm is not None else group_names
        for group_name in iterator:
            group = f[group_name]
            required = [
                args.gt_key,
                args.mano_angles_key,
                args.global_orient_key,
                args.trans_key,
                args.betas_key,
                args.side_key,
            ]
            missing = [key for key in required if key not in group]
            if missing:
                raise KeyError(f"Missing keys in group {group_name}: {missing}")

            gt_joints = np.asarray(group[args.gt_key], dtype=np.float32)
            mano_fk_joints = reconstruct_mano_fk_joints(
                group,
                mano_angles_key=args.mano_angles_key,
                global_orient_key=args.global_orient_key,
                trans_key=args.trans_key,
                betas_key=args.betas_key,
                side_key=args.side_key,
                device=device,
            )
            masks, group_summary, rows = mark_group(
                gt_joints,
                mano_fk_joints,
                point_distance_thr_m=point_distance_thr_m,
                bone_long_thr=float(args.bone_long_thr),
            )
            if group_name in removed_by_group:
                comparison_summary = add_filter_report_comparison_masks(masks, removed_by_group[group_name])
                group_summary.update(comparison_summary)
                summary["total_filter_report_bad_frames"] += comparison_summary["filter_report_bad_frames"]
                summary["total_direct_only_vs_filter_report_bad_frames"] += comparison_summary[
                    "direct_only_vs_filter_report_bad_frames"
                ]
                summary["total_direct_only_vs_filter_report_bad_joint_entries"] += comparison_summary[
                    "direct_only_vs_filter_report_bad_joint_entries"
                ]
                summary["total_filter_report_only_vs_direct_bad_frames"] += comparison_summary[
                    "filter_report_only_vs_direct_bad_frames"
                ]
                summary["total_filter_report_only_vs_direct_bad_joint_entries"] += comparison_summary[
                    "filter_report_only_vs_direct_bad_joint_entries"
                ]
            group_masks[group_name] = masks
            summary["groups"][group_name] = group_summary
            summary["total_original_frames"] += group_summary["original_frames"]
            summary["total_bad_frames"] += group_summary["bad_frames"]
            summary["total_bad_joint_entries"] += group_summary["bad_joint_entries"]
            summary["total_bone_endpoint_bad_frames"] += group_summary["bone_endpoint_bad_frames"]
            summary["total_bone_endpoint_bad_joint_entries"] += group_summary["bone_endpoint_bad_joint_entries"]
            summary["total_direct_only_bad_frames"] += group_summary["direct_only_bad_frames"]
            summary["total_direct_only_bad_joint_entries"] += group_summary["direct_only_bad_joint_entries"]
            summary["total_bone_endpoint_only_bad_frames"] += group_summary["bone_endpoint_only_bad_frames"]
            summary["total_bone_endpoint_only_bad_joint_entries"] += group_summary["bone_endpoint_only_bad_joint_entries"]
            for row in rows:
                row["group"] = group_name
                detail_rows.append(row)
            print(
                f"group {group_name}: bad_frame {group_summary['bad_frames']}/"
                f"{group_summary['original_frames']}, bad joints {group_summary['bad_joint_entries']}; "
                f"direct-only frames {group_summary['direct_only_bad_frames']}, "
                f"bone-endpoint-only frames {group_summary['bone_endpoint_only_bad_frames']}"
            )

    if not args.dry_run:
        write_marked_hdf5(
            input_path,
            output_path,
            group_masks,
            attrs={
                "bad_joint_marker_source": str(input_path),
                "bad_joint_marker_gt_key": args.gt_key,
                "bad_joint_marker_reference": "mano_fk",
                "bad_joint_marker_point_distance_thr_mm": float(args.point_distance_thr_mm),
                "bad_joint_marker_bone_long_thr": float(args.bone_long_thr),
            },
        )

    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    if detail_rows:
        keys = sorted({key for row in detail_rows for key in row.keys()})
        with open(report_dir / "bad_joints.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(detail_rows)

    print("\nDone")
    print(f"Output: {output_path}" if not args.dry_run else f"Output skipped (--dry-run), would write {output_path}")
    print(f"Report: {report_dir}")
    print(
        "Bad frames "
        f"{summary['total_bad_frames']} / {summary['total_original_frames']}; "
        f"bad joint entries {summary['total_bad_joint_entries']}"
    )
    print(
        "Diff frames direct-only / bone-endpoint-only: "
        f"{summary['total_direct_only_bad_frames']} / "
        f"{summary['total_bone_endpoint_only_bad_frames']}"
    )


if __name__ == "__main__":
    main()
