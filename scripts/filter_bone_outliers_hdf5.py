#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


HAND_BONES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
ROOT = 0
MCP_ANCHORS = [5, 9, 13, 17]
MCP_ANCHORS_WITH_WRIST = [0, 5, 9, 13, 17]
JOINT_KEYS = ("world_hand_joints", "world_hand_joints_true", "world_hand_joints_biased")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter long-bone/keypoint outliers from an HDF5 file.")
    parser.add_argument("--input", required=True, help="Input HDF5 path")
    parser.add_argument("--output", default="", help="Output HDF5 path. Default: *_outlier_filtered.hdf5")
    parser.add_argument("--report-dir", default="", help="Directory for CSV/JSON reports")
    parser.add_argument("--bone-long-thr", type=float, default=50.0)
    parser.add_argument("--enable-temporal-bone-jump", action="store_true")
    parser.add_argument("--enable-temporal-joint-jump", action="store_true")
    parser.add_argument("--bone-tjump-thr", type=float, default=6.0)
    parser.add_argument("--joint-tjump-thr", type=float, default=5.0)
    parser.add_argument("--include-root-bones", action="store_true")
    parser.add_argument("--use-wrist-for-alignment", action="store_true")
    parser.add_argument("--min-valid-joints", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Only write reports, do not create filtered HDF5")
    return parser.parse_args()


def finite_joint(P: np.ndarray) -> np.ndarray:
    return np.isfinite(P).all(axis=-1)


def checked_bones(include_root_bones: bool) -> list[tuple[int, int]]:
    if include_root_bones:
        return HAND_BONES_21[:]
    return [(a, b) for a, b in HAND_BONES_21 if a != ROOT and b != ROOT]


def calc_bone_lengths(P: np.ndarray, bones: list[tuple[int, int]]) -> np.ndarray:
    T = P.shape[0]
    L = np.full((T, len(bones)), np.nan, dtype=np.float64)
    finite = finite_joint(P)
    for bi, (a, b) in enumerate(bones):
        ok = finite[:, a] & finite[:, b]
        if np.any(ok):
            L[ok, bi] = np.linalg.norm(P[ok, b] - P[ok, a], axis=-1)
    return L


def robust_refs(L: np.ndarray) -> np.ndarray:
    refs = np.full(L.shape[1], np.nan, dtype=np.float64)
    for bi in range(L.shape[1]):
        v = L[:, bi]
        v = v[np.isfinite(v) & (v > 1e-8)]
        if v.size:
            refs[bi] = float(np.median(v))
    return refs


def rigid_align_row(A: np.ndarray, B: np.ndarray, ids: list[int]) -> np.ndarray:
    if len(ids) < 3:
        return A.copy()
    X = A[ids]
    Y = B[ids]
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        return A.copy()
    xm = X.mean(axis=0)
    ym = Y.mean(axis=0)
    H = (X - xm).T @ (Y - ym)
    try:
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        return (A - xm) @ R + ym
    except np.linalg.LinAlgError:
        return A.copy()


def align_A_to_B_by_palm(A: np.ndarray, B: np.ndarray, anchor_ids: list[int]) -> np.ndarray:
    finite_A = finite_joint(A[None])[0]
    finite_B = finite_joint(B[None])[0]
    ids = [j for j in anchor_ids if finite_A[j] and finite_B[j]]
    if len(ids) >= 3:
        aligned = rigid_align_row(A, B, ids)
        if len(ids) >= 4:
            res = np.linalg.norm(B[ids] - aligned[ids], axis=1)
            order = np.argsort(res)
            keep = [ids[i] for i in order[: max(3, len(ids) - 1)]]
            aligned = rigid_align_row(A, B, keep)
        return aligned

    valid = np.flatnonzero(finite_A & finite_B)
    valid = np.array([j for j in valid if j != ROOT], dtype=np.int64)
    if valid.size >= 4:
        aligned = rigid_align_row(A, B, valid.tolist())
        res = np.linalg.norm(B[valid] - aligned[valid], axis=1)
        order = np.argsort(res)
        keep = valid[order[: max(4, int(np.ceil(0.8 * valid.size)))]]
        return rigid_align_row(A, B, keep.tolist())
    if valid.size >= 1:
        return A + np.median(B[valid] - A[valid], axis=0)
    return A.copy()


def scan_joints(P: np.ndarray, args: argparse.Namespace) -> tuple[set[int], dict[str, int]]:
    T = P.shape[0]
    bad_frames: set[int] = set()
    issue_counts = {
        "joint_nan_or_inf": 0,
        "bone_length_extreme_long": 0,
        "bone_length_two_sided_long_jump": 0,
        "joint_two_sided_jump": 0,
    }

    finite = finite_joint(P)
    bad = ~finite
    for t in np.flatnonzero(bad.any(axis=1)):
        if np.any(bad[t, 1:]):
            bad_frames.add(int(t))
            issue_counts["joint_nan_or_inf"] += int(np.sum(bad[t, 1:]))

    bones = checked_bones(args.include_root_bones)
    L = calc_bone_lengths(P, bones)
    refs = robust_refs(L)
    good_refs = refs[np.isfinite(refs) & (refs > 1e-8)]
    seq_scale = float(np.median(good_refs)) if good_refs.size else 1.0
    if not np.isfinite(seq_scale) or seq_scale <= 1e-8:
        seq_scale = 1.0

    for bi, _bone in enumerate(bones):
        ref = refs[bi]
        if not np.isfinite(ref) or ref <= 1e-8:
            continue
        cur = L[:, bi]
        hit = np.isfinite(cur) & (cur > 1e-8) & ((cur / ref) > args.bone_long_thr)
        frames = np.flatnonzero(hit)
        bad_frames.update(int(t) for t in frames)
        issue_counts["bone_length_extreme_long"] += int(len(frames))

    if args.enable_temporal_bone_jump:
        for bi, _bone in enumerate(bones):
            for t in range(1, T - 1):
                cur, prev, nxt = L[t, bi], L[t - 1, bi], L[t + 1, bi]
                if not (np.isfinite(cur) and np.isfinite(prev) and np.isfinite(nxt)):
                    continue
                if cur <= 1e-8 or prev <= 1e-8 or nxt <= 1e-8:
                    continue
                if min(float(cur / prev), float(cur / nxt)) > args.bone_tjump_thr:
                    bad_frames.add(int(t))
                    issue_counts["bone_length_two_sided_long_jump"] += 1

    anchor_ids = MCP_ANCHORS_WITH_WRIST if args.use_wrist_for_alignment else MCP_ANCHORS
    if args.enable_temporal_joint_jump:
        prev_res = np.full((T, 21), np.nan, dtype=np.float64)
        next_res = np.full((T, 21), np.nan, dtype=np.float64)

        for t in range(1, T):
            if finite[t - 1].sum() < args.min_valid_joints or finite[t].sum() < args.min_valid_joints:
                continue
            aligned_prev = align_A_to_B_by_palm(P[t - 1].astype(np.float64), P[t].astype(np.float64), anchor_ids)
            r = np.linalg.norm(P[t] - aligned_prev, axis=1)
            ok = finite[t - 1] & finite[t] & np.isfinite(r)
            prev_res[t, ok] = r[ok]

        for t in range(0, T - 1):
            if finite[t + 1].sum() < args.min_valid_joints or finite[t].sum() < args.min_valid_joints:
                continue
            aligned_next = align_A_to_B_by_palm(P[t + 1].astype(np.float64), P[t].astype(np.float64), anchor_ids)
            r = np.linalg.norm(P[t] - aligned_next, axis=1)
            ok = finite[t + 1] & finite[t] & np.isfinite(r)
            next_res[t, ok] = r[ok]

        for t in range(T):
            for j in range(1, 21):
                if not finite[t, j]:
                    continue
                rp = prev_res[t, j]
                rn = next_res[t, j]
                if not (np.isfinite(rp) and np.isfinite(rn)):
                    continue
                if min(float(rp), float(rn)) / seq_scale > args.joint_tjump_thr:
                    bad_frames.add(int(t))
                    issue_counts["joint_two_sided_jump"] += 1

    return bad_frames, issue_counts


def sequence_views(group: h5py.Group, args: argparse.Namespace):
    for key in JOINT_KEYS:
        if key not in group:
            continue
        data = group[key]
        if data.ndim == 3 and data.shape[1:] == (21, 3):
            yield key, None, np.asarray(data, dtype=np.float64)
        elif data.ndim == 4 and data.shape[2:] == (21, 3):
            for sample_idx in range(data.shape[0]):
                yield key, sample_idx, np.asarray(data[sample_idx], dtype=np.float64)


def copy_attrs(src: Any, dst: Any) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def filter_dataset(data: h5py.Dataset, keep: np.ndarray) -> np.ndarray:
    arr = data[()]
    T = len(keep)
    if arr.ndim >= 1 and arr.shape[0] == T:
        return arr[keep]
    if arr.ndim >= 2 and arr.shape[1] == T:
        return arr[:, keep]
    return arr


def write_filtered_hdf5(input_path: Path, output_path: Path, group_keep: dict[str, np.ndarray]) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output exists, refusing to overwrite: {output_path}")

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["outlier_filter_source"] = str(input_path)
        for group_name in src.keys():
            src_group = src[group_name]
            dst_group = dst.create_group(group_name)
            copy_attrs(src_group, dst_group)
            keep = group_keep[group_name]
            dst_group.attrs["outlier_filter_original_frames"] = int(len(keep))
            dst_group.attrs["outlier_filter_kept_frames"] = int(np.sum(keep))
            dst_group.attrs["outlier_filter_removed_frames"] = int(len(keep) - np.sum(keep))

            for key, item in src_group.items():
                if isinstance(item, h5py.Dataset):
                    filtered = filter_dataset(item, keep)
                    kwargs = {}
                    if item.compression is not None and filtered.shape != ():
                        kwargs["compression"] = item.compression
                    dst_data = dst_group.create_dataset(key, data=filtered, dtype=item.dtype if item.dtype.kind == "O" else None, **kwargs)
                    copy_attrs(item, dst_data)
                elif isinstance(item, h5py.Group):
                    src.copy(item, dst_group, name=key)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_outlier_filtered.hdf5")
    report_dir = Path(args.report_dir) if args.report_dir else output_path.with_suffix("").with_name(f"{output_path.stem}_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    group_keep: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total_original_frames": 0,
        "total_removed_frames": 0,
        "total_kept_frames": 0,
        "groups": {},
    }

    scan_args = SimpleNamespace(
        bone_long_thr=args.bone_long_thr,
        bone_tjump_thr=args.bone_tjump_thr,
        joint_tjump_thr=args.joint_tjump_thr,
        enable_temporal_bone_jump=args.enable_temporal_bone_jump,
        enable_temporal_joint_jump=args.enable_temporal_joint_jump,
        include_root_bones=args.include_root_bones,
        use_wrist_for_alignment=args.use_wrist_for_alignment,
        min_valid_joints=args.min_valid_joints,
    )

    with h5py.File(input_path, "r") as f:
        group_names = list(f.keys())
        iterator = tqdm(group_names, desc="scan groups") if tqdm is not None else group_names
        for group_name in iterator:
            group = f[group_name]
            T = int(group["world_hand_joints"].shape[0])
            bad_frames: set[int] = set()
            issue_counts_total: dict[str, int] = {}

            for key, sample_idx, P in sequence_views(group, scan_args):
                frames, issue_counts = scan_joints(P, scan_args)
                bad_frames.update(frames)
                label = key if sample_idx is None else f"{key}[{sample_idx}]"
                for issue, count in issue_counts.items():
                    issue_counts_total[issue] = issue_counts_total.get(issue, 0) + count
                    if count:
                        rows.append({
                            "group": group_name,
                            "sequence": label,
                            "issue": issue,
                            "hit_count": count,
                            "bad_frame_count_after_union": len(frames),
                        })

            keep = np.ones(T, dtype=bool)
            if bad_frames:
                keep[np.array(sorted(bad_frames), dtype=np.int64)] = False
            group_keep[group_name] = keep

            removed = int(np.sum(~keep))
            kept = int(np.sum(keep))
            summary["total_original_frames"] += T
            summary["total_removed_frames"] += removed
            summary["total_kept_frames"] += kept
            summary["groups"][group_name] = {
                "original_frames": T,
                "removed_frames": removed,
                "kept_frames": kept,
                "removed_ratio": removed / T if T else 0.0,
                "issue_hit_counts": issue_counts_total,
                "removed_frame_indices": sorted(int(x) for x in bad_frames),
            }
            print(f"group {group_name}: removed {removed}/{T}, kept {kept}")

    if not args.dry_run:
        write_filtered_hdf5(input_path, output_path, group_keep)

    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(report_dir / "issue_counts.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["group", "sequence", "issue", "hit_count", "bad_frame_count_after_union"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone")
    print(f"Output: {output_path}" if not args.dry_run else f"Output: skipped (--dry-run), would write {output_path}")
    print(f"Report: {report_dir}")
    print(
        "Removed "
        f"{summary['total_removed_frames']} / {summary['total_original_frames']} frames "
        f"({summary['total_removed_frames'] / summary['total_original_frames']:.4%})"
    )


if __name__ == "__main__":
    main()
