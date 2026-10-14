"""Verify complete pose exports against paper or release-checkpoint results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_net.utils import transform_joint3d
from scripts.export_hypernet_finetune_predictions import (
    biased_sample_indices,
    choose_calib_indices,
    compute_joint_metrics,
    decode_side,
    make_calibration_rng,
    read_bad_frame_mask,
    read_valid_joint_mask,
)

REFERENCES = {
    "interhand": {
        "units": 20,
        "mpjpe_mm": 15.63970100445052,
        "pa_mpjpe_mm": 10.454345230634013,
        "tip_mpjpe_mm": 23.28891794507702,
        "aucj": 0.7015746623637008,
        "pinch_distance_l2_mm": 34.89984022453427,
    },
    "avp": {
        "units": 3,
        "mpjpe_mm": 17.751970224910313,
        "pa_mpjpe_mm": 13.140390316645304,
        "tip_mpjpe_mm": 31.73461080425315,
        "aucj": 0.6572197596585589,
        "pinch_distance_l2_mm": 42.786240577697754,
    },
}


INTERHAND_BASELINE_REFERENCES = {
    "full_ft": {
        "units": 20,
        "mpjpe_mm": 18.109690475588042,
        "pa_mpjpe_mm": 12.76903495917954,
        "tip_mpjpe_mm": 29.21202201396227,
        "aucj": 0.6681493015529849,
        "pinch_distance_l2_mm": 37.11298452690244,
    },
    "last_layer": {
        "units": 20,
        "mpjpe_mm": 16.62146127782762,
        "pa_mpjpe_mm": 12.014720981116929,
        "tip_mpjpe_mm": 25.921316041300695,
        "aucj": 0.6922072981470931,
        "pinch_distance_l2_mm": 39.17637433235844,
    },
    "ridge": {
        "units": 20,
        "mpjpe_mm": 17.620393276835483,
        "pa_mpjpe_mm": 13.000382762100722,
        "tip_mpjpe_mm": 28.277084914346535,
        "aucj": 0.6763577845855742,
        "pinch_distance_l2_mm": 45.35369820271929,
    },
    "lora": {
        "units": 20,
        "mpjpe_mm": 16.3837846989433,
        "pa_mpjpe_mm": 11.674652803497311,
        "tip_mpjpe_mm": 25.334295102705557,
        "aucj": 0.6943662499757043,
        "pinch_distance_l2_mm": 37.28338532770673,
    },
    "best_world_hand_joints_biased": {
        "units": 20,
        "mpjpe_mm": 21.520181838423014,
        "pa_mpjpe_mm": 16.55017198063433,
        "tip_mpjpe_mm": 37.147013656795025,
        "aucj": 0.6105334799302076,
        "pinch_distance_l2_mm": 48.49084075540304,
    },
}


BASELINE_REFERENCES = {
    "interhand": INTERHAND_BASELINE_REFERENCES,
    "avp": {
        "full_ft": {
            "units": 3,
            "mpjpe_mm": 19.99667742186122,
            "pa_mpjpe_mm": 15.730948108961284,
            "tip_mpjpe_mm": 36.67234898441368,
            "aucj": 0.6299805388065574,
            "pinch_distance_l2_mm": 49.88756491120454,
        },
        "last_layer": {
            "units": 3,
            "mpjpe_mm": 18.857674673199654,
            "pa_mpjpe_mm": 14.672527477003177,
            "tip_mpjpe_mm": 33.88295674489604,
            "aucj": 0.6432835832679212,
            "pinch_distance_l2_mm": 43.63226699158391,
        },
        "ridge": {
            "units": 3,
            "mpjpe_mm": 24.713979206151436,
            "pa_mpjpe_mm": 19.210771339280246,
            "tip_mpjpe_mm": 45.49791229267915,
            "aucj": 0.5729476551997741,
            "pinch_distance_l2_mm": 57.539183377523585,
        },
        # These rows use a different four-recording dataset and group order.
        # Keep the completeness gate: a three-recording validation run must
        # not certify either paper row.
        "lora": {
            "units": 4,
            "mpjpe_mm": 18.716793972998857,
            "pa_mpjpe_mm": 13.936204027363408,
            "tip_mpjpe_mm": 33.22961072747906,
            "aucj": 0.6481532503579448,
            "pinch_distance_l2_mm": 54.00893179777725,
        },
        "best_world_hand_joints_biased": {
            "units": 4,
            "mpjpe_mm": 20.48727124929428,
            "pa_mpjpe_mm": 14.629777520895004,
            "tip_mpjpe_mm": 36.219934932887554,
            "aucj": 0.6101386594008282,
            "pinch_distance_l2_mm": 51.38509364161657,
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    errors = np.linalg.norm(pred - gt, axis=-1)[valid]
    thresholds = np.linspace(0.0, 0.05, 101)
    pck = np.array([(errors <= threshold).mean() for threshold in thresholds])
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    pinch_distances = []
    for tip in (8, 12, 16, 20):
        gt_distance = np.linalg.norm(gt[:, 4] - gt[:, tip], axis=-1)
        mask = (gt_distance < 0.01) & valid[:, 4] & valid[:, tip]
        pinch_distances.append(np.linalg.norm(pred[:, 4] - pred[:, tip], axis=-1)[mask])
    distances = np.concatenate(pinch_distances)
    return {
        "aucj": float(integrate(pck, thresholds) / 0.05),
        "pinch_distance_l2_mm": (
            float(distances.mean() * 1000.0) if distances.size else None
        ),
        "pinch_frame_pairs": int(distances.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--setting", choices=REFERENCES, required=True)
    parser.add_argument(
        "--method",
        choices=["hypernet", *INTERHAND_BASELINE_REFERENCES],
        default="hypernet",
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--reference-profile", choices=["paper", "release"], default="paper"
    )
    parser.add_argument(
        "--conditioning-mode", choices=["legacy", "eval"], default="legacy"
    )
    parser.add_argument("--tolerance-mm", type=float, default=0.05)
    parser.add_argument("--tolerance-aucj", type=float, default=0.0005)
    args = parser.parse_args()
    if not 0 <= args.tolerance_mm <= 0.05:
        parser.error("Use a numerical tolerance between 0 and 0.05 mm.")
    if not 0 <= args.tolerance_aucj <= 0.0005:
        parser.error("Use an AUCJ tolerance between 0 and 0.0005.")
    expected = (
        REFERENCES[args.setting]
        if args.method == "hypernet"
        else BASELINE_REFERENCES[args.setting][args.method]
    )
    release_reference = None
    if args.reference_profile == "release":
        if args.method != "hypernet":
            parser.error("The release profile is for YOCO; use paper for baselines.")
        release_reference = json.loads(
            (ROOT / "reproduction/release_reference.json").read_text()
        )
        expected = release_reference["references"][args.setting][args.conditioning_mode]
    per_seed = []
    issues = []
    with h5py.File(args.data_path, "r") as source:
        expected_sessions = {
            f"{name}/sample_{sample}"
            for name, group in source.items()
            if decode_side(np.asarray(group["mano_side"])) == "right"
            for sample in biased_sample_indices(group, "world_hand_joints_biased", -1)
        }
        for seed in (0, 1, 2):
            directory = args.output_root / f"seed_{seed}" / args.method / "k4"
            receipt = json.loads((directory / "metrics.json").read_text())
            sessions = receipt["sessions"]
            if len(sessions) != expected["units"]:
                issues.append(f"seed {seed}: unexpected session count")
            if {row["session"] for row in sessions} != expected_sessions:
                issues.append(f"seed {seed}: source session set differs")
            reference = None
            if args.reference_root is not None:
                reference_path = (
                    args.reference_root
                    / directory.relative_to(args.output_root)
                    / "metrics.json"
                )
                reference = {
                    row["session"]: row
                    for row in json.loads(reference_path.read_text())["sessions"]
                }
                if set(reference) != {row["session"] for row in sessions}:
                    issues.append(f"seed {seed}: historical session set differs")
            rows = []
            with h5py.File(directory / "predictions.hdf5", "r") as predictions:
                for session in sessions:
                    name = session["session"]
                    saved = predictions[name]
                    calib = saved["calib_idx"][()]
                    indices = saved["eval_idx"][()]
                    if release_reference is not None and (
                        session.get("conditioning_mode") != args.conditioning_mode
                        or saved.attrs.get("conditioning_mode")
                        != args.conditioning_mode
                        or session.get("inference_seed") != seed
                        or saved.attrs.get("inference_seed") != seed
                    ):
                        issues.append(
                            f"seed {seed}, {name}: release conditioning differs"
                        )
                    if (
                        len(calib) != 4
                        or len(np.unique(calib)) != 4
                        or np.any(calib < 0)
                        or np.any(calib >= 200)
                        or np.any(indices < 200)
                        or session["calibration_num"] != 200
                        or session["side"] != "right"
                    ):
                        issues.append(f"seed {seed}, {name}: protocol mismatch")
                    if reference is not None and name in reference:
                        old = reference[name]
                        for key in (
                            "calib_idx",
                            "num_eval_samples",
                            "num_metric_joints",
                            "num_masked_bad_joints",
                        ):
                            if key == "calib_idx" and key not in old:
                                with h5py.File(
                                    reference_path.parent / "predictions.hdf5", "r"
                                ) as archived_predictions:
                                    old_calib = archived_predictions[name][key][()]
                                if not np.array_equal(calib, old_calib):
                                    issues.append(f"seed {seed}, {name}: {key} differs")
                                continue
                            if session[key] != old[key]:
                                issues.append(f"seed {seed}, {name}: {key} differs")
                    group = source[session["group"]]
                    gt = np.asarray(transform_joint3d(group["world_hand_joints"][()]))
                    expected_calib, expected_eval = choose_calib_indices(
                        len(gt),
                        4,
                        make_calibration_rng(
                            seed, session["group"], session["sample_idx"], 4
                        ),
                        "random",
                        False,
                        bad_frame_mask=read_bad_frame_mask(group, len(gt)),
                        calibration_num=200,
                    )
                    if not np.array_equal(calib, expected_calib):
                        issues.append(f"seed {seed}, {name}: calibration RNG differs")
                    if not np.array_equal(indices, expected_eval):
                        issues.append(f"seed {seed}, {name}: evaluation frames differ")
                    valid = read_valid_joint_mask(group, len(gt))
                    if valid is None:
                        valid = np.ones(gt.shape[:2], dtype=bool)
                    pred = saved["pred_joints"][()][indices]
                    computed = compute_joint_metrics(pred, gt[indices], valid[indices])
                    if session["num_eval_samples"] != len(indices):
                        issues.append(f"seed {seed}, {name}: evaluation count differs")
                    for key in ("num_metric_joints", "num_masked_bad_joints"):
                        if session[key] != computed[key]:
                            issues.append(f"seed {seed}, {name}: {key} receipt differs")
                    row = {}
                    for key in ("mpjpe_mm", "pa_mpjpe_mm", "tip_mpjpe_mm"):
                        row[key] = float(computed[key])
                        if abs(row[key] - float(session[key])) > 0.00001:
                            issues.append(f"seed {seed}, {name}: {key} receipt differs")
                    row.update(geometry_metrics(pred, gt[indices], valid[indices]))
                    rows.append(row)
            per_seed.append(
                {
                    "seed": seed,
                    "num_units": len(sessions),
                    "num_eval_frames": sum(s["num_eval_samples"] for s in sessions),
                    "mean": {
                        key: float(np.mean([row[key] for row in rows]))
                        for key in rows[0]
                        if all(row[key] is not None for row in rows)
                    },
                }
            )
            if args.setting == "avp" and args.method != "hypernet":
                # Table 2 baseline pinch scores pool valid pinch events within
                # each seed; YOCO instead averages recording-level scores.
                pinch_rows = [
                    row for row in rows if row["pinch_distance_l2_mm"] is not None
                ]
                if pinch_rows:
                    per_seed[-1]["mean"]["pinch_distance_l2_mm"] = float(
                        np.average(
                            [row["pinch_distance_l2_mm"] for row in pinch_rows],
                            weights=[row["pinch_frame_pairs"] for row in pinch_rows],
                        )
                    )
    mean = {
        key: float(np.mean([seed["mean"][key] for seed in per_seed]))
        for key in per_seed[0]["mean"]
    }
    deltas = {key: mean[key] - expected[key] for key in expected if key != "units"}
    for key, delta in deltas.items():
        tolerance = args.tolerance_aucj if key == "aucj" else args.tolerance_mm
        if not np.isfinite(delta) or abs(delta) > tolerance:
            issues.append(f"{key}: difference {delta:.6f} exceeds tolerance")
    files = [args.data_path]
    for directory in (args.model_dir, args.base_model_dir):
        files.extend(
            directory / name for name in ("model.pth", "config.yaml", "norm_stats.pkl")
        )
    assets = [
        {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    ]
    if release_reference is not None:
        for index, expected_sha in (
            (0, release_reference["dataset_sha256"][args.setting]),
            (1, release_reference["checkpoint_sha256"]),
            (2, release_reference["checkpoint_config_sha256"]),
            (3, release_reference["normalization_sha256"]),
            (4, release_reference["base_checkpoint_sha256"]),
            (5, release_reference["base_config_sha256"]),
            (6, release_reference["normalization_sha256"]),
        ):
            if assets[index]["sha256"] != expected_sha:
                issues.append(f"release asset hash differs: {files[index].name}")
    report = {
        "accuracy_passed": not issues,
        "setting": args.setting,
        "method": args.method,
        "reference_profile": args.reference_profile,
        "conditioning_mode": (
            args.conditioning_mode if release_reference is not None else None
        ),
        "scope": "Pose accuracy; hardware timings are not qualified by this check",
        "aggregation": {
            "pose_and_aucj": "recording mean within seed, then seed mean",
            "pinch": (
                "pinch-event weighted mean within seed, then seed mean"
                if args.setting == "avp" and args.method != "hypernet"
                else "recording mean within seed, then seed mean"
            ),
        },
        "tolerance_mm": args.tolerance_mm,
        "tolerance_aucj": args.tolerance_aucj,
        "mean": mean,
        "reference": expected,
        "delta": deltas,
        "per_seed": per_seed,
        "issues": issues,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "source": {
            "commit": subprocess.check_output(
                ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "-c", f"safe.directory={ROOT}", "branch", "--show-current"],
                cwd=ROOT,
                text=True,
            ).strip(),
            "readme_sha256": sha256(ROOT / "README.md"),
            "verifier_sha256": sha256(Path(__file__)),
        },
        "assets": assets,
    }
    filename = (
        "verification.json"
        if args.method == "hypernet"
        else f"verification-{args.method}.json"
    )
    destination = args.output_root / filename
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("accuracy_passed", "mean", "delta", "issues")
            },
            indent=2,
        )
    )
    print(f"Verification receipt: {destination}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
