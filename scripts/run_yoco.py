"""Run YOCO on an NPZ input or one group of a prepared HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="NPZ with raw_pairs, target_pairs, raw_frames, optional betas.",
    )
    source.add_argument("--data-path", type=Path, help="Prepared release HDF5 dataset.")
    parser.add_argument("--group", default="1", help="HDF5 group (default: 1).")
    parser.add_argument(
        "--sample-idx", type=int, default=0, help="HDF5 bias variant index."
    )
    parser.add_argument("--calibration-window", type=int, default=200)
    parser.add_argument(
        "--frames", type=int, default=32, help="HDF5 frames after calibration."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dir", type=Path, default=Path("assets/models/yoco"))
    parser.add_argument(
        "--canonicalizer-dir", type=Path, default=Path("assets/models/frozen_estimator")
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("yoco_predictions.npz"))
    return parser.parse_args()


def _sequence(group, key: str, sample_idx: int) -> np.ndarray:
    value = np.asarray(group[key], dtype=np.float32)
    if value.ndim == 4:
        if sample_idx < 0 or sample_idx >= value.shape[0]:
            raise ValueError(
                f"sample-idx is outside {key}'s {value.shape[0]} variants."
            )
        value = value[sample_idx]
    if value.ndim != 3 or value.shape[1:] != (21, 3):
        raise ValueError(f"{key} must have shape (T, 21, 3) or (S, T, 21, 3).")
    return value


def load_inputs(args: argparse.Namespace, num_pairs: int) -> dict[str, np.ndarray]:
    if args.input is not None:
        with np.load(args.input, allow_pickle=False) as data:
            for key in ("raw_pairs", "target_pairs", "raw_frames"):
                if key not in data:
                    raise ValueError(f"NPZ input is missing {key}.")
            return {
                key: np.array(data[key], copy=True)
                for key in ("raw_pairs", "target_pairs", "raw_frames", "betas")
                if key in data or key != "betas"
            }

    import h5py

    if args.frames < 1 or args.calibration_window < num_pairs:
        raise ValueError(
            "frames must be positive and calibration-window must hold all pairs."
        )
    with h5py.File(args.data_path, "r") as data:
        group = data[args.group]
        side = np.asarray(group["mano_side"]).reshape(-1)[0]
        side = side.decode("utf-8") if isinstance(side, bytes) else str(side)
        if side != args.side:
            raise ValueError(
                f"Group {args.group} contains {side} hands, not {args.side}."
            )
        raw = _sequence(group, "world_hand_joints_biased", args.sample_idx)
        target = _sequence(group, "world_hand_joints", args.sample_idx)
        if len(raw) != len(target) or len(raw) <= args.calibration_window:
            raise ValueError(
                "Dataset needs matching raw/target lengths and held-out frames."
            )
        eligible = np.arange(args.calibration_window)
        if "bad_frame" in group:
            bad = np.asarray(group["bad_frame"], dtype=bool).reshape(-1)
            if bad.shape != (len(raw),):
                raise ValueError("bad_frame must contain one value per frame.")
            eligible = eligible[~bad[eligible]]
        if len(eligible) < num_pairs:
            raise ValueError(
                "Too few valid calibration frames in the requested window."
            )
        calib_idx = np.random.default_rng(args.seed).choice(
            eligible, num_pairs, replace=False
        )
        stop = min(len(raw), args.calibration_window + args.frames)
        result = {
            "raw_pairs": raw[calib_idx],
            "target_pairs": target[calib_idx],
            "raw_frames": raw[args.calibration_window : stop],
            "calib_idx": calib_idx,
            "frame_idx": np.arange(args.calibration_window, stop),
        }
        if "betas" in group:
            betas = np.asarray(group["betas"], dtype=np.float32)
            if betas.shape == (len(raw), 10):
                betas = betas[args.calibration_window : stop]
            result["betas"] = betas
        return result


def main() -> None:
    args = parse_args()
    if args.output.suffix != ".npz":
        raise ValueError("output must use the .npz extension.")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    from calibration_net.yoco import YOCO

    model = YOCO.from_pretrained(
        args.model_dir,
        canonicalizer_dir=args.canonicalizer_dir,
        side=args.side,
        device=args.device,
    )
    inputs = load_inputs(args, model.num_calibration_pairs)
    model.calibrate(inputs["raw_pairs"], inputs["target_pairs"])
    result = model.predict(
        inputs["raw_frames"], betas=inputs.get("betas"), batch_size=args.batch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(
            output,
            theta=result.theta,
            joints=result.joints,
            side=np.asarray(args.side),
            **{key: inputs[key] for key in ("calib_idx", "frame_idx") if key in inputs},
        )
    print(
        f"Saved theta {result.theta.shape}, joints {result.joints.shape}: {args.output}"
    )


if __name__ == "__main__":
    main()
