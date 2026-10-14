"""Rebuild the four-recording cohort used by Table 2's no-calibration/LoRA rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--train-new-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Use a fresh output path; existing assets are never overwritten.")
    mapping = [
        ("val", "1", 1853),
        ("val", "2", 1471),
        ("val", "3", 1404),
        ("train_new", "0", 1387),
    ]
    with h5py.File(args.val_data, "r") as val, h5py.File(
        args.train_new_data, "r"
    ) as train_new:
        sources = {"val": val, "train_new": train_new}
        for source, name, frames in mapping:
            group = sources[source][name]
            for key in (
                "world_hand_joints",
                "world_hand_joints_true",
                "world_hand_joints_biased",
            ):
                if group[key].shape[0] != frames or group[key].shape[-2:] != (21, 3):
                    parser.error(f"Unexpected shape in {source}/{name}/{key}.")
            side = np.asarray(group["mano_side"]).reshape(-1)[0]
            if isinstance(side, bytes):
                side = side.decode()
            if side != "right":
                parser.error(f"Expected a right-hand recording in {source}/{name}.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "x") as output:
            for index, (source, name, _) in enumerate(mapping):
                sources[source].copy(name, output, name=str(index))
            output.attrs["paper_cohort"] = "table2_legacy_four_right_recordings"
    receipt = {
        "purpose": "Table 2 no-calibration and LoRA compatibility cohort",
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (args.val_data, args.train_new_data)
        ],
        "mapping": [
            {
                "output_group": str(index),
                "source": source,
                "source_group": name,
                "frames": frames,
            }
            for index, (source, name, frames) in enumerate(mapping)
        ],
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
    }
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Prepared {args.output}; provenance: {manifest}")


if __name__ == "__main__":
    main()
