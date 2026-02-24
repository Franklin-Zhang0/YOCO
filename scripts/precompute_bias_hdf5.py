"""CLI utility to precompute MANO-angle-space bias into a new HDF5.

Example
-------
/data/250010203/franklin/envs/teleop/bin/python \
  scripts/precompute_bias_hdf5.py \
  --in /path/to/input.hdf5 \
  --out /path/to/output_biased.hdf5 \
  --input-keys world_hand_joints \
  --bias-cfg config/hyperlora_calib.yaml \
  --device cuda \
  --chunk-size 512 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import yaml

from calibration_net.policy.preprocess_bias import precompute_bias_hdf5


def _load_bias_cfg(path: str) -> Dict[str, Any]:
    """Load `bias_cfg` dict from either YAML config or JSON dict file."""
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Assume YAML; allow pointing at the full training config and extracting dataset.bias_cfg
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if isinstance(cfg, dict) and "dataset" in cfg and isinstance(cfg["dataset"], dict) and "bias_cfg" in cfg["dataset"]:
        return dict(cfg["dataset"]["bias_cfg"])

    # Or YAML is already the bias_cfg mapping
    if isinstance(cfg, dict):
        return dict(cfg)

    raise ValueError(f"Unsupported config format at {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input HDF5 path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output HDF5 path")
    ap.add_argument("--input-keys", dest="input_keys", required=True, help="Base joints key name (e.g. world_hand_joints)")
    ap.add_argument(
        "--bias-cfg",
        dest="bias_cfg_path",
        required=True,
        help="Path to YAML training config containing dataset.bias_cfg, or a JSON/YAML mapping with BiasConfig fields",
    )
    ap.add_argument("--device", default="cuda", help="Torch device for MANO forward (default: cuda if available)")
    ap.add_argument("--chunk-size", type=int, default=1024, help="Frames per GPU forward chunk")
    ap.add_argument("--num-samples", type=int, default=1, help="Number of biased samples to generate per sequence")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    ap.add_argument("--no-copy", action="store_true", help="Do not copy original datasets (only write new bias outputs)")

    args = ap.parse_args()

    bias_cfg = _load_bias_cfg(args.bias_cfg_path)

    precompute_bias_hdf5(
        in_path=args.in_path,
        out_path=args.out_path,
        input_keys=args.input_keys,
        bias_cfg=bias_cfg,
        device=args.device,
        chunk_size=args.chunk_size,
        copy_all_keys=not args.no_copy,
        overwrite=args.overwrite,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
