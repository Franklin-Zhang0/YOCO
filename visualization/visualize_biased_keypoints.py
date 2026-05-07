"""Visualize precomputed biased vs true keypoints in Viser.

This viewer is intended for datasets that were processed by
[`scripts/precompute_bias_hdf5.py`](scripts/precompute_bias_hdf5.py:1), which writes:
- `{input_keys}_biased`
- `{input_keys}_true`

Example
-------
/data/250010203/franklin/envs/teleop/bin/python \
  visualization/visualize_biased_keypoints.py \
  --dataset-path /path/to/your_biased.hdf5 \
  --input-keys world_hand_joints \
  --port 8088

Notes
-----
- Uses the same 21-joint connectivity as [`visualization/visualize_keypoint_recon.py`](visualization/visualize_keypoint_recon.py:159).
- If you want local-hand visualization, set `--to-local` to apply
  [`calibration_net.utils.transform_joint3d()`](calibration_net/utils.py:291).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import tyro
import viser
import time

from calibration_net.utils import transform_joint3d


MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _decode_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == (1,):
        return _decode_str(x[0])
    return str(x)


def _make_bone_segments(joints: np.ndarray) -> np.ndarray:
    """(21,3) -> (E,2,3) for viser line segments."""
    segs = []
    for a, b in MANO_CONNECTIONS:
        segs.append(np.stack([joints[a], joints[b]], axis=0))
    return np.stack(segs, axis=0)


@dataclass
class Args:
    dataset_path: str
    input_keys: str = "world_hand_joints"
    port: int = 8088

    # Visualization options
    to_local: bool = True
    show_raw: bool = False

    # Optional fixed selection. If not provided, UI sliders are used.
    demo_id: Optional[str] = None


def main(args: Args) -> None:
    f = h5py.File(args.dataset_path, "r")

    demo_keys = list(f.keys())
    try:
        demo_keys = sorted(demo_keys, key=lambda k: int(k))
    except Exception:
        demo_keys = sorted(demo_keys)

    if len(demo_keys) == 0:
        raise ValueError(f"No demos found in {args.dataset_path}")

    if args.demo_id is not None:
        if args.demo_id not in f:
            raise KeyError(f"demo_id '{args.demo_id}' not found. Available: {demo_keys[:10]}...")
        demo_keys = [args.demo_id]

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", 1.0, 1.0)

    # UI
    demo_slider = server.gui.add_slider(
        "demo",
        min=0,
        max=max(0, len(demo_keys) - 1),
        step=1,
        initial_value=0,
        disabled=(len(demo_keys) == 1),
    )
    
    init_demo_len = len(f[demo_keys[0]][f"{args.input_keys}_biased"])
    if len(f[demo_keys[0]][f"{args.input_keys}_biased"].shape) == 4:
        init_demo_len = f[demo_keys[0]][f"{args.input_keys}_biased"].shape[1]
        
    safe_init_len = max(1, init_demo_len)

    frame_slider = server.gui.add_slider("frame", min=0, max=safe_init_len - 1, step=1, initial_value=0)
    
    sample_slider = server.gui.add_slider(
        "sample", min=0, max=1, step=1, initial_value=0, visible=False
    )

    # Read current demo into memory to keep interaction snappy.
    cache = {"demo_idx": None, "biased": None, "true": None, "raw": None, "num_samples": 1}

    def load_demo(demo_idx: int) -> None:
        demo_key = demo_keys[demo_idx]
        g = f[demo_key]

        biased_key = f"{args.input_keys}_biased"
        true_key = f"{args.input_keys}_true"

        if biased_key not in g or true_key not in g:
            raise KeyError(
                f"Missing precomputed bias columns in demo '{demo_key}'. "
                f"Expected '{biased_key}' and '{true_key}'. "
                "Run scripts/precompute_bias_hdf5.py first."
            )

        biased = np.asarray(g[biased_key], dtype=np.float32)
        true = np.asarray(g[true_key], dtype=np.float32)

        raw = None
        if args.show_raw:
            if args.input_keys in g:
                raw = np.asarray(g[args.input_keys], dtype=np.float32)
            else:
                raw = np.zeros_like(true)

        if args.to_local:
            biased = transform_joint3d(biased)
            true = transform_joint3d(true)
            if raw is not None:
                raw = transform_joint3d(raw)

        cache["demo_idx"] = demo_idx
        cache["biased"] = biased
        cache["true"] = true
        cache["raw"] = raw

        if biased.ndim == 4:
            cache["num_samples"] = biased.shape[0]
            T = biased.shape[1]
            sample_slider.max = max(1, cache["num_samples"] - 1)
            sample_slider.visible = True
        else:
            cache["num_samples"] = 1
            T = biased.shape[0]
            sample_slider.visible = False

        frame_slider.max = max(1, T - 1)
        frame_slider.value = 0

        # Display meta if present.
        side = _decode_str(np.asarray(g.get("mano_side", np.array(["unknown"], dtype=object))))
        server.gui.add_markdown(
            "meta",
            f"**demo**: `{demo_key}`  \\n**frames**: `{T}`  \\n**mano_side**: `{side}`  \\n**to_local**: `{args.to_local}`",
        )

    def render(demo_idx: int, frame_idx: int, sample_idx: int = 0) -> None:
        if cache["demo_idx"] != demo_idx:
            load_demo(demo_idx)

        if cache["num_samples"] > 1:
            biased = cache["biased"][sample_idx, frame_idx]
        else:
            biased = cache["biased"][frame_idx]
            
        true = cache["true"][frame_idx]
        raw = cache["raw"][frame_idx] if cache["raw"] is not None else None

        with server.atomic():
            # True
            server.scene.add_point_cloud(
                name="/true/joints",
                points=true,
                colors=np.array([1.0, 0.0, 0.0]),
                point_size=0.004,
            )
            server.scene.add_line_segments(
                name="/true/bones",
                points=_make_bone_segments(true),
                colors=np.array([1.0, 0.0, 0.0]),
                line_width=2.0,
            )

            # Biased
            server.scene.add_point_cloud(
                name="/biased/joints",
                points=biased,
                colors=np.array([0.0, 1.0, 0.0]),
                point_size=0.004,
            )
            server.scene.add_line_segments(
                name="/biased/bones",
                points=_make_bone_segments(biased),
                colors=np.array([0.0, 1.0, 0.0]),
                line_width=2.0,
            )

            # Raw (optional)
            if raw is not None:
                server.scene.add_point_cloud(
                    name="/raw/joints",
                    points=raw,
                    colors=np.array([0.0, 0.5, 1.0]),
                    point_size=0.003,
                )
                server.scene.add_line_segments(
                    name="/raw/bones",
                    points=_make_bone_segments(raw),
                    colors=np.array([0.0, 0.5, 1.0]),
                    line_width=1.0,
                )

    @demo_slider.on_update
    def _(_evt) -> None:
        render(int(demo_slider.value), int(frame_slider.value), int(sample_slider.value))

    @frame_slider.on_update
    def _(_evt) -> None:
        render(int(demo_slider.value), int(frame_slider.value), int(sample_slider.value))
        
    @sample_slider.on_update
    def _(_evt) -> None:
        render(int(demo_slider.value), int(frame_slider.value), int(sample_slider.value))

    # initial draw
    render(int(demo_slider.value), int(frame_slider.value), int(sample_slider.value))

    server.gui.add_markdown(
        "help",
        """
Use the **demo** and **frame** sliders.

Legend:
- **red**: `{input_keys}_true`
- **green**: `{input_keys}_biased`
- **blue**: `{input_keys}` (optional, `--show-raw`)
""".replace("{input_keys}", args.input_keys),
    )
    
    # frame_slider.value = 0
    while True:
        time.sleep(0.2)


if __name__ == "__main__":
    main(tyro.cli(Args))
