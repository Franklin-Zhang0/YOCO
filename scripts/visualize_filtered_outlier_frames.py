#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import viser

sys.path.insert(0, ".")
from calibration_net.utils import transform_joint3d


DEFAULT_HDF5 = (
    "hand_keypoint_data/Interhand_val/"
    "InterHand_retargeted_30fps_biased_super_shapedir_fix.hdf5"
)
DEFAULT_REPORT = (
    "hand_keypoint_data/Interhand_val/"
    "InterHand_retargeted_30fps_biased_super_shapedir_fix_bone50_only_filtered_report/"
    "summary.json"
)

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize frames removed by filter_bone_outliers_hdf5.py as a continuous sequence."
    )
    parser.add_argument("--hdf5-path", default=DEFAULT_HDF5, help="Original unfiltered HDF5 path")
    parser.add_argument("--summary-json", default=DEFAULT_REPORT, help="Outlier filter summary.json")
    parser.add_argument("--episode", default="", help="Episode/group to show. Default: first group in summary")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument(
        "--show-true-joints-key",
        default="world_hand_joints_true",
        help="Optional comparison joint key, if present.",
    )
    return parser.parse_args()


def decode_hdf5_string(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def make_bone_segments(joints: np.ndarray) -> np.ndarray:
    return np.stack([np.stack([joints[a], joints[b]], axis=0) for a, b in MANO_CONNECTIONS], axis=0)


def to_vis_joints(joints: np.ndarray) -> np.ndarray:
    return np.asarray(transform_joint3d(joints), dtype=np.float32)


def read_removed_indices(summary_path: Path) -> dict[str, list[int]]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    groups = summary.get("groups", {})
    return {
        str(group_name): [int(x) for x in payload.get("removed_frame_indices", [])]
        for group_name, payload in groups.items()
    }


def read_joint_sequence(group: h5py.Group, key: str, indices: np.ndarray, sample_idx: int = 0) -> np.ndarray | None:
    if key not in group:
        return None
    data = group[key]
    if data.ndim == 3 and data.shape[1:] == (21, 3):
        return to_vis_joints(np.asarray(data[indices]))
    if data.ndim == 4 and data.shape[2:] == (21, 3):
        sample_idx = min(sample_idx, data.shape[0] - 1)
        return to_vis_joints(np.asarray(data[sample_idx, indices]))
    return None


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5_path)
    summary_path = Path(args.summary_json)
    removed_by_group = read_removed_indices(summary_path)
    episodes = [g for g, indices in removed_by_group.items() if indices]
    if not episodes:
        raise RuntimeError(f"No removed frames found in {summary_path}")
    if args.episode:
        if args.episode not in removed_by_group:
            raise KeyError(f"Episode {args.episode!r} not found in {summary_path}")
        episodes = [args.episode]

    hdf5_file = h5py.File(hdf5_path, "r")
    sequence_cache: dict[tuple[str, int], dict[str, np.ndarray | str | int]] = {}

    def load_sequence(ep: str, sample_idx: int) -> dict[str, np.ndarray | str | int]:
        cache_key = (ep, sample_idx)
        if cache_key in sequence_cache:
            return sequence_cache[cache_key]

        group = hdf5_file[ep]
        removed_indices = np.array(removed_by_group[ep], dtype=np.int64)
        side = "unknown"
        if "mano_side" in group and len(group["mano_side"]) > 0:
            side = decode_hdf5_string(group["mano_side"][0])

        payload: dict[str, np.ndarray | str | int] = {
            "side": side,
            "removed_indices": removed_indices,
            "world": read_joint_sequence(group, "world_hand_joints", removed_indices),
            "true": read_joint_sequence(group, args.show_true_joints_key, removed_indices),
            "biased": read_joint_sequence(group, "world_hand_joints_biased", removed_indices, sample_idx),
        }
        sequence_cache[cache_key] = payload
        return payload

    server = viser.ViserServer(port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.35, 0.0, 0.35)
        client.camera.look_at = (0.0, 0.0, 0.0)

    server.scene.add_grid("/grid", 1.0, 1.0)
    ep_slider = server.gui.add_slider("episode", min=0, max=max(0, len(episodes) - 1), step=1, initial_value=0)
    frame_slider = server.gui.add_slider("removed-seq-frame", min=0, max=1, step=1, initial_value=0)
    sample_slider = server.gui.add_slider("biased-sample", min=0, max=1, step=1, initial_value=0, visible=False)
    play_button = server.gui.add_button("Play/Pause")
    info_markdown = server.gui.add_markdown("info", "")
    is_playing = {"value": False}

    def clear_layer(name: str) -> None:
        server.scene.add_point_cloud(
            name=f"/{name}_joints",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            point_size=args.point_size,
        )
        server.scene.add_line_segments(
            name=f"/{name}_bones",
            points=np.zeros((0, 2, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            line_width=args.line_width,
        )

    def draw_layer(name: str, joints_seq: np.ndarray | None, frame_idx: int, color: np.ndarray) -> None:
        if joints_seq is None or len(joints_seq) == 0:
            clear_layer(name)
            return
        joints = joints_seq[frame_idx]
        server.scene.add_point_cloud(
            name=f"/{name}_joints",
            points=joints,
            colors=color,
            point_size=args.point_size,
        )
        server.scene.add_line_segments(
            name=f"/{name}_bones",
            points=make_bone_segments(joints),
            colors=color,
            line_width=args.line_width,
        )

    def current_num_samples(ep: str) -> int:
        group = hdf5_file[ep]
        if "world_hand_joints_biased" in group and group["world_hand_joints_biased"].ndim == 4:
            return int(group["world_hand_joints_biased"].shape[0])
        return 1

    def render(ep_idx: int, frame_idx: int, sample_idx: int) -> None:
        ep = episodes[ep_idx]
        num_samples = current_num_samples(ep)
        sample_slider.visible = num_samples > 1
        sample_slider.max = max(1, num_samples - 1)
        sample_idx = min(sample_idx, num_samples - 1)

        seq = load_sequence(ep, sample_idx)
        removed_indices = seq["removed_indices"]
        assert isinstance(removed_indices, np.ndarray)
        T = len(removed_indices)
        if T == 0:
            return
        if frame_idx >= T:
            frame_idx = T - 1
            frame_slider.value = frame_idx
        frame_slider.max = max(1, T - 1)

        with server.atomic():
            draw_layer("world_hand_joints", seq["world"], frame_idx, np.array([1.0, 0.0, 0.0]))
            draw_layer("world_hand_joints_true", seq["true"], frame_idx, np.array([0.0, 0.8, 0.0]))
            draw_layer("world_hand_joints_biased", seq["biased"], frame_idx, np.array([0.0, 0.25, 1.0]))

        info_markdown.content = (
            f"**episode**: `{ep}`\n"
            f"**side**: `{seq['side']}`\n"
            f"**removed seq frame**: `{frame_idx}` / `{T - 1}`\n"
            f"**original frame index**: `{int(removed_indices[frame_idx])}`\n"
            f"**biased sample**: `{sample_idx}` / `{num_samples - 1}`\n"
            f"**removed frames in episode**: `{T}`"
        )

    @ep_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        frame_slider.value = 0
        sample_slider.value = 0
        render(int(ep_slider.value), 0, 0)

    @frame_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        render(int(ep_slider.value), int(frame_slider.value), int(sample_slider.value))

    @sample_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        render(int(ep_slider.value), int(frame_slider.value), int(sample_slider.value))

    @play_button.on_click
    def _(_evt) -> None:
        is_playing["value"] = not is_playing["value"]

    server.gui.add_markdown(
        "legend",
        "\n".join(
            [
                "Legend:",
                "- **red**: world_hand_joints from removed frames",
                "- **green**: world_hand_joints_true, if present",
                "- **blue**: world_hand_joints_biased for selected sample",
            ]
        ),
    )

    render(0, 0, 0)
    print(f"Viser server running at http://localhost:{args.port}")
    print(f"HDF5: {hdf5_path}")
    print(f"Summary: {summary_path}")
    print("Press Ctrl+C to stop.")

    dt = 1.0 / max(args.fps, 1e-6)
    try:
        while True:
            if is_playing["value"]:
                next_frame = int(frame_slider.value) + 1
                if next_frame > int(frame_slider.max):
                    next_frame = 0
                frame_slider.value = next_frame
                render(int(ep_slider.value), next_frame, int(sample_slider.value))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")
    finally:
        hdf5_file.close()


if __name__ == "__main__":
    main()
