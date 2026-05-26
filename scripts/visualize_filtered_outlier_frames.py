#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import viser

sys.path.insert(0, ".")
from calibration_net.utils import transform_joint3d


DEFAULT_HDF5 = (
    "hand_keypoint_data/Interhand_val/"
    "InterHand_retargeted_30fps_biased_super.hdf5"
)
DEFAULT_REPORT = (
    "hand_keypoint_data/Interhand_val/"
    "InterHand_retargeted_30fps_biased_super_bone50_only_filtered_report/"
    "summary.json"
)

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

LAYER_COLORS = {
    "world_hand_joints": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "world_hand_joints_true": np.array([0.0, 0.8, 0.0], dtype=np.float32),
    "world_hand_joints_biased": np.array([0.0, 0.25, 1.0], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize removed/kept frame sequences from filter_bone_outliers_hdf5.py."
    )
    parser.add_argument("--hdf5-path", default=DEFAULT_HDF5, help="Original unfiltered HDF5 path")
    parser.add_argument("--summary-json", default=DEFAULT_REPORT, help="Outlier filter summary.json")
    parser.add_argument("--episode", default="", help="Episode/group to show. Default: first group in summary")
    parser.add_argument("--initial-seq", choices=["removed", "kept"], default="removed")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--show-true-joints-key", default="world_hand_joints_true")
    return parser.parse_args()


def decode_hdf5_string(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def make_bone_segments(joints: np.ndarray) -> np.ndarray:
    return np.stack([np.stack([joints[a], joints[b]], axis=0) for a, b in MANO_CONNECTIONS], axis=0)


def line_segment_colors(color: np.ndarray, num_segments: int) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.broadcast_to(color, (num_segments, 2, 3)).copy()


def to_vis_joints(joints: np.ndarray) -> np.ndarray:
    return np.asarray(transform_joint3d(joints), dtype=np.float32)


def read_summary(summary_path: Path) -> tuple[dict[str, set[int]], dict[str, int]]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    removed_by_group: dict[str, set[int]] = {}
    original_count_by_group: dict[str, int] = {}
    for group_name, payload in summary.get("groups", {}).items():
        removed_by_group[str(group_name)] = set(int(x) for x in payload.get("removed_frame_indices", []))
        original_count_by_group[str(group_name)] = int(payload.get("original_frames", 0))
    return removed_by_group, original_count_by_group


def frame_indices_for(seq_kind: str, removed: set[int], original_frames: int) -> np.ndarray:
    if seq_kind == "removed":
        return np.array(sorted(removed), dtype=np.int64)
    if seq_kind == "kept":
        return np.array([idx for idx in range(original_frames) if idx not in removed], dtype=np.int64)
    raise ValueError(f"Unknown sequence: {seq_kind}")


def read_joint_sequence(group: h5py.Group, key: str, indices: np.ndarray, sample_idx: int = 0) -> np.ndarray | None:
    if key not in group or len(indices) == 0:
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
    removed_by_group, original_count_by_group = read_summary(summary_path)
    episodes = [g for g in removed_by_group.keys() if original_count_by_group.get(g, 0) > 0]
    if args.episode:
        if args.episode not in removed_by_group:
            raise KeyError(f"Episode {args.episode!r} not found in {summary_path}")
        episodes = [args.episode]
    if not episodes:
        raise RuntimeError(f"No episodes found in {summary_path}")

    hdf5_file = h5py.File(hdf5_path, "r")
    sequence_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def current_num_samples(ep: str) -> int:
        group = hdf5_file[ep]
        if "world_hand_joints_biased" in group and group["world_hand_joints_biased"].ndim == 4:
            return int(group["world_hand_joints_biased"].shape[0])
        return 1

    def load_sequence(ep: str, seq_kind: str, sample_idx: int) -> dict[str, Any]:
        cache_key = (ep, seq_kind, sample_idx)
        if cache_key in sequence_cache:
            return sequence_cache[cache_key]

        group = hdf5_file[ep]
        original_frames = original_count_by_group.get(ep, int(group["world_hand_joints"].shape[0]))
        removed = removed_by_group.get(ep, set())
        indices = frame_indices_for(seq_kind, removed, original_frames)
        side = "unknown"
        if "mano_side" in group and len(group["mano_side"]) > 0:
            side = decode_hdf5_string(group["mano_side"][0])

        payload = {
            "side": side,
            "sequence": seq_kind,
            "frame_indices": indices,
            "removed_count": len(removed),
            "kept_count": original_frames - len(removed),
            "original_frames": original_frames,
            "world_hand_joints": read_joint_sequence(group, "world_hand_joints", indices),
            "world_hand_joints_true": read_joint_sequence(group, args.show_true_joints_key, indices),
            "world_hand_joints_biased": read_joint_sequence(group, "world_hand_joints_biased", indices, sample_idx),
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
    sequence_dropdown = server.gui.add_dropdown("sequence", ("removed", "kept"), initial_value=args.initial_seq)
    frame_slider = server.gui.add_slider("sequence-frame", min=0, max=1, step=1, initial_value=0)
    sample_slider = server.gui.add_slider("biased-sample", min=0, max=1, step=1, initial_value=0, visible=False)
    layer_visible = {
        "world_hand_joints": server.gui.add_checkbox("world_hand_joints", True),
        "world_hand_joints_true": server.gui.add_checkbox(args.show_true_joints_key, True),
        "world_hand_joints_biased": server.gui.add_checkbox("world_hand_joints_biased", True),
    }
    play_button = server.gui.add_button("Play/Pause")
    info_markdown = server.gui.add_markdown("info", "")
    is_playing = {"value": False}

    def clear_layer(name: str) -> None:
        server.scene.add_point_cloud(
            name=f"/{name}/joints",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            point_size=args.point_size,
        )
        server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=np.zeros((0, 2, 3), dtype=np.float32),
            colors=np.zeros((0, 2, 3), dtype=np.float32),
            line_width=args.line_width,
        )

    def draw_layer(name: str, joints_seq: np.ndarray | None, frame_idx: int) -> None:
        if not bool(layer_visible[name].value) or joints_seq is None or len(joints_seq) == 0:
            clear_layer(name)
            return
        joints = joints_seq[frame_idx]
        color = LAYER_COLORS[name]
        bone_segments = make_bone_segments(joints)
        server.scene.add_point_cloud(
            name=f"/{name}/joints",
            points=joints,
            colors=color,
            point_size=args.point_size,
        )
        server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=bone_segments,
            colors=line_segment_colors(color, bone_segments.shape[0]),
            line_width=args.line_width,
        )

    def render(ep_idx: int, seq_kind: str, frame_idx: int, sample_idx: int) -> None:
        ep = episodes[ep_idx]
        num_samples = current_num_samples(ep)
        sample_slider.visible = num_samples > 1
        sample_slider.max = max(1, num_samples - 1)
        sample_idx = min(sample_idx, num_samples - 1)

        seq = load_sequence(ep, seq_kind, sample_idx)
        frame_indices = seq["frame_indices"]
        T = len(frame_indices)
        if T == 0:
            for layer_name in layer_visible:
                clear_layer(layer_name)
            frame_slider.max = 1
            frame_slider.value = 0
            info_markdown.content = (
                f"**episode**: `{ep}`\n"
                f"**sequence**: `{seq_kind}`\n"
                "**frames**: `0`"
            )
            return
        if frame_idx >= T:
            frame_idx = T - 1
            frame_slider.value = frame_idx
        frame_slider.max = max(1, T - 1)

        with server.atomic():
            for layer_name in layer_visible:
                draw_layer(layer_name, seq[layer_name], frame_idx)

        info_markdown.content = (
            f"**episode**: `{ep}`\n"
            f"**side**: `{seq['side']}`\n"
            f"**sequence**: `{seq_kind}`\n"
            f"**sequence frame**: `{frame_idx}` / `{T - 1}`\n"
            f"**original frame index**: `{int(frame_indices[frame_idx])}`\n"
            f"**biased sample**: `{sample_idx}` / `{num_samples - 1}`\n"
            f"**removed / kept / original frames**: "
            f"`{seq['removed_count']}` / `{seq['kept_count']}` / `{seq['original_frames']}`"
        )

    def rerender(reset_frame: bool = False) -> None:
        if reset_frame:
            frame_slider.value = 0
        render(
            int(ep_slider.value),
            str(sequence_dropdown.value),
            int(frame_slider.value),
            int(sample_slider.value),
        )

    @ep_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        sample_slider.value = 0
        rerender(reset_frame=True)

    @sequence_dropdown.on_update
    def _(_evt) -> None:
        rerender(reset_frame=True)

    @frame_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        rerender()

    @sample_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        rerender()

    for checkbox in layer_visible.values():
        @checkbox.on_update
        def _(_evt) -> None:
            rerender()

    @play_button.on_click
    def _(_evt) -> None:
        is_playing["value"] = not is_playing["value"]

    server.gui.add_markdown(
        "legend",
        "\n".join(
            [
                "Legend:",
                "- **red**: world_hand_joints",
                f"- **green**: {args.show_true_joints_key}, if present",
                "- **blue**: world_hand_joints_biased",
            ]
        ),
    )

    rerender(reset_frame=True)
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
                rerender()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")
    finally:
        hdf5_file.close()


if __name__ == "__main__":
    main()
