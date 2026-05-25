#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

sys.path.insert(0, ".")
from calibration_net.utils import mano_recon, transform_joint3d


MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

MASK_CHOICES = {
    "direct": ("bad_frame", "gt_joint_mask"),
    "bone_endpoint": ("bad_frame_bone_endpoint", "gt_joint_mask_bone_endpoint"),
    "direct_only": ("bad_frame_direct_only", "gt_joint_mask_direct_only"),
    "bone_endpoint_only": ("bad_frame_bone_endpoint_only", "gt_joint_mask_bone_endpoint_only"),
    "filter_report": ("bad_frame_filter_report", "gt_joint_mask_filter_report"),
    "direct_only_vs_filter_report": (
        "bad_frame_direct_only_vs_filter_report",
        "gt_joint_mask_direct_only_vs_filter_report",
    ),
    "filter_report_only_vs_direct": (
        "bad_frame_filter_report_only_vs_direct",
        "gt_joint_mask_filter_report_only_vs_direct",
    ),
    "kept": ("bad_frame", "gt_joint_mask"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT joint masks saved by mark_gt_joint_outliers_hdf5.py.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--episode", default="")
    parser.add_argument("--initial-seq", choices=list(MASK_CHOICES.keys()), default="direct_only")
    parser.add_argument("--gt-key", default="world_hand_joints")
    parser.add_argument("--mano-angles-key", default="mano_angles")
    parser.add_argument("--global-orient-key", default="global_orient")
    parser.add_argument("--trans-key", default="trans")
    parser.add_argument("--betas-key", default="betas")
    parser.add_argument("--side-key", default="mano_side")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--bad-point-size", type=float, default=0.014)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--bad-line-width", type=float, default=8.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def decode_hdf5_string(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def make_bone_segments(joints: np.ndarray) -> np.ndarray:
    return np.stack([np.stack([joints[a], joints[b]], axis=0) for a, b in MANO_CONNECTIONS], axis=0)


def make_bone_points(joints: np.ndarray, mask: np.ndarray | None = None, samples_per_bone: int = 16) -> np.ndarray:
    points: list[np.ndarray] = []
    ts = np.linspace(0.0, 1.0, samples_per_bone, dtype=np.float32)
    for a, b in MANO_CONNECTIONS:
        if mask is not None and not (bool(mask[a]) or bool(mask[b])):
            continue
        segment = joints[a][None, :] * (1.0 - ts[:, None]) + joints[b][None, :] * ts[:, None]
        points.append(segment.astype(np.float32))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0)


def line_colors(color: np.ndarray, num_segments: int) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.broadcast_to(color, (num_segments, 2, 3)).copy()


def to_vis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(transform_joint3d(joints), dtype=np.float32)


def reconstruct_mano_fk(group: h5py.Group, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    mano_angles = torch.as_tensor(np.asarray(group[args.mano_angles_key], dtype=np.float32), device=device)
    global_orient = torch.as_tensor(np.asarray(group[args.global_orient_key], dtype=np.float32), device=device)
    trans = torch.as_tensor(np.asarray(group[args.trans_key], dtype=np.float32), device=device)
    betas = torch.as_tensor(np.asarray(group[args.betas_key], dtype=np.float32), device=device)
    if betas.ndim == 1:
        betas = betas.unsqueeze(0).expand(mano_angles.shape[0], -1)
    side = decode_hdf5_string(group[args.side_key][0] if len(group[args.side_key]) > 0 else group[args.side_key][()])
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


def frame_indices(group: h5py.Group, seq_kind: str) -> np.ndarray:
    frame_key, _mask_key = MASK_CHOICES[seq_kind]
    if frame_key not in group:
        return np.zeros((0,), dtype=np.int64)
    bad = np.asarray(group[frame_key], dtype=bool)
    if seq_kind == "kept":
        return np.flatnonzero(~bad).astype(np.int64)
    return np.flatnonzero(bad).astype(np.int64)


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    import viser

    h5 = h5py.File(hdf5_path, "r")
    episodes = list(h5.keys())
    if args.episode:
        if args.episode not in h5:
            raise KeyError(f"episode {args.episode!r} not found")
        episodes = [args.episode]

    cache: dict[str, dict[str, Any]] = {}

    def load_episode(ep: str) -> dict[str, Any]:
        if ep in cache:
            return cache[ep]
        group = h5[ep]
        fk = reconstruct_mano_fk(group, args, device)
        side = "unknown"
        if args.side_key in group and len(group[args.side_key]) > 0:
            side = decode_hdf5_string(group[args.side_key][0])
        payload = {
            "gt": np.asarray(group[args.gt_key], dtype=np.float32),
            "fk": fk,
            "side": side,
        }
        for _seq_kind, (_frame_key, mask_key) in MASK_CHOICES.items():
            if mask_key in group:
                payload[mask_key] = np.asarray(group[mask_key], dtype=bool)
            else:
                payload[mask_key] = np.zeros(payload["gt"].shape[:2], dtype=bool)
        cache[ep] = payload
        return payload

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", 1.0, 1.0)
    ep_slider = server.gui.add_slider("episode", min=0, max=max(0, len(episodes) - 1), step=1, initial_value=0)
    seq_dropdown = server.gui.add_dropdown("sequence", tuple(MASK_CHOICES.keys()), initial_value=args.initial_seq)
    frame_slider = server.gui.add_slider("sequence-frame", min=0, max=1, step=1, initial_value=0)
    show_gt = server.gui.add_checkbox("world_hand_joints", True)
    show_fk = server.gui.add_checkbox("mano_fk_joints", True)
    show_bad = server.gui.add_checkbox("masked_joints", True)
    play_button = server.gui.add_button("Play/Pause")
    info = server.gui.add_markdown("info", "")
    playing = {"value": False}

    def clear(name: str) -> None:
        server.scene.add_point_cloud(
            name=f"/{name}/points",
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
        server.scene.add_point_cloud(
            name=f"/{name}/bone_points",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            point_size=args.point_size * 0.7,
        )

    def draw_joints(name: str, joints: np.ndarray, color: np.ndarray, visible: bool) -> None:
        if not visible:
            clear(name)
            return
        vis = to_vis(joints)
        bones = make_bone_segments(vis)
        server.scene.add_point_cloud(
            name=f"/{name}/points",
            points=vis,
            colors=color,
            point_size=args.point_size,
        )
        server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=bones,
            colors=line_colors(color, bones.shape[0]),
            line_width=args.line_width,
        )
        bone_points = make_bone_points(vis)
        server.scene.add_point_cloud(
            name=f"/{name}/bone_points",
            points=bone_points,
            colors=np.broadcast_to(color.reshape(1, 3), bone_points.shape).copy(),
            point_size=args.point_size * 0.7,
        )

    def clear_bad_layer() -> None:
        server.scene.add_point_cloud(
            name="/masked_joints/points",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            point_size=args.bad_point_size,
        )
        server.scene.add_line_segments(
            name="/masked_joints/bones",
            points=np.zeros((0, 2, 3), dtype=np.float32),
            colors=np.zeros((0, 2, 3), dtype=np.float32),
            line_width=args.bad_line_width,
        )
        server.scene.add_point_cloud(
            name="/masked_joints/bone_points",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            point_size=args.bad_point_size * 0.45,
        )

    def draw_bad_layer(joints: np.ndarray, mask: np.ndarray) -> None:
        if not bool(show_bad.value) or not np.any(mask):
            clear_bad_layer()
            return
        vis_joints = to_vis(joints)
        points = vis_joints[mask]
        colors = np.broadcast_to(np.array([[1.0, 0.9, 0.0]], dtype=np.float32), points.shape).copy()
        bad_bones = [
            np.stack([vis_joints[a], vis_joints[b]], axis=0)
            for a, b in MANO_CONNECTIONS
            if bool(mask[a]) or bool(mask[b])
        ]
        bone_segments = (
            np.stack(bad_bones, axis=0).astype(np.float32)
            if bad_bones
            else np.zeros((0, 2, 3), dtype=np.float32)
        )
        server.scene.add_point_cloud(
            name="/masked_joints/points",
            points=points,
            colors=colors,
            point_size=args.bad_point_size,
        )
        server.scene.add_line_segments(
            name="/masked_joints/bones",
            points=bone_segments,
            colors=line_colors(np.array([1.0, 0.9, 0.0], dtype=np.float32), bone_segments.shape[0]),
            line_width=args.bad_line_width,
        )
        bone_points = make_bone_points(vis_joints, mask=mask)
        yellow = np.array([1.0, 0.9, 0.0], dtype=np.float32)
        server.scene.add_point_cloud(
            name="/masked_joints/bone_points",
            points=bone_points,
            colors=np.broadcast_to(yellow.reshape(1, 3), bone_points.shape).copy(),
            point_size=args.bad_point_size * 0.45,
        )

    def render(reset_frame: bool = False) -> None:
        ep = episodes[int(ep_slider.value)]
        seq_kind = str(seq_dropdown.value)
        group = h5[ep]
        indices = frame_indices(group, seq_kind)
        if reset_frame:
            frame_slider.value = 0
        frame_slider.max = max(1, len(indices) - 1)
        if len(indices) == 0:
            clear("gt")
            clear("fk")
            clear_bad_layer()
            info.content = f"**episode**: `{ep}`\n**sequence**: `{seq_kind}`\n**frames**: `0`"
            return

        seq_frame = min(int(frame_slider.value), len(indices) - 1)
        original_frame = int(indices[seq_frame])
        data = load_episode(ep)
        _frame_key, mask_key = MASK_CHOICES[seq_kind]
        mask = data[mask_key][original_frame]
        gt = data["gt"][original_frame]
        fk = data["fk"][original_frame]
        dist_mm = np.linalg.norm(gt - fk, axis=-1) * 1000.0

        with server.atomic():
            draw_joints("gt", gt, np.array([1.0, 0.1, 0.1], dtype=np.float32), bool(show_gt.value))
            draw_joints("fk", fk, np.array([0.0, 0.35, 1.0], dtype=np.float32), bool(show_fk.value))
            draw_bad_layer(gt, mask)

        masked = np.flatnonzero(mask)
        masked_text = ", ".join(f"{int(j)}:{dist_mm[j]:.1f}mm" for j in masked[:12])
        if len(masked) > 12:
            masked_text += ", ..."
        info.content = (
            f"**episode**: `{ep}`\n"
            f"**side**: `{data['side']}`\n"
            f"**sequence**: `{seq_kind}`\n"
            f"**sequence frame**: `{seq_frame}` / `{len(indices) - 1}`\n"
            f"**original frame index**: `{original_frame}`\n"
            f"**masked joints**: `{masked_text or 'none'}`"
        )

    @ep_slider.on_update
    def _(_evt) -> None:
        render(reset_frame=True)

    @seq_dropdown.on_update
    def _(_evt) -> None:
        render(reset_frame=True)

    @frame_slider.on_update
    def _(_evt) -> None:
        render()

    for checkbox in (show_gt, show_fk, show_bad):
        @checkbox.on_update
        def _(_evt) -> None:
            render()

    @play_button.on_click
    def _(_evt) -> None:
        playing["value"] = not playing["value"]

    server.gui.add_markdown(
        "legend",
        "\n".join(
            [
                "Legend:",
                "- **red**: world_hand_joints",
                "- **blue**: MANO FK joints",
                "- **yellow**: currently selected mask joints and adjacent bones",
            ]
        ),
    )
    render(reset_frame=True)
    print(f"Viser server running at http://localhost:{args.port}")
    print(f"HDF5: {hdf5_path}")

    try:
        dt = 1.0 / max(args.fps, 1e-6)
        while True:
            if playing["value"]:
                next_frame = int(frame_slider.value) + 1
                if next_frame > int(frame_slider.max):
                    next_frame = 0
                frame_slider.value = next_frame
                render()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")
    finally:
        h5.close()


if __name__ == "__main__":
    main()
