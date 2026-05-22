#!/usr/bin/env python3
"""
Visualize the paired pose HDF5 (world_hand_joints and world_hand_joints_biased).
"""

import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path
import numpy as np
import h5py
import torch
import viser
import time

from calibration_net.policy.refine_inferencer import RefineModelInferencer, resolve_torch_device
from calibration_net.utils import mano_recon, scale_hand_fingers, transform_joint3d

DEFAULT_MODEL_PATH = "combined_cnn_mlp_leaky_global_mixer_scheduler_composite_20260308_205854/best"

LAYER_COLORS = {
    "true": np.array([1.0, 0.05, 0.05]),
    "biased_raw": np.array([0.05, 0.25, 1.0]),
    "gt_mano_fk_raw": np.array([0.0, 0.95, 0.25]),
    "true_model_raw": np.array([0.0, 0.55, 0.55]),
    "biased_model_raw": np.array([0.65, 0.0, 0.85]),
    "best_biased": np.array([1.0, 0.65, 0.0]),
    "best_original": np.array([0.15, 0.9, 1.0]),
    "gt_mano_fk": np.array([0.55, 1.0, 0.05]),
    "lora": np.array([1.0, 0.15, 0.75]),
    "full_ft": np.array([1.0, 1.0, 0.0]),
    "residual_linear": np.array([0.75, 0.45, 0.15]),
    "last_layer": np.array([0.35, 0.45, 1.0]),
    "hypernet": np.array([0.95, 0.0, 0.35]),
    "hypernet_lora_ft": np.array([0.65, 1.0, 0.65]),
}

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def _make_bone_segments(joints: np.ndarray) -> np.ndarray:
    """(21,3) -> (E,2,3) for viser line segments."""
    segs = []
    for a, b in MANO_CONNECTIONS:
        segs.append(np.stack([joints[a], joints[b]], axis=0))
    return np.stack(segs, axis=0)

def _line_segment_colors(color: np.ndarray, num_segments: int) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.broadcast_to(color, (num_segments, 2, 3)).copy()

def _num_frames(joints: np.ndarray | None) -> int:
    if joints is None:
        return 0
    if joints.ndim == 4:
        return joints.shape[1]
    return joints.shape[0]

def _frame_joints(joints: np.ndarray, frame_idx: int, sample_idx: int = 0) -> np.ndarray:
    if joints.ndim == 4:
        return joints[sample_idx, frame_idx]
    return joints[frame_idx]

def _mean_bone_length(joints: np.ndarray | None) -> float | None:
    if joints is None:
        return None
    bone_lengths = [
        np.linalg.norm(joints[..., a, :] - joints[..., b, :], axis=-1)
        for a, b in MANO_CONNECTIONS
    ]
    return float(np.mean(bone_lengths))

def _scale_to_reference(joints: np.ndarray | None, reference: np.ndarray | None) -> tuple[np.ndarray | None, float | None]:
    if joints is None or reference is None:
        return None, None
    source_length = _mean_bone_length(joints)
    reference_length = _mean_bone_length(reference)
    if source_length is None or reference_length is None or source_length < 1e-12:
        return joints, None

    scale = reference_length / source_length
    wrist = joints[..., 0:1, :]
    return wrist + (joints - wrist) * scale, scale

def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.array(value)

def add_scale_reference(server: viser.ViserServer) -> None:
    """Add compact centimeter rulers for X/Y/Z axes near the origin."""
    origin = np.array([-0.08, -0.08, 0.0], dtype=np.float32)
    axes = {
        "x": {
            "direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "tick": np.array([0.0, 0.007, 0.0], dtype=np.float32),
            "color": np.array([1.0, 0.15, 0.15]),
            "label_offset": np.array([0.012, 0.012, 0.0], dtype=np.float32),
        },
        "y": {
            "direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "tick": np.array([0.007, 0.0, 0.0], dtype=np.float32),
            "color": np.array([0.15, 1.0, 0.15]),
            "label_offset": np.array([0.012, 0.012, 0.0], dtype=np.float32),
        },
        "z": {
            "direction": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "tick": np.array([0.007, 0.0, 0.0], dtype=np.float32),
            "color": np.array([0.2, 0.45, 1.0]),
            "label_offset": np.array([0.012, 0.0, 0.012], dtype=np.float32),
        },
    }

    for axis_name, spec in axes.items():
        direction = spec["direction"]
        tick = spec["tick"]
        points = [np.stack([origin, origin + direction * 0.10], axis=0)]
        for i in range(11):
            p = origin + direction * (i * 0.01)
            tick_scale = 1.8 if i in (0, 10) else 1.0
            points.append(np.stack([p, p + tick * tick_scale], axis=0))

        server.scene.add_line_segments(
            name=f"/scale_reference/{axis_name}_ruler",
            points=np.stack(points, axis=0),
            colors=spec["color"],
            line_width=2.0,
        )
        label_pos = origin + direction * 0.10 + spec["label_offset"]
        server.scene.add_label(
            f"/scale_reference/{axis_name}_label",
            f"{axis_name.upper()} 10 cm",
            position=tuple(label_pos.tolist()),
            font_size_mode="screen",
            font_screen_scale=0.7,
            anchor="center-center",
        )

    server.scene.add_label(
        "/scale_reference/tick_label",
        "1 cm ticks",
        position=tuple((origin + np.array([0.035, -0.018, 0.0], dtype=np.float32)).tolist()),
        font_size_mode="screen",
        font_screen_scale=0.7,
        anchor="top-center",
    )

def _decode_hdf5_string(value) -> str:
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize paired poses HDF5")
    parser.add_argument('hdf5_path', type=str, help='Path to paired poses HDF5 file (e.g., paired_poses.hdf5)')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='Default model path for model predictions')
    parser.add_argument('--true-model-path', type=str, default=None, help='Model path for world_hand_joints predictions')
    parser.add_argument('--biased-model-path', type=str, default=None, help='Model path for world_hand_joints_biased predictions')
    parser.add_argument('--finetune-output-dir', type=str, default='expdata_annie/finetune_baselines_mac_bone50_filtered', help='Directory containing finetune baseline outputs')
    parser.add_argument('--finetune-k', type=int, default=4, help='Calibration K for finetune predictions to visualize')
    parser.add_argument(
        '--finetune-methods',
        nargs='*',
        default=[
            'best_biased',
            'best_original',
            'gt_mano_fk',
            'lora',
            'full_ft',
            'residual_linear',
            'last_layer',
            'hypernet',
            'hypernet_lora_ft',
        ],
        help='Finetune methods to load from predictions.hdf5',
    )
    parser.add_argument('--device', type=str, default=None, help='Device to use for model inference')
    parser.add_argument('--port', type=int, default=8087, help='Viser server port')
    parser.add_argument('--fps', type=float, default=30.0, help='Playback FPS')
    parser.add_argument('--point_size', type=float, default=0.002, help='Point size for visualization')
    parser.add_argument('--line_width', type=float, default=1.0, help='Line width for visualization')
    return parser.parse_args()

def main():
    args = parse_args()
    true_model_path = args.true_model_path or args.model_path
    biased_model_path = args.biased_model_path or args.model_path
    device = resolve_torch_device(args.device)
    
    print(f"Loading HDF5 file: {args.hdf5_path}")
    print(f"True-joint model: {true_model_path}")
    print(f"Biased-joint model: {biased_model_path}")
    hdf5_file = h5py.File(args.hdf5_path, 'r')
    episodes = sorted(list(hdf5_file.keys()), key=lambda x: int(x) if x.isdigit() else x)
    episode_cache = {}
    inferencer_cache = {}
    finetune_output_dir = Path(args.finetune_output_dir)

    if not episodes:
        print("No episodes found in the HDF5 file.")
        return

    def get_inferencer(model_path: str, side: str, paired_poses: dict[str, np.ndarray]):
        cache_key = (model_path, side)
        if cache_key not in inferencer_cache:
            inferencer_cache[cache_key] = RefineModelInferencer(
                model_dir=model_path,
                side=side,
                device=device,
                paired_poses=paired_poses,
                fixed_noise=True,
            )
        else:
            inferencer_cache[cache_key].set_side(side)
            inferencer_cache[cache_key].update_condition(paired_poses=paired_poses)
        return inferencer_cache[cache_key]

    def reconstruct_model_joints(
        model_path: str,
        input_joints: np.ndarray | None,
        side: str,
        betas: np.ndarray,
        paired_poses: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        if input_joints is None:
            return None

        inferencer = get_inferencer(model_path, side, paired_poses)
        pred_theta = inferencer.inference(input_joints, batched=(input_joints.ndim == 3))
        pred_theta = torch.as_tensor(pred_theta, dtype=torch.float32, device=inferencer.device)
        betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=inferencer.device)
        if betas_tensor.ndim == 1:
            betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)

        _, recon_joint3d = mano_recon(
            pred_theta,
            side=side,
            betas=betas_tensor,
            apply_left_shapedirs_fix=False,
        )
        model = inferencer.model
        if hasattr(model, 'network') and hasattr(model.network, 'finger_scaling'):
            with torch.no_grad():
                recon_joint3d = scale_hand_fingers(recon_joint3d, model.network.finger_scaling)
        return _to_numpy(transform_joint3d(recon_joint3d))

    def reconstruct_gt_mano_joints(
        mano_angles: np.ndarray | None,
        side: str,
        betas: np.ndarray,
    ) -> np.ndarray | None:
        if mano_angles is None:
            return None

        gt_theta = torch.as_tensor(mano_angles, dtype=torch.float32, device=device)
        betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device)
        if betas_tensor.ndim == 1:
            betas_tensor = betas_tensor.unsqueeze(0).expand(gt_theta.shape[0], -1)

        _, gt_joint3d = mano_recon(
            gt_theta,
            side=side,
            betas=betas_tensor,
            apply_left_shapedirs_fix=False,
        )
        return _to_numpy(transform_joint3d(gt_joint3d))

    def load_episode(ep: str, sample_idx: int = 0):
        cache_key = (ep, sample_idx)
        if cache_key in episode_cache:
            return episode_cache[cache_key]

        grp = hdf5_file[ep]
        side = _decode_hdf5_string(grp['mano_side'][0])
        betas = np.array(grp['betas'])

        if 'world_hand_joints' in grp:
            true_raw = np.array(grp['world_hand_joints'])
            true_vis = _to_numpy(transform_joint3d(true_raw))
        else:
            true_raw = None
            true_vis = None

        if 'world_hand_joints_biased' in grp:
            jb_dataset = grp['world_hand_joints_biased']
            if jb_dataset.ndim == 4:
                biased_raw = np.array(jb_dataset[sample_idx])
            else:
                biased_raw = np.array(jb_dataset)
            biased_vis = _to_numpy(transform_joint3d(biased_raw))
        else:
            biased_raw = None
            biased_vis = None

        gt_mano_angles = np.array(grp['mano_angles']) if 'mano_angles' in grp else None
        gt_mano_fk_vis = reconstruct_gt_mano_joints(gt_mano_angles, side, betas)

        paired_poses = {}
        if biased_raw is not None:
            paired_poses['world_hand_joints_biased'] = biased_raw
        if true_raw is not None:
            paired_poses['world_hand_joints'] = true_raw

        true_model_vis = reconstruct_model_joints(
            true_model_path,
            true_raw,
            side,
            betas,
            paired_poses,
        )
        biased_model_vis = reconstruct_model_joints(
            biased_model_path,
            biased_raw,
            side,
            betas,
            paired_poses,
        )
        finetuned = {}
        for method in args.finetune_methods:
            pred_path = finetune_output_dir / method / f"k{args.finetune_k}" / "predictions.hdf5"
            if not pred_path.exists():
                continue
            with h5py.File(pred_path, "r") as pred_file:
                pred_key = f"{ep}/sample_{sample_idx}/pred_joints"
                if pred_key in pred_file:
                    finetuned[method] = np.array(pred_file[pred_key])

        episode_cache[cache_key] = {
            'side': side,
            'true': true_vis,
            'biased': biased_vis,
            'gt_mano_fk': gt_mano_fk_vis,
            'true_model': true_model_vis,
            'biased_model': biased_model_vis,
            'finetuned': finetuned,
        }
        return episode_cache[cache_key]

    server = viser.ViserServer(port=args.port)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.25, 0.0, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)

    server.scene.add_grid(
        "/grid",
        1.0,
        1.0,
        cell_size=0.01,
        section_size=0.05,
        cell_color=(90, 90, 90),
        section_color=(170, 170, 170),
        cell_thickness=0.4,
        section_thickness=1.0,
    )
    add_scale_reference(server)
    
    ep_slider = server.gui.add_slider("episode", min=0, max=max(0, len(episodes)-1), step=1, initial_value=0)
    
    # We will update frame_slider.max dynamically
    frame_slider = server.gui.add_slider("frame", min=0, max=100, step=1, initial_value=0)
    sample_slider = server.gui.add_slider("sample", min=0, max=1, step=1, initial_value=0, visible=False)
    playing = server.gui.add_button("▶ Play/Pause")
    is_playing = {"value": False}
    
    info_markdown = server.gui.add_markdown("info", "")
    metrics_markdown = server.gui.add_markdown("frame MPJPE", "")
    layer_names = [
        "true",
        "biased_raw",
        "gt_mano_fk_raw",
        "true_model_raw",
        "biased_model_raw",
        *[f"finetuned_{method}" for method in args.finetune_methods],
    ]
    layer_visible = {}
    for layer_name in layer_names:
        layer_visible[layer_name] = server.gui.add_checkbox(layer_name, True)

    def layer_is_visible(name: str) -> bool:
        return name not in layer_visible or bool(layer_visible[name].value)

    def frame_mpjpe_mm(joints: np.ndarray | None, gt_joints: np.ndarray | None, frame_idx: int) -> float | None:
        if joints is None or gt_joints is None:
            return None
        if frame_idx >= _num_frames(joints) or frame_idx >= _num_frames(gt_joints):
            return None
        pred_frame = _frame_joints(joints, frame_idx)
        gt_frame = _frame_joints(gt_joints, frame_idx)
        if pred_frame.shape != gt_frame.shape:
            return None
        return float(np.mean(np.linalg.norm(pred_frame - gt_frame, axis=-1)) * 1000.0)

    def update_metrics_panel(layer_data: dict[str, np.ndarray | None], gt_joints: np.ndarray | None, frame_idx: int) -> None:
        visible_metrics = []
        for name, joints in layer_data.items():
            if name == "true" or not layer_is_visible(name):
                continue
            mpjpe = frame_mpjpe_mm(joints, gt_joints, frame_idx)
            if mpjpe is None:
                continue
            visible_metrics.append((name, mpjpe))

        if not visible_metrics:
            metrics_markdown.content = "**Frame MPJPE vs GT**\n\nNo visible comparable layers."
            return

        lines = ["**Frame MPJPE vs GT**", ""]
        for name, mpjpe in visible_metrics:
            label_name = name.removeprefix("finetuned_")
            lines.append(f"- `{label_name}`: `{mpjpe:.2f} mm`")
        metrics_markdown.content = "\n".join(lines)

    def draw_joint_layer(name: str, joints: np.ndarray | None, frame_idx: int, color: np.ndarray) -> None:
        if not layer_is_visible(name):
            joints = None

        if joints is None:
            server.scene.add_point_cloud(
                name=f"/{name}/joints",
                points=np.zeros((0, 3)),
                colors=np.zeros((0, 3)),
                point_size=args.point_size,
            )
            server.scene.add_line_segments(
                name=f"/{name}/bones",
                points=np.zeros((0, 2, 3)),
                colors=np.zeros((0, 2, 3)),
                line_width=args.line_width,
            )
            return

        frame_joints = _frame_joints(joints, frame_idx)
        bone_segments = _make_bone_segments(frame_joints)
        server.scene.add_point_cloud(
            name=f"/{name}/joints",
            points=frame_joints,
            colors=color,
            point_size=args.point_size,
        )
        server.scene.add_line_segments(
            name=f"/{name}/bones",
            points=bone_segments,
            colors=_line_segment_colors(color, bone_segments.shape[0]),
            line_width=args.line_width,
        )
    
    def render(ep_idx, frame_idx, sample_idx=0):
        ep = episodes[ep_idx]
        grp = hdf5_file[ep]
        if 'world_hand_joints_biased' in grp and grp['world_hand_joints_biased'].ndim == 4:
            num_samples = grp['world_hand_joints_biased'].shape[0]
        else:
            num_samples = 1
        sample_slider.visible = num_samples > 1
        sample_slider.max = max(1, num_samples - 1)
        if sample_idx >= num_samples:
            sample_idx = num_samples - 1
            sample_slider.value = sample_idx

        ep_data = load_episode(ep, sample_idx)
        
        jt = ep_data['true']
        jb = ep_data['biased']
        gt_mano_fk = ep_data['gt_mano_fk']
        jt_model = ep_data['true_model']
        jb_model = ep_data['biased_model']
        finetuned = ep_data['finetuned']
        
        # Determine number of frames
        T = max(
            _num_frames(jt),
            _num_frames(jb),
            _num_frames(gt_mano_fk),
            _num_frames(jt_model),
            _num_frames(jb_model),
            *(_num_frames(joints) for joints in finetuned.values()),
        )
            
        if T == 0:
            return

        if frame_idx >= T:
            frame_idx = T - 1
            frame_slider.value = frame_idx
            
        frame_slider.max = max(1, T - 1)

        layer_data = {
            "true": jt,
            "biased_raw": jb,
            "gt_mano_fk_raw": gt_mano_fk,
            "true_model_raw": jt_model,
            "biased_model_raw": jb_model,
        }
        for method, joints in finetuned.items():
            layer_data[f"finetuned_{method}"] = joints
        
        with server.atomic():
            draw_joint_layer("true", jt, frame_idx, LAYER_COLORS["true"])
            draw_joint_layer("biased_raw", jb, frame_idx, LAYER_COLORS["biased_raw"])
            draw_joint_layer("gt_mano_fk_raw", gt_mano_fk, frame_idx, LAYER_COLORS["gt_mano_fk_raw"])
            draw_joint_layer("true_model_raw", jt_model, frame_idx, LAYER_COLORS["true_model_raw"])
            draw_joint_layer("biased_model_raw", jb_model, frame_idx, LAYER_COLORS["biased_model_raw"])
            for method, joints in finetuned.items():
                color = LAYER_COLORS.get(method, np.array([0.8, 0.8, 0.8]))
                draw_joint_layer(f"finetuned_{method}", joints, frame_idx, color)
            update_metrics_panel(layer_data, jt, frame_idx)

        info_markdown.content = (
            f"**episode**: `{ep}`\n"
            f"**frames**: `{T}`\n"
            f"**side**: `{ep_data['side']}`\n"
            f"**current frame**: `{frame_idx}`\n"
            f"**sample**: `{sample_idx}` / `{num_samples - 1}`\n"
            f"**finetune_k**: `{args.finetune_k}`\n"
            f"**loaded_finetune_methods**: `{', '.join(finetuned.keys()) if finetuned else 'none'}`"
        )

    @ep_slider.on_update
    def _(_evt):
        if getattr(_evt, "client_id", None) is None:
            return
        frame_slider.value = 0
        sample_slider.value = 0
        render(int(ep_slider.value), 0, 0)
        
    @frame_slider.on_update
    def _(_evt):
        if getattr(_evt, "client_id", None) is None:
            return
        render(int(ep_slider.value), int(frame_slider.value), int(sample_slider.value))

    @sample_slider.on_update
    def _(_evt):
        if getattr(_evt, "client_id", None) is None:
            return
        render(int(ep_slider.value), int(frame_slider.value), int(sample_slider.value))

    for checkbox in layer_visible.values():
        @checkbox.on_update
        def _(_evt):
            if getattr(_evt, "client_id", None) is None:
                return
            render(int(ep_slider.value), int(frame_slider.value), int(sample_slider.value))

    @playing.on_click
    def _(_evt):
        is_playing["value"] = not is_playing["value"]

    # legend
    server.gui.add_markdown(
        "help",
        "\n".join([
            "Legend:",
            "- **red**: world_hand_joints",
            "- **blue**: world_hand_joints_biased raw",
            "- **green**: gt MANO pose -> FK joints raw",
            "- **teal**: model(world_hand_joints) raw",
            "- **purple**: model(world_hand_joints_biased) raw",
            "- **amber**: best model on biased joints",
            "- **cyan**: best model on original joints",
            "- **lime**: gt MANO FK predictions file",
            "- **pink**: finetuned LoRA on biased joints",
            "- **yellow**: finetuned full_ft on biased joints",
            "- **brown**: finetuned residual_linear on biased joints",
            "- **periwinkle**: finetuned last_layer on biased joints",
            "- **magenta**: HyperNet/HyperLoRA conditioned on K paired poses",
            "- **pale green**: HyperNet LoRA further finetuned",
        ]),
    )

    # Initial render
    render(0, 0, 0)
    print(f"Viser server running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    
    dt = 1.0 / max(1e-6, args.fps)
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

if __name__ == '__main__':
    main()
