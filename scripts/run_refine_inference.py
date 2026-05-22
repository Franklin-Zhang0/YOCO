#!/usr/bin/env python3
"""
Load merged HDF5 as paired poses, create a RefineModelInferencer,
and run inference on the calibration pkl with viser visualization.
"""

import sys
sys.path.insert(0, '.')

import argparse
import pickle
from pathlib import Path

import numpy as np
import h5py
import torch
import viser
import time

from calibration_net.policy.refine_inferencer import (
    RefineModelInferencer,
    resolve_torch_device,
)
from calibration_net.utils import mano_recon, transform_joint3d, scale_hand_fingers


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RefineModelInferencer on calibration pkl using paired poses with viser visualization'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the HyperLoRA model directory'
    )
    parser.add_argument(
        '--paired_poses_path',
        type=str,
        default='20260306_151100/paired_poses.hdf5',
        help='Path to the merged HDF5 file'
    )
    parser.add_argument(
        '--input_pkl',
        type=str,
        default='20260306_151100/left_calibration_1.pkl',
        help='Path to the calibration pkl to inference on'
    )
    parser.add_argument(
        '--side',
        type=str,
        default='left',
        help='Hand side (left or right)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu). Default: cuda if available else cpu'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8087,
        help='Viser server port'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Playback FPS'
    )
    parser.add_argument(
        '--point_size',
        type=float,
        default=0.002,
        help='Point size for visualization'
    )
    parser.add_argument(
        '--line_width',
        type=float,
        default=1.0,
        help='Line width for visualization'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    device = resolve_torch_device(args.device)
    print(f"Using device: {device}")
    
    # Load paired poses from HDF5
    print(f"\nLoading paired poses from: {args.paired_poses_path}")
    with h5py.File(args.paired_poses_path, 'r') as f:
        demo = f['0']
        paired_poses = {
            'world_hand_joints_biased': np.array(demo['world_hand_joints_biased']),
            'world_hand_joints': np.array(demo['world_hand_joints']),
        }
    
    print(f"  world_hand_joints_biased shape: {paired_poses['world_hand_joints_biased'].shape}")
    print(f"  world_hand_joints shape: {paired_poses['world_hand_joints'].shape}")
    
    # Load the calibration pkl
    print(f"\nLoading calibration data from: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        calib_data = pickle.load(f)
    
    world_hand_joints = calib_data['world_hand_joints']
    print(f"  Calibration world_hand_joints shape: {world_hand_joints.shape}")
    
    # Create the inferencer
    print(f"\nCreating RefineModelInferencer with model: {args.model_dir}")
    try:
        inferencer = RefineModelInferencer(
            model_dir=args.model_dir,
            side=args.side,
            device=device,
            paired_poses=paired_poses,
            fixed_noise=True,
        )
        print("  Inferencer created successfully")
        status = inferencer.canonicalizer_status()
        if status is not None:
            print(f"  {status}")
    except Exception as e:
        print(f"  Error creating inferencer: {e}")
        raise
    
    # Run batched inference
    print(f"\nRunning batched inference on {world_hand_joints.shape[0]} frames...")
    mano_angles_pred = inferencer.inference(world_hand_joints, batched=True)
    
    print(f"\nInference complete!")
    print(f"  Output mano_angles shape: {mano_angles_pred.shape}")
    
    # Reconstruct predicted joints from mano_angles using mano_recon
    print("\nReconstructing joints from predicted mano_angles...")
    runtime_device = inferencer.device
    pred_theta = torch.as_tensor(
        mano_angles_pred,
        dtype=torch.float32,
        device=runtime_device,
    )
    betas_tensor = torch.as_tensor(
        calib_data['betas'],
        dtype=torch.float32,
        device=runtime_device,
    )
    if betas_tensor.ndim == 1:
        betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)
    side = calib_data['mano_side'][0]
    
    _, recon_joint3d = mano_recon(
        pred_theta,
        side=side,
        betas=betas_tensor,
        apply_left_shapedirs_fix=True,
    )
    
    # Apply finger scaling if model has it
    model = inferencer.model
    if hasattr(model, 'network') and hasattr(model.network, 'finger_scaling'):
        with torch.no_grad():
            recon_joint3d = scale_hand_fingers(recon_joint3d, model.network.finger_scaling)
    
    # Transform to local coordinates for visualization
    recon_joint3d = transform_joint3d(recon_joint3d).cpu().numpy()
    
    # Transform input keypoints to local coordinates
    input_keypoints = transform_joint3d(world_hand_joints)
    if torch.is_tensor(input_keypoints):
        input_keypoints = input_keypoints.cpu().numpy()
    
    # Set up viser visualization
    # print(f"\nStarting viser server on port {args.port}...")
    server = viser.ViserServer(port=args.port)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.25, 0.0, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)
    
    server.scene.add_grid("/grid", 1.0, 1.0)
    
    T = world_hand_joints.shape[0]
    
    frame_slider = server.gui.add_slider("frame", min=0, max=max(1, T-1), step=1, initial_value=0)
    playing = server.gui.add_button("▶ Play/Pause")
    is_playing = {"value": False}
    
    info_markdown = server.gui.add_markdown("info", f"**frames**: `{T}`\n**side**: `{side}`")
    
    def render_frame(frame_idx):
        with server.atomic():
            # Predicted joints (blue)
            server.scene.add_point_cloud(
                name="/joints_pred",
                points=recon_joint3d[frame_idx],
                colors=np.array([0.0, 0.0, 1.0]),
                point_size=args.point_size,
            )
            server.scene.add_line_segments(
                name="/bones_pred",
                points=_make_bone_segments(recon_joint3d[frame_idx]),
                colors=np.array([0.0, 0.0, 1.0]),
                line_width=args.line_width,
            )
            
            # Input joints (red)
            server.scene.add_point_cloud(
                name="/joints_input",
                points=input_keypoints[frame_idx],
                colors=np.array([1.0, 0.0, 0.0]),
                point_size=args.point_size,
            )
            server.scene.add_line_segments(
                name="/bones_input",
                points=_make_bone_segments(input_keypoints[frame_idx]),
                colors=np.array([1.0, 0.0, 0.0]),
                line_width=args.line_width,
            )
    
    @frame_slider.on_update
    def _(_evt):
        if getattr(_evt, "client_id", None) is None:
            return
        render_frame(int(frame_slider.value))
    
    @playing.on_click
    def _(_evt):
        is_playing["value"] = not is_playing["value"]
    
    # Initial render
    render_frame(0)
    
    # Add legend
    server.gui.add_markdown("help", """
Legend:
- **blue**: predicted joints (model output)
- **red**: input keypoints
""".strip())
    
    # print(f"Viser server running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    
    # Main loop
    dt = 1.0 / max(1e-6, args.fps)
    try:
        while True:
            if is_playing["value"]:
                next_frame = int(frame_slider.value) + 1
                if next_frame > int(frame_slider.max):
                    next_frame = 0
                frame_slider.value = next_frame
                render_frame(next_frame)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")


if __name__ == '__main__':
    main()
