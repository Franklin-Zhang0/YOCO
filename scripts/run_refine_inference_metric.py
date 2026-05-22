#!/usr/bin/env python3
"""
Load merged HDF5 as paired poses, create a RefineModelInferencer,
run inference on the calibration pkl, and compute evaluation metrics
(no visualization).
"""

import sys
sys.path.insert(0, '.')

import argparse
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import h5py
import torch
import torch.nn.functional as F

from calibration_net.policy.refine_inferencer import RefineModelInferencer
from calibration_net.utils import mano_recon, transform_joint3d, scale_hand_fingers
from calibration_net.policy.loss import (
    compute_relative_distances,
    relative_distance_loss,
    TIP_INDICIES,
)

# Human-readable names for the 10 fingertip pairs (C(5,2))
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
TIP_PAIR_NAMES = [
    f'{a}-{b}' for a, b in combinations(FINGER_NAMES, 2)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RefineModelInferencer on calibration pkl and compute metrics (no visualization)'
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # ── Load paired poses from HDF5 ──────────────────────────────────────
    print(f"\nLoading paired poses from: {args.paired_poses_path}")
    with h5py.File(args.paired_poses_path, 'r') as f:
        demo = f['0']
        paired_poses = {
            'world_hand_joints_biased': np.array(demo['world_hand_joints_biased']),
            'world_hand_joints': np.array(demo['world_hand_joints']),
        }

    print(f"  world_hand_joints_biased shape: {paired_poses['world_hand_joints_biased'].shape}")
    print(f"  world_hand_joints shape: {paired_poses['world_hand_joints'].shape}")

    # ── Load the calibration pkl ─────────────────────────────────────────
    print(f"\nLoading calibration data from: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        calib_data = pickle.load(f)

    world_hand_joints = calib_data['world_hand_joints']
    print(f"  Calibration world_hand_joints shape: {world_hand_joints.shape}")

    # ── Create the inferencer ────────────────────────────────────────────
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

    # ── Run batched inference ────────────────────────────────────────────
    print(f"\nRunning batched inference on {world_hand_joints.shape[0]} frames...")
    mano_angles_pred = inferencer.inference(world_hand_joints, batched=True)

    print(f"\nInference complete!")
    print(f"  Output mano_angles shape: {mano_angles_pred.shape}")

    # ── Reconstruct predicted joints from mano_angles ────────────────────
    print("\nReconstructing joints from predicted mano_angles...")
    pred_theta = torch.tensor(mano_angles_pred, dtype=torch.float32)
    betas_tensor = torch.tensor(calib_data['betas'], dtype=torch.float32).unsqueeze(0).expand(pred_theta.shape[0], -1)
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

    # Transform to local coordinates
    recon_joint3d = transform_joint3d(recon_joint3d)
    if torch.is_tensor(recon_joint3d):
        recon_joint3d = recon_joint3d.cpu()

    # Transform input keypoints to local coordinates
    input_keypoints = transform_joint3d(world_hand_joints)
    if torch.is_tensor(input_keypoints):
        input_keypoints = input_keypoints.cpu()

    # ── Convert to float tensors for metric computation ──────────────────
    pred_tensor = recon_joint3d.float() if torch.is_tensor(recon_joint3d) else torch.tensor(recon_joint3d, dtype=torch.float32)
    gt_tensor = input_keypoints.float() if torch.is_tensor(input_keypoints) else torch.tensor(input_keypoints, dtype=torch.float32)

    # ── Compute metrics ──────────────────────────────────────────────────
    with torch.no_grad():
        # 1. Global MSE (all 21 joints)
        global_mse = F.mse_loss(pred_tensor, gt_tensor).item()

        # 2. Tip MSE (5 fingertip joints only)
        tip_mse = F.mse_loss(pred_tensor[:, TIP_INDICIES], gt_tensor[:, TIP_INDICIES]).item()

        # 3. Relative distance loss (weighted, from loss.py)
        rel_dist_loss = relative_distance_loss(pred_tensor, gt_tensor).item()

        # 4. Mean absolute tip distance error (unweighted)
        pred_dists = compute_relative_distances(pred_tensor)   # (B, 10)
        gt_dists = compute_relative_distances(gt_tensor)       # (B, 10)
        abs_dist_error = (pred_dists - gt_dists).abs()         # (B, 10)
        mean_abs_dist_error = abs_dist_error.mean().item()

        # 5. Per-pair mean absolute distance error
        per_pair_mae = abs_dist_error.mean(dim=0)              # (10,)

        # 6. Per-fingertip MSE (individual tips)
        per_tip_mse = {}
        for i, name in enumerate(FINGER_NAMES):
            tip_idx = TIP_INDICIES[i]
            per_tip_mse[name] = F.mse_loss(pred_tensor[:, tip_idx], gt_tensor[:, tip_idx]).item()

    # ── Print summary ────────────────────────────────────────────────────
    n_frames = pred_tensor.shape[0]
    sep = '=' * 60

    print(f"\n{sep}")
    print(f"  METRIC SUMMARY  ({n_frames} frames, side={side})")
    print(sep)

    print(f"\n{'Metric':<40s} {'Value':>12s}")
    print('-' * 54)
    print(f"{'Global MSE (21 joints)':<40s} {global_mse:>12.6f}")
    print(f"{'Tip MSE (5 fingertips)':<40s} {tip_mse:>12.6f}")
    print(f"{'Relative distance loss (weighted)':<40s} {rel_dist_loss:>12.6f}")
    print(f"{'Mean |tip dist error| (unweighted)':<40s} {mean_abs_dist_error:>12.6f}")

    print(f"\n{'Per-Fingertip MSE':<40s} {'Value':>12s}")
    print('-' * 54)
    for name in FINGER_NAMES:
        print(f"  {name + ' tip':<38s} {per_tip_mse[name]:>12.6f}")

    print(f"\n{'Per-Pair Mean |Tip Distance Error|':<40s} {'Value (m)':>12s}")
    print('-' * 54)
    for j, pair_name in enumerate(TIP_PAIR_NAMES):
        print(f"  {pair_name:<38s} {per_pair_mae[j].item():>12.6f}")

    print(f"\n{sep}")
    print("Done.")


if __name__ == '__main__':
    main()
