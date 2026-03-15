#!/usr/bin/env python3
"""
Merge pkl files from biased and refined directories into an HDF5 file
suitable for use as paired_poses input to RefineModelInferencer.
"""

import sys
sys.path.insert(0, '.')

import argparse
import pickle
import os
from pathlib import Path

import numpy as np
import h5py
import torch

from calibration_net.utils import mano_recon


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge biased and refined pkl files into HDF5 for paired_poses'
    )
    parser.add_argument(
        '--biased_dir',
        type=str,
        default='20260306_151100',
        help='Path to biased pkl directory'
    )
    parser.add_argument(
        '--refined_dir',
        type=str,
        default='20260306_151100_refined',
        help='Path to refined pkl directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='20260306_151100/paired_poses.hdf5',
        help='Output HDF5 path'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default='left_calibration_1.pkl',
        help='Comma-separated list of pkl filenames to exclude'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse exclude list
    exclude_files = set([f.strip() for f in args.exclude.split(',')])
    print(f"Excluding files: {exclude_files}")
    
    # Find pkl files in both directories
    biased_dir = Path(args.biased_dir)
    refined_dir = Path(args.refined_dir)
    
    biased_files = sorted([f for f in biased_dir.glob('*.pkl') if f.name not in exclude_files])
    refined_files = sorted([f for f in refined_dir.glob('*.pkl') if f.name not in exclude_files])
    
    print(f"Found {len(biased_files)} biased files and {len(refined_files)} refined files")
    
    # Match files by name
    biased_names = {f.name: f for f in biased_files}
    refined_names = {f.name: f for f in refined_files}
    
    matching_names = sorted(set(biased_names.keys()) & set(refined_names.keys()))
    print(f"Found {len(matching_names)} matching pkl pairs")
    
    if len(matching_names) == 0:
        print("No matching files found! Exiting.")
        return
    
    biased_means = []
    gt_means = []
    betas = None
    mano_side = None
    
    for i, name in enumerate(matching_names):
        print(f"\nProcessing [{i+1}/{len(matching_names)}]: {name}")
        
        # Load biased pkl
        biased_path = biased_names[name]
        with open(biased_path, 'rb') as f:
            biased_data = pickle.load(f)
        
        # Load refined pkl
        refined_path = refined_names[name]
        with open(refined_path, 'rb') as f:
            refined_data = pickle.load(f)
        
        # Get biased world_hand_joints and compute mean
        biased_joints = biased_data['world_hand_joints']  # (90, 21, 3)
        biased_mean = np.mean(biased_joints, axis=0)  # (21, 3)
        biased_means.append(biased_mean)
        print(f"  Biased joints shape: {biased_joints.shape}, mean shape: {biased_mean.shape}")
        
        # Reconstruct GT joints from refined mano_angles using MANO forward kinematics
        mano_angles = refined_data['mano_angles']  # (90, 45) float32
        betas_data = refined_data['betas']  # (10,) float32
        side = refined_data['mano_side'][0]  # 'left'
        
        # Prepare tensors for mano_recon
        mano_theta = torch.tensor(mano_angles, dtype=torch.float32)
        mano_orient = torch.zeros(mano_angles.shape[0], 3)
        transl = torch.zeros(mano_angles.shape[0], 3)
        betas_tensor = torch.tensor(betas_data, dtype=torch.float32)
        
        # Run MANO forward kinematics
        verts, joints = mano_recon(mano_theta, mano_orient, transl, side)#, betas_tensor)
        gt_joints = joints.numpy()  # (90, 21, 3)
        gt_mean = np.mean(gt_joints, axis=0)  # (21, 3)
        gt_means.append(gt_mean)
        
        print(f"  Refined mano_angles shape: {mano_angles.shape}")
        print(f"  GT joints shape: {gt_joints.shape}, mean shape: {gt_mean.shape}")
        
        # Store betas and side from first file
        if betas is None:
            betas = betas_data
            mano_side = side
    
    # Stack all mean poses
    biased_means = np.stack(biased_means, axis=0)  # (N, 21, 3)
    gt_means = np.stack(gt_means, axis=0)  # (N, 21, 3)
    
    print(f"\n{'='*50}")
    print(f"Stacked biased_means shape: {biased_means.shape}")
    print(f"Stacked gt_means shape: {gt_means.shape}")
    print(f"Betas shape: {betas.shape}")
    print(f"Mano side: {mano_side}")
    
    # Save as HDF5
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        demo = f.create_group('0')
        demo.create_dataset('world_hand_joints_biased', data=biased_means.astype(np.float32))
        demo.create_dataset('world_hand_joints', data=gt_means.astype(np.float32))
        demo.create_dataset('betas', data=betas.astype(np.float32))
        demo.attrs['mano_side'] = mano_side
    
    print(f"\n{'='*50}")
    print(f"Successfully saved to: {output_path}")
    print(f"  - world_hand_joints_biased: {biased_means.shape}, dtype=float32")
    print(f"  - world_hand_joints: {gt_means.shape}, dtype=float32")
    print(f"  - betas: {betas.shape}, dtype=float32")
    print(f"  - mano_side: '{mano_side}'")


if __name__ == '__main__':
    main()
