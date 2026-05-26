"""Offline bias preprocessing for HDF5 datasets.

Goal
----
Move MANO-angle-space bias generation out of runtime data loading.

This module precomputes, per demo/sequence:
- `{input_keys}_true`: joints from the *true* MANO hand pose
- `{input_keys}_biased`: joints from the *biased* MANO hand pose
- `mano_angles_biased`: the biased 45D MANO hand pose

Then training/inference datasets can *only load* these columns.

GPU usage
---------
MANO forward (via [`calibration_net.utils.mano_recon()`](calibration_net/utils.py:382)) runs on the
specified torch device (default: cuda if available).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import os

import h5py
import numpy as np
import torch

from calibration_net.policy.bias_generator import BiasConfig, BiasGenerator


def _decode_mano_side(side_value: Any) -> str:
    """Decode `mano_side` stored as HDF5 object/bytes into 'left'/'right'."""
    if isinstance(side_value, (list, tuple, np.ndarray)):
        # common: shape (1,) object array
        if len(side_value) == 0:
            return "right"
        side_value = side_value[0]
    if isinstance(side_value, bytes):
        side_value = side_value.decode("utf-8")
    return str(side_value).lower()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


@torch.no_grad()
def _mano_forward_in_chunks(
    bias_gen: BiasGenerator,
    *,
    hand_pose: np.ndarray,
    side: str,
    device: torch.device,
    chunk_size: int,
    betas: Optional[np.ndarray] = None,
    global_orient: Optional[np.ndarray] = None,
    transl: Optional[np.ndarray] = None,
    apply_left_shapedirs_fix: bool = False,
) -> np.ndarray:
    """Run MANO forward in time chunks to limit GPU memory.

    Args:
        hand_pose: (T,45)
        global_orient: (T,3) or None
        transl: (T,3) or None
        betas: (10,) or (T,10) or None
        apply_left_shapedirs_fix: apply MANO's left-hand shapedirs fix.

    Returns:
        joints: (T,21,3) float32
    """
    pose = np.asarray(hand_pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 45:
        raise ValueError(f"Expected hand_pose shape (T,45), got {pose.shape}")

    T = pose.shape[0]
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = T

    # Keep large arrays on CPU and only move the active chunk to GPU.
    betas_np = None if betas is None else np.asarray(betas, dtype=np.float32)
    go_np = None if global_orient is None else np.asarray(global_orient, dtype=np.float32)
    tr_np = None if transl is None else np.asarray(transl, dtype=np.float32)

    out = np.empty((T, 21, 3), dtype=np.float32)

    for s in range(0, T, chunk_size):
        e = min(T, s + chunk_size)

        hp_t = torch.tensor(pose[s:e], device=device, dtype=torch.float32)

        go_t = None
        if go_np is not None:
            go_t = torch.tensor(go_np[s:e], device=device, dtype=torch.float32)

        tr_t = None
        if tr_np is not None:
            tr_t = torch.tensor(tr_np[s:e], device=device, dtype=torch.float32)

        betas_t = None
        if betas_np is not None:
            if betas_np.ndim == 1:
                betas_t = torch.tensor(betas_np, device=device, dtype=torch.float32)
            else:
                betas_t = torch.tensor(betas_np[s:e], device=device, dtype=torch.float32)

        joints = bias_gen.mano_hand_pose_to_joints(
            hp_t,
            side=side,
            betas=betas_t,
            global_orient=go_t,
            transl=tr_t,
            apply_left_shapedirs_fix=apply_left_shapedirs_fix,
        )
        out[s:e] = joints.detach().cpu().numpy().astype(np.float32)

    return out


def precompute_bias_hdf5(
    *,
    in_path: str,
    out_path: str,
    input_keys: str,
    bias_cfg: Dict[str, Any],
    device: str | torch.device = "cuda",
    chunk_size: int = 1024,
    copy_all_keys: bool = True,
    overwrite: bool = False,
    num_samples: int = 1,
    apply_shapedir_fix: bool = False,
) -> None:
    """Precompute MANO-angle-space bias columns for an HDF5 dataset.

    The input HDF5 is expected to be organized as groups '0', '1', ... each containing
    sequence arrays like `mano_angles`, `mano_side`, etc.

    Args:
        in_path: input .hdf5
        out_path: output .hdf5 to create
        input_keys: base keypoint column name (e.g. 'world_hand_joints') used to name
            the new columns `{input_keys}_true` and `{input_keys}_biased`.
        bias_cfg: kwargs for [`BiasConfig`](calibration_net/policy/bias_generator.py:47)
        device: torch device string or object (prefer 'cuda')
        chunk_size: time chunk size for MANO forward
        copy_all_keys: if True, copy all existing datasets from input to output
        overwrite: if True, overwrite out_path if it exists
        apply_shapedir_fix: if True, apply the left-hand MANO shapedirs fix
            when reconstructing generated true and biased keypoints.
    """

    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            raise FileExistsError(f"Output exists: {out_path}. Pass overwrite=True to replace it.")

    _ensure_parent_dir(out_path)

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")

    cfg = BiasConfig(**bias_cfg)
    bias_gen = BiasGenerator(cfg)

    with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        # preserve file-level attrs
        for k, v in f_in.attrs.items():
            f_out.attrs[k] = v

        keys = list(f_in.keys())
        # numeric keys are common; preserve order but deterministic
        try:
            keys = sorted(keys, key=lambda x: int(x))
        except Exception:
            keys = sorted(keys)

        for demo_key in keys:
            g_in = f_in[demo_key]
            g_out = f_out.create_group(demo_key)

            # copy datasets
            if copy_all_keys:
                for k in g_in.keys():
                    g_in.copy(k, g_out)

            # load required inputs
            if "mano_angles" not in g_in:
                raise KeyError(f"Missing 'mano_angles' in demo {demo_key}")
            mano_angles = np.asarray(g_in["mano_angles"], dtype=np.float32)  # (T,45)

            side = _decode_mano_side(np.asarray(g_in.get("mano_side", np.array(["right"], dtype=object))))

            betas = None
            if "betas" in g_in:
                betas = np.asarray(g_in["betas"], dtype=np.float32)

            global_orient = None
            if "global_orient" in g_in:
                global_orient = np.asarray(g_in["global_orient"], dtype=np.float32)

            transl = None
            # dataset uses 'trans'
            if "trans" in g_in:
                transl = np.asarray(g_in["trans"], dtype=np.float32)
            elif "transl" in g_in:
                transl = np.asarray(g_in["transl"], dtype=np.float32)

            # apply bias in pose space
            demo_id = int(demo_key) if str(demo_key).isdigit() else hash(demo_key) % (2**31 - 1)

            # MANO forward for true (only needed once)
            joints_true = _mano_forward_in_chunks(
                bias_gen,
                hand_pose=mano_angles,
                side=side,
                betas=None, # ground truth always uses neutral betas
                global_orient=global_orient,
                transl=transl,
                device=torch_device,
                chunk_size=chunk_size,
                apply_left_shapedirs_fix=apply_shapedir_fix,
            )

            mano_angles_biased_list = []
            joints_biased_list = []

            for s_idx in range(num_samples):
                # Ensure each sample gets a different reproducible bias
                sample_demo_id = demo_id + s_idx * 1000000 
                
                mano_angles_biased = bias_gen.apply_bias_to_hand_pose(mano_angles, demo_id=sample_demo_id).astype(np.float32)
                
                joints_biased = _mano_forward_in_chunks(
                    bias_gen,
                    hand_pose=mano_angles_biased,
                    side=side,
                    betas=betas,
                    global_orient=global_orient,
                    transl=transl,
                    device=torch_device,
                    chunk_size=chunk_size,
                    apply_left_shapedirs_fix=apply_shapedir_fix,
                )
                
                mano_angles_biased_list.append(mano_angles_biased)
                joints_biased_list.append(joints_biased)

            # Stack into new first dim: (num_samples, T, ...)
            mano_angles_biased_all = np.stack(mano_angles_biased_list, axis=0)
            joints_biased_all = np.stack(joints_biased_list, axis=0)

            # write new datasets
            g_out.create_dataset(f"{input_keys}_true", data=joints_true, dtype=np.float32)
            g_out.create_dataset(f"{input_keys}_biased", data=joints_biased_all, dtype=np.float32)
            g_out.create_dataset("mano_angles_biased", data=mano_angles_biased_all, dtype=np.float32)

            # track provenance
            g_out.attrs["bias_cfg"] = str(asdict(cfg))
            g_out.attrs["bias_device"] = str(torch_device)
            g_out.attrs["bias_chunk_size"] = int(chunk_size)
            g_out.attrs["bias_num_samples"] = int(num_samples)
            g_out.attrs["apply_shapedir_fix"] = bool(apply_shapedir_fix)
