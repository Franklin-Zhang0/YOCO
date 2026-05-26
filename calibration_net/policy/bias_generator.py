"""calibration_net.policy.bias_generator

Sequence-consistent bias generation for HyperLoRA calibration.

This module generates *synthetic* keypoint bias by:
  1) applying a bias in MANO pose space (45D axis-angle = MANO `hand_pose`), and
  2) converting both true and biased MANO poses back to 21x3 joints via MANO forward.

Within a single sequence/demo, the bias "style" is fixed (sampled once per demo_id).
Across demos, bias styles differ.

The default bias model aims to resemble common glove capture miscalibration:
  - stable per-finger zero-offset templates
  - small per-joint residual offsets
  - session-wide, per-finger, and per-joint gain errors
  - mild adjacent-finger crosstalk
  - proximal-to-distal within-finger coupling

Notes
-----
- We assume MANO pose joint ordering is grouped by fingers with 3 joints per finger:
  thumb(0..2), index(3..5), middle(6..8), ring(9..11), pinky(12..14).
  This matches common MANO conventions (15 joints * 3 axis-angle = 45).
- Forward is computed via [`calibration_net.utils.mano_recon()`](calibration_net/utils.py:382),
  which requires `MANO_PATH` and `manopth`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from calibration_net.utils import mano_recon


_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
_FINGER_JOINT_SLICES_15 = {
    "thumb": slice(0, 3),
    "index": slice(3, 6),
    "middle": slice(6, 9),
    "ring": slice(9, 12),
    "pinky": slice(12, 15),
}
_FINGER_JOINT_ATTENUATION = np.asarray(
    [
        [1.0, 0.85, 0.60],
        [1.0, 0.70, 0.40],
        [1.0, 0.70, 0.40],
        [1.0, 0.70, 0.40],
        [1.0, 0.70, 0.40],
    ],
    dtype=np.float32,
)
_ADJACENT_FINGER_BASE = np.asarray([0.18, 0.30, 0.36, 0.28], dtype=np.float32)
_PROXIMAL_TO_DISTAL_BASE = np.asarray(
    [
        [0.00, 0.00, 0.00],
        [0.32, 0.00, 0.00],
        [0.14, 0.24, 0.00],
    ],
    dtype=np.float32,
)
_MIN_SCALE = 0.5
_MAX_SCALE = 1.5


@dataclass
class BiasConfig:
    """Configuration for MANO-pose-space bias generation."""

    enabled: bool = True

    # Per-finger shared axis-angle offset template. Each finger offset is attenuated
    # across its joints so proximal joints are affected more than distal joints.
    per_finger_offset_prob: float = 0.9
    per_finger_offset_std: float = 0.15  # radians (axis-angle magnitude scale)

    # Per-joint residual axis-angle offset (15 joints)
    per_joint_residual_prob: float = 0.7
    per_joint_residual_std: float = 0.05  # radians

    # Optional joint mask in MANO 15-joint hand_pose indexing (0..14). None => all.
    biased_joint_indices: Optional[list[int]] = None

    # Optional clamp on each 3D axis-angle delta norm (radians). None => no clamp.
    clamp_delta_norm: Optional[float] = None

    # Scale parameter (global multiplier)
    global_scale_prob: float = 0.0
    global_scale_std: float = 0.0

    # Scale parameter (per-finger multiplier)
    per_finger_scale_prob: float = 0.0
    per_finger_scale_std: float = 0.0

    # Scale parameter (per-joint multiplier)
    per_joint_scale_prob: float = 0.0
    per_joint_scale_std: float = 0.0

    # Correlation parameters are interpreted as structured leakage terms:
    # - finger_correlation_*: same-joint-level crosstalk between adjacent fingers
    # - joint_correlation_*: proximal-to-distal coupling within a finger
    finger_correlation_prob: float = 0.0
    finger_correlation_std: float = 0.0
    joint_correlation_prob: float = 0.0
    joint_correlation_std: float = 0.0

    # Random seed for reproducibility (demo_id is added to this seed).
    seed: Optional[int] = None


class BiasGenerator:
    """Generates demo-consistent MANO hand_pose bias and converts to biased joints."""

    def __init__(self, config: BiasConfig):
        self.config = config
        self.bias_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def clear_cache(self) -> None:
        self.bias_cache.clear()

    def _rng_for_demo(self, demo_id: int) -> np.random.RandomState:
        base = 0 if self.config.seed is None else int(self.config.seed)
        return np.random.RandomState(base + int(demo_id))

    def _joint_mask_15(self) -> np.ndarray:
        if self.config.biased_joint_indices is None:
            return np.ones((15,), dtype=bool)
        mask = np.zeros((15,), dtype=bool)
        for j in self.config.biased_joint_indices:
            if 0 <= int(j) < 15:
                mask[int(j)] = True
        return mask

    def _sample_scalar_scale(
        self,
        rng: np.random.RandomState,
        *,
        prob: float,
        std: float,
    ) -> float:
        if rng.rand() >= prob:
            return 1.0
        scale = 1.0 + float(rng.randn()) * float(std)
        return float(np.clip(scale, _MIN_SCALE, _MAX_SCALE))

    def _sample_array_scale(
        self,
        rng: np.random.RandomState,
        *,
        shape: tuple[int, ...],
        prob: float,
        std: float,
        attenuation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if rng.rand() >= prob:
            return np.ones(shape, dtype=np.float32)

        noise = rng.randn(*shape).astype(np.float32) * float(std)
        if attenuation is not None:
            noise = noise * attenuation.astype(np.float32)

        scale = 1.0 + noise
        return np.clip(scale, _MIN_SCALE, _MAX_SCALE).astype(np.float32)

    def _expand_finger_offsets(
        self,
        finger_offset: np.ndarray,
    ) -> np.ndarray:
        finger_offset_per_joint = np.zeros((15, 3), dtype=np.float32)
        for f_i, finger in enumerate(_FINGER_ORDER):
            sl = _FINGER_JOINT_SLICES_15[finger]
            weights = _FINGER_JOINT_ATTENUATION[f_i][:, None]
            finger_offset_per_joint[sl] = finger_offset[f_i][None, :] * weights
        return finger_offset_per_joint

    def _sample_adjacent_finger_crosstalk(
        self,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        crosstalk = np.zeros((5, 5), dtype=np.float32)
        if rng.rand() >= self.config.finger_correlation_prob:
            return crosstalk

        raw = (
            np.abs(rng.randn(len(_ADJACENT_FINGER_BASE))).astype(np.float32)
            * float(self.config.finger_correlation_std)
        )
        edge_strengths = np.clip(raw * _ADJACENT_FINGER_BASE, 0.0, 0.35)
        for idx, strength in enumerate(edge_strengths):
            crosstalk[idx, idx + 1] = strength
            crosstalk[idx + 1, idx] = strength
        return crosstalk

    def _sample_proximal_coupling(
        self,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        coupling = np.zeros((3, 3), dtype=np.float32)
        if rng.rand() >= self.config.joint_correlation_prob:
            return coupling

        raw = np.abs(rng.randn(3)).astype(np.float32) * float(
            self.config.joint_correlation_std
        )
        raw = np.clip(raw, 0.0, 0.35)
        coupling[1, 0] = raw[0] * _PROXIMAL_TO_DISTAL_BASE[1, 0]
        coupling[2, 0] = raw[1] * _PROXIMAL_TO_DISTAL_BASE[2, 0]
        coupling[2, 1] = raw[2] * _PROXIMAL_TO_DISTAL_BASE[2, 1]
        return coupling

    def _sample_bias_params(self, demo_id: int) -> Dict[str, np.ndarray]:
        rng = self._rng_for_demo(demo_id)
        joint_mask_1d = self._joint_mask_15()
        joint_mask = joint_mask_1d.astype(np.float32).reshape(15, 1)
        joint_attenuation = _FINGER_JOINT_ATTENUATION.reshape(15, 1)

        # (5,3) per-finger offsets (thumb..pinky)
        if rng.rand() < self.config.per_finger_offset_prob:
            finger_offset = (
                rng.randn(5, 3).astype(np.float32)
                * float(self.config.per_finger_offset_std)
            )
        else:
            finger_offset = np.zeros((5, 3), dtype=np.float32)
        finger_offset_per_joint = self._expand_finger_offsets(finger_offset) * joint_mask

        # (15,3) per-joint residual
        if rng.rand() < self.config.per_joint_residual_prob:
            joint_residual = (
                rng.randn(15, 3).astype(np.float32)
                * float(self.config.per_joint_residual_std)
                * joint_attenuation
            )
        else:
            joint_residual = np.zeros((15, 3), dtype=np.float32)
        joint_residual = joint_residual * joint_mask

        # Scale parameters
        global_scale = self._sample_scalar_scale(
            rng,
            prob=self.config.global_scale_prob,
            std=self.config.global_scale_std,
        )
        finger_scale = self._sample_array_scale(
            rng,
            shape=(5,),
            prob=self.config.per_finger_scale_prob,
            std=self.config.per_finger_scale_std,
        )
        per_joint_scale = self._sample_array_scale(
            rng,
            shape=(15, 1),
            prob=self.config.per_joint_scale_prob,
            std=self.config.per_joint_scale_std,
            attenuation=joint_attenuation,
        )

        joint_scale = np.ones((15, 1), dtype=np.float32) * global_scale
        for f_i, finger in enumerate(_FINGER_ORDER):
            sl = _FINGER_JOINT_SLICES_15[finger]
            joint_scale[sl, :] *= finger_scale[f_i]
        joint_scale *= per_joint_scale
        joint_scale[~joint_mask_1d] = 1.0

        finger_crosstalk = self._sample_adjacent_finger_crosstalk(rng)
        joint_coupling = self._sample_proximal_coupling(rng)

        params = {
            "finger_offset_per_joint": finger_offset_per_joint,  # (15,3)
            "joint_residual": joint_residual,  # (15,3)
            "joint_scale": joint_scale,  # (15,1)
            "joint_mask": joint_mask,
            "finger_crosstalk": finger_crosstalk,  # (5,5)
            "joint_coupling": joint_coupling,  # (3,3)
        }
        return params

    def get_bias_params(self, demo_id: int) -> Dict[str, np.ndarray]:
        if demo_id not in self.bias_cache:
            self.bias_cache[demo_id] = self._sample_bias_params(demo_id)
        return self.bias_cache[demo_id]

    def apply_bias_to_hand_pose(self, hand_pose: np.ndarray, demo_id: int) -> np.ndarray:
        """Apply MANO hand_pose bias.

        Args:
            hand_pose: (T,45) or (45,) MANO hand_pose axis-angle.
            demo_id: demo/sequence id used to get consistent bias.

        Returns:
            biased_hand_pose with same shape as input.
        """
        if not self.config.enabled:
            return hand_pose

        pose = np.asarray(hand_pose, dtype=np.float32)
        original_shape = pose.shape
        if pose.ndim == 1:
            pose = pose[None, :]
        assert pose.shape[-1] == 45, f"Expected hand_pose last dim 45, got {pose.shape}"

        params = self.get_bias_params(demo_id)
        N = pose.shape[0]
        pose_reshaped = pose.reshape(N, 5, 3, 3)
        scaled_pose = pose_reshaped * params["joint_scale"].reshape(1, 5, 3, 1)

        # Adjacent fingers can bleed into each other at the same joint level.
        crosstalk = np.einsum(
            "fg,ngja->nfja",
            params["finger_crosstalk"],
            scaled_pose,
        )
        crosstalk *= _FINGER_JOINT_ATTENUATION.reshape(1, 5, 3, 1)
        crosstalk *= params["joint_mask"].reshape(1, 5, 3, 1)

        # Glove tendon / calibration leakage tends to flow proximal to distal only.
        coupling = np.einsum(
            "jk,nfka->nfja",
            params["joint_coupling"],
            scaled_pose,
        )
        coupling *= params["joint_mask"].reshape(1, 5, 3, 1)

        structured_offset = (
            params["finger_offset_per_joint"] + params["joint_residual"]
        ).reshape(1, 5, 3, 3)
        biased = (scaled_pose + crosstalk + coupling + structured_offset).reshape(N, 45)

        if self.config.clamp_delta_norm is not None:
            # clamp each 3D axis-angle delta norm (per joint)
            delta = (biased - pose).reshape(-1, 15, 3)
            norms = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8
            maxn = float(self.config.clamp_delta_norm)
            scale = np.minimum(1.0, maxn / norms)
            delta = delta * scale
            biased = (pose.reshape(-1, 15, 3) + delta).reshape(-1, 45)

        if len(original_shape) == 1:
            return biased[0]
        return biased

    @torch.no_grad()
    def mano_hand_pose_to_joints(
        self,
        hand_pose: torch.Tensor,
        *,
        side: str,
        betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        apply_left_shapedirs_fix: bool = False,
    ) -> torch.Tensor:
        """Convert MANO `hand_pose` (45D) to 21x3 joints using MANO forward.

        Args:
            hand_pose: (T,45) or (45,) tensor.
            side: 'left' or 'right'.
            betas: (10,) or (T,10) or None.
            global_orient: (3,) or (T,3) or None. If None, uses zeros.
            transl: (3,) or (T,3) or None. If None, uses zeros.
            apply_left_shapedirs_fix: apply MANO's left-hand shapedirs fix.

        Returns:
            joints: (T,21,3)
        """
        if hand_pose.ndim == 1:
            hand_pose = hand_pose.unsqueeze(0)
        assert hand_pose.shape[-1] == 45

        T = hand_pose.shape[0]
        device = hand_pose.device
        dtype = hand_pose.dtype

        if global_orient is None:
            global_orient = torch.zeros((T, 3), device=device, dtype=dtype)
        elif global_orient.ndim == 1:
            global_orient = global_orient.unsqueeze(0).expand(T, -1)

        if transl is None:
            transl = torch.zeros((T, 3), device=device, dtype=dtype)
        elif transl.ndim == 1:
            transl = transl.unsqueeze(0).expand(T, -1)

        if betas is None:
            betas = torch.zeros((T, 10), device=device, dtype=dtype)
        else:
            if betas.ndim == 1:
                betas = betas.unsqueeze(0).expand(T, -1)
            elif betas.shape[0] != T:
                # allow (T,10) only
                betas = betas.expand(T, -1)

        _, joints = mano_recon(
            mano_theta_pred=hand_pose,
            mano_orient=global_orient,
            transl=transl,
            side=side,
            betas=betas,
            apply_left_shapedirs_fix=apply_left_shapedirs_fix,
        )
        return joints  # (T,21,3)

    @torch.no_grad()
    def generate_true_and_biased_joints_from_hand_pose(
        self,
        hand_pose_true: np.ndarray,
        *,
        demo_id: int,
        side: str,
        betas: Optional[np.ndarray] = None,
        global_orient: Optional[np.ndarray] = None,
        transl: Optional[np.ndarray] = None,
        device: str | torch.device = "cpu",
        apply_left_shapedirs_fix: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: (hand_pose_true -> joints_true) and (hand_pose_biased -> joints_biased)."""

        hand_pose_true_np = np.asarray(hand_pose_true, dtype=np.float32)
        hand_pose_biased_np = self.apply_bias_to_hand_pose(hand_pose_true_np, demo_id=demo_id)

        hp_true = torch.tensor(hand_pose_true_np, device=device, dtype=torch.float32)
        hp_biased = torch.tensor(hand_pose_biased_np, device=device, dtype=torch.float32)

        betas_t = None if betas is None else torch.tensor(np.asarray(betas, dtype=np.float32), device=device)
        go_t = None if global_orient is None else torch.tensor(np.asarray(global_orient, dtype=np.float32), device=device)
        tr_t = None if transl is None else torch.tensor(np.asarray(transl, dtype=np.float32), device=device)

        joints_true = self.mano_hand_pose_to_joints(
            hp_true,
            side=side,
            betas=betas_t,
            global_orient=go_t,
            transl=tr_t,
            apply_left_shapedirs_fix=apply_left_shapedirs_fix,
        )
        joints_biased = self.mano_hand_pose_to_joints(
            hp_biased,
            side=side,
            betas=betas_t,
            global_orient=go_t,
            transl=tr_t,
            apply_left_shapedirs_fix=apply_left_shapedirs_fix,
        )

        return (
            joints_true.detach().cpu().numpy(),
            joints_biased.detach().cpu().numpy(),
            hand_pose_biased_np,
        )


def create_paired_hypernet_input(
    biased_keypoints: np.ndarray,
    true_keypoints: np.ndarray,
    num_pairs: int = 4,
    flatten: bool = True,
) -> np.ndarray:
    """Create a simple paired conditioning input from biased/true keypoints.

    This helper is kept for backward compatibility with the previous pipeline.
    The new preference-extractor-based encoder will generally consume structured
    pair tokens (K,2,21,3) rather than a flattened vector.
    """

    assert biased_keypoints.shape == true_keypoints.shape
    assert biased_keypoints.shape[0] >= num_pairs

    biased = biased_keypoints[:num_pairs]
    true = true_keypoints[:num_pairs]

    paired = np.stack([biased, true], axis=1)  # (K,2,21,3)
    return paired.reshape(-1) if flatten else paired
