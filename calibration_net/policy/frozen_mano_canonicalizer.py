from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from calibration_net.policy.model import load_model
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.utils import Side, mano_recon, scale_hand_fingers


def _cfg_get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def decode_mano_side(side_value: Any) -> str:
    if isinstance(side_value, (list, tuple, np.ndarray)):
        if len(side_value) == 0:
            raise ValueError('mano_side is empty')
        side_value = side_value[0]
    if isinstance(side_value, bytes):
        side_value = side_value.decode('utf-8')
    side = str(side_value).lower()
    if side not in {'left', 'right'}:
        raise ValueError(f"Expected mano_side to be 'left' or 'right', got {side_value!r}")
    return side


class FrozenManoCanonicalizer:
    def __init__(
        self,
        extractor_path: str,
        device: str | None = None,
        chunk_size: int = 16384,
        max_batch_frames: int | None = None,
    ):
        if not extractor_path:
            raise ValueError(
                'Frozen MANO canonicalizer is enabled, but extractor_path is empty.'
            )

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.extractor_path = extractor_path
        self.device = device
        self.chunk_size = max(int(chunk_size), 1)
        if max_batch_frames is None:
            max_batch_frames = max(self.chunk_size * 16, 65536)
        self.max_batch_frames = max(int(max_batch_frames), self.chunk_size)

        self.model, self.norm_stats, self.cfg = load_model(extractor_path, device=device)
        self.model.eval()

        output_size = self.cfg['model'].get('output_size')
        if output_size != 45:
            raise ValueError(
                'Frozen MANO canonicalizer requires a MANO-theta extractor with '
                f'output_size=45, got {output_size!r} from {extractor_path}'
            )

        process_x = self.cfg['dataset'].get('preprocessing_x', [])
        process_y = [
            process if process != 'normalize' else 'denormalize'
            for process in self.cfg['dataset'].get('preprocessing_y', [])
        ]
        self.process_x_fn = build_preprocessing(process_x, 'x', norm_stats=self.norm_stats)
        self.process_y_fn = build_preprocessing(process_y, 'y', norm_stats=self.norm_stats)

    @property
    def status_line(self) -> str:
        return (
            'Frozen MANO canonicalizer active: '
            f'{self.extractor_path} (chunk_size={self.chunk_size}, max_batch_frames={self.max_batch_frames})'
        )

    @torch.no_grad()
    def _canonicalize_flat_np(self, flat_joints: np.ndarray, side: str) -> np.ndarray:
        side = decode_mano_side(side)
        flat_joints = np.asarray(flat_joints)
        if flat_joints.ndim != 3 or flat_joints.shape[-2:] != (21, 3):
            raise ValueError(
                'Frozen MANO canonicalizer expects flat input shape (N, 21, 3), '
                f'got {flat_joints.shape}'
            )

        flat_joints = flat_joints.astype(np.float32, copy=False)
        if flat_joints.shape[0] == 0:
            return flat_joints.copy()

        canonical_flat = []
        enc_side = Side.__dict__[side.upper()].value
        for start in range(0, flat_joints.shape[0], self.chunk_size):
            stop = min(flat_joints.shape[0], start + self.chunk_size)
            x_chunk = self.process_x_fn(flat_joints[start:stop])
            x_chunk = torch.as_tensor(x_chunk, device=self.device, dtype=torch.float32)
            side_chunk = torch.full(
                (x_chunk.shape[0],),
                enc_side,
                device=self.device,
                dtype=torch.float32,
            )

            pred_theta = self.model(x_chunk, side=side_chunk)
            if isinstance(pred_theta, dict):
                pred_theta = pred_theta['pred_y']
            pred_theta = pred_theta.detach().cpu().numpy()
            pred_theta = self.process_y_fn(pred_theta)
            pred_theta = torch.as_tensor(pred_theta, device=self.device, dtype=torch.float32)

            zero_betas = torch.zeros((pred_theta.shape[0], 10), device=self.device, dtype=pred_theta.dtype)
            _, recon_joint3d = mano_recon(
                pred_theta,
                side=side,
                betas=zero_betas,
                apply_left_shapedirs_fix=True,
            )

            if hasattr(self.model, 'network') and hasattr(self.model.network, 'finger_scaling'):
                finger_scaling = self.model.network.finger_scaling.detach().to(self.device)
                if finger_scaling.dim() == 1:
                    finger_scaling = finger_scaling.unsqueeze(0).expand(pred_theta.shape[0], -1)
                recon_joint3d = scale_hand_fingers(recon_joint3d, finger_scaling)

            canonical_flat.append(recon_joint3d.detach().cpu().numpy())

        return np.concatenate(canonical_flat, axis=0)

    @torch.no_grad()
    def canonicalize_many(self, joints_list: list[np.ndarray], side: str) -> list[np.ndarray]:
        if len(joints_list) == 0:
            return []

        flat_batches = []
        shapes = []
        dtypes = []
        for joints in joints_list:
            joints_np = np.asarray(joints)
            if joints_np.shape[-2:] != (21, 3):
                raise ValueError(
                    'Frozen MANO canonicalizer expects input shape (..., 21, 3), '
                    f'got {joints_np.shape}'
                )
            shapes.append(joints_np.shape)
            dtypes.append(joints_np.dtype)
            flat_batches.append(joints_np.reshape(-1, 21, 3).astype(np.float32, copy=False))

        canonical_flat = self._canonicalize_flat_np(np.concatenate(flat_batches, axis=0), side)

        outputs = []
        start = 0
        for shape, dtype in zip(shapes, dtypes):
            count = int(np.prod(shape[:-2]))
            stop = start + count
            restored = canonical_flat[start:stop].reshape(*shape[:-2], 21, 3)
            if np.issubdtype(dtype, np.floating):
                restored = restored.astype(dtype, copy=False)
            outputs.append(restored)
            start = stop
        return outputs

    @torch.no_grad()
    def canonicalize(self, joints: np.ndarray | torch.Tensor, side: str) -> np.ndarray | torch.Tensor:
        side = decode_mano_side(side)
        input_is_torch = isinstance(joints, torch.Tensor)
        input_device = joints.device if input_is_torch else None
        input_dtype = joints.dtype if input_is_torch else None

        joints_np = joints.detach().cpu().numpy() if input_is_torch else np.asarray(joints)
        if joints_np.shape[-2:] != (21, 3):
            raise ValueError(
                'Frozen MANO canonicalizer expects input shape (..., 21, 3), '
                f'got {joints_np.shape}'
            )

        leading_shape = joints_np.shape[:-2]
        flat_joints = joints_np.reshape(-1, 21, 3).astype(np.float32, copy=False)
        if flat_joints.shape[0] == 0:
            return joints.clone() if input_is_torch else joints_np.copy()

        canonical_np = self._canonicalize_flat_np(flat_joints, side).reshape(*leading_shape, 21, 3)

        if input_is_torch:
            target_dtype = input_dtype if input_dtype.is_floating_point else torch.float32
            return torch.as_tensor(canonical_np, device=input_device, dtype=target_dtype)

        if np.issubdtype(joints_np.dtype, np.floating):
            return canonical_np.astype(joints_np.dtype, copy=False)
        return canonical_np.astype(np.float32, copy=False)


def build_frozen_mano_canonicalizer(
    cfg: Mapping[str, Any] | Any,
    *,
    device: str | None = None,
) -> FrozenManoCanonicalizer | None:
    if not _cfg_get(cfg, 'enabled', False):
        return None
    extractor_path = _cfg_get(cfg, 'extractor_path', '')
    chunk_size = _cfg_get(cfg, 'chunk_size', 16384)
    max_batch_frames = _cfg_get(cfg, 'max_batch_frames', None)
    return FrozenManoCanonicalizer(
        extractor_path=extractor_path,
        device=device,
        chunk_size=chunk_size,
        max_batch_frames=max_batch_frames,
    )
