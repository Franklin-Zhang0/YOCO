"""Small, local-checkpoint inference interface for YOCO."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch

_CUDA_INFERENCE_LOCK = RLock()


@contextmanager
def _inference_precision(device: str | torch.device) -> Iterator[None]:
    if torch.device(device).type != "cuda":
        yield
        return
    # TF32 controls are process-global; serialize this API's CUDA calls and
    # restore precision last because allow_tf32's setter changes "medium".
    with _CUDA_INFERENCE_LOCK:
        precision = torch.get_float32_matmul_precision()
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type="cuda", enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.set_float32_matmul_precision(precision)
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


@dataclass(frozen=True)
class YOCOResult:
    """MANO axis-angle pose in radians and wrist-local joints in meters.

    ``theta`` has shape (45,) or (N, 45); ``joints`` has shape (21, 3) or
    (N, 21, 3), matching the single-frame or batched input to ``predict``.
    Joint coordinates use the repository's wrist-origin, palm-aligned frame,
    not the input world coordinate frame.
    """

    theta: np.ndarray
    joints: np.ndarray


def _checkpoint_directory(directory: str | Path) -> Path:
    directory = Path(directory).expanduser().resolve()
    for filename in ("model.pth", "config.yaml", "norm_stats.pkl"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {directory / filename}")
    return directory


def _strict_checkpoint_state(model: torch.nn.Module, directory: Path) -> None:
    # The shared research loader uses strict=False; the public API rejects
    # incomplete state dictionaries instead of silently keeping random weights.
    try:
        state = torch.load(
            directory / "model.pth", map_location="cpu", weights_only=True, mmap=True
        )
    except RuntimeError as error:
        if "mmap can only be used with files saved" not in str(error):
            raise
        state = torch.load(
            directory / "model.pth", map_location="cpu", weights_only=True
        )
    model.load_state_dict(state, strict=True)


def _load_inferencer(
    model_dir: Path,
    canonicalizer_dir: Path,
    side: str,
    device: torch.device,
) -> Any:
    from calibration_net.policy.model import load_model
    from calibration_net.policy.refine_inferencer import RefineModelInferencer

    loaded = load_model(str(model_dir), device=device)
    model, _, cfg = loaded
    _strict_checkpoint_state(model, model_dir)
    if not hasattr(model, "hyperlora") or not hasattr(model, "preference_extractor"):
        raise ValueError("YOCO requires a paired-pose HyperLoRA checkpoint.")
    if cfg["model"].get("output_size") != 45:
        raise ValueError("YOCO requires a checkpoint with 45 MANO pose outputs.")
    canonicalizer_cfg = cfg["dataset"].setdefault("frozen_mano_canonicalizer", {})
    canonicalizer_cfg["enabled"] = True
    canonicalizer_cfg["extractor_path"] = str(canonicalizer_dir)
    model.eval()
    inferencer = RefineModelInferencer(
        str(model_dir), side, device=device, loaded_model=loaded
    )
    _strict_checkpoint_state(inferencer.input_canonicalizer.model, canonicalizer_dir)
    inferencer.model.eval()
    return inferencer


class YOCO:
    """Calibrate YOCO from paired poses, then refine new hand-keypoint frames.

    Inputs are NumPy-compatible arrays in meters, using the repository's
    21-joint MANO order: wrist, then thumb, index, middle, ring and pinky, each
    from its base joint through its fingertip. Calibration targets are matching
    observed, accurate keypoints, not MANO angle vectors.
    """

    def __init__(self, inferencer: Any, *, side: str):
        self._inferencer = inferencer
        self.side = side
        self.device = inferencer.device
        self.num_calibration_pairs = int(
            inferencer.cfg["dataset"].get("num_hypernet_pairs", 16)
        )
        if self.num_calibration_pairs < 1:
            raise ValueError("Checkpoint specifies no calibration pairs.")
        self._calibrated = False

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        canonicalizer_dir: str | Path,
        side: str = "right",
        device: str | torch.device = "cpu",
    ) -> YOCO:
        """Load trusted local checkpoints and licensed MANO assets.

        Both checkpoint directories contain ``model.pth``, ``config.yaml`` and
        ``norm_stats.pkl``. Checkpoint configuration and normalization files are
        trusted inputs; do not load files from unknown sources. ``MANO_PATH``
        must be the absolute path to your licensed MANO model directory. ROS and a robot URDF
        are not required. A requested unavailable accelerator raises an error.
        """
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested, but CUDA is unavailable.")
            if (
                resolved_device.index is not None
                and resolved_device.index >= torch.cuda.device_count()
            ):
                raise ValueError(
                    f"Requested CUDA device {resolved_device} is unavailable."
                )
        elif resolved_device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS was requested, but MPS is unavailable.")
        elif resolved_device.type != "cpu":
            raise ValueError(f"Unsupported inference device: {resolved_device}")

        mano_path = os.environ.get("MANO_PATH")
        if not mano_path:
            raise ValueError("Set MANO_PATH to your licensed MANO model directory.")
        if not Path(mano_path).is_absolute():
            raise ValueError(
                "MANO_PATH must be an absolute path, without an unexpanded '~'."
            )
        for filename in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
            if not (Path(mano_path) / filename).is_file():
                raise FileNotFoundError(
                    f"Missing MANO model: {Path(mano_path) / filename}"
                )

        model_dir = _checkpoint_directory(model_dir)
        canonicalizer_dir = _checkpoint_directory(canonicalizer_dir)
        inferencer = _load_inferencer(
            model_dir, canonicalizer_dir, side, resolved_device
        )
        return cls(inferencer, side=side)

    @staticmethod
    def _joints(value: Any, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim not in (2, 3) or value.shape[-2:] != (21, 3):
            raise ValueError(f"{name} must have shape (21, 3) or (N, 21, 3).")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        if value.ndim == 3 and value.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one frame.")
        return value

    def calibrate(self, raw_pairs: Any, target_pairs: Any) -> YOCO:
        """Start an independent calibration session from exactly K paired poses.

        Both arrays must have shape (K, 21, 3); K is four for the release model.
        Previous adapters are removed before extracting the new conditioning.
        Pairs are never padded or silently subsampled.
        """
        raw_pairs = self._joints(raw_pairs, "raw_pairs")
        target_pairs = self._joints(target_pairs, "target_pairs")
        expected = (self.num_calibration_pairs, 21, 3)
        if raw_pairs.shape != expected or target_pairs.shape != expected:
            raise ValueError(
                f"Calibration requires both arrays to have shape {expected}."
            )
        with _inference_precision(self.device):
            self._calibrated = False
            self._inferencer.model.hyperlora.unmerge_weight()
            self._inferencer.model.eval()
            self._inferencer.update_condition(
                {
                    "world_hand_joints_biased": raw_pairs,
                    "world_hand_joints": target_pairs,
                }
            )
            if self._inferencer.last_hyper_input is None:
                raise RuntimeError(
                    "Checkpoint did not produce calibration conditioning."
                )
            self._calibrated = True
        return self

    def _reconstruct_joints(
        self, theta: np.ndarray, betas: np.ndarray | None
    ) -> np.ndarray:
        from calibration_net.utils import (
            mano_recon,
            scale_hand_fingers,
            transform_joint3d,
        )

        theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        beta_tensor = (
            None
            if betas is None
            else torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        )
        with torch.no_grad():
            _, joints = mano_recon(
                theta_tensor,
                side=self.side,
                betas=beta_tensor,
                apply_left_shapedirs_fix=True,
            )
            network = self._inferencer.model.network
            if hasattr(network, "finger_scaling"):
                joints = scale_hand_fingers(joints, network.finger_scaling)
            joints = transform_joint3d(joints)
        return joints.detach().cpu().numpy().astype(np.float32, copy=False)

    def predict(
        self,
        raw_frames: Any,
        *,
        betas: Any = None,
        batch_size: int = 1024,
    ) -> YOCOResult:
        """Refine one frame or a batch after calibration.

        ``raw_frames`` is (21, 3) or (N, 21, 3). Optional MANO shape ``betas``
        is (10,) for a shared hand shape or (N, 10) for frame-specific shapes;
        omitting it uses the neutral MANO shape. Output joints are wrist-local,
        palm-aligned, in meters, while ``theta`` contains 45 pose radians.
        """
        with _inference_precision(self.device):
            if not self._calibrated:
                raise RuntimeError(
                    "Call calibrate(raw_pairs, target_pairs) before predict()."
                )
            raw_frames = self._joints(raw_frames, "raw_frames")
            single_frame = raw_frames.ndim == 2
            if single_frame:
                raw_frames = raw_frames[None]
            if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or batch_size < 1
            ):
                raise ValueError("batch_size must be a positive integer.")
            if betas is not None:
                betas = np.asarray(betas, dtype=np.float32)
                if betas.shape not in {(10,), (len(raw_frames), 10)}:
                    raise ValueError("betas must have shape (10,) or (N, 10).")
                if not np.isfinite(betas).all():
                    raise ValueError("betas contains NaN or infinite values.")

            theta_chunks = []
            joint_chunks = []
            for start in range(0, len(raw_frames), batch_size):
                frames = raw_frames[start : start + batch_size]
                theta = np.asarray(
                    self._inferencer.inference(frames, batched=True), dtype=np.float32
                ).reshape(len(frames), 45)
                chunk_betas = (
                    betas[start : start + len(frames)]
                    if betas is not None and betas.ndim == 2
                    else betas
                )
                theta_chunks.append(theta)
                joint_chunks.append(self._reconstruct_joints(theta, chunk_betas))
            theta = np.concatenate(theta_chunks)
            joints = np.concatenate(joint_chunks)
            return (
                YOCOResult(theta[0], joints[0])
                if single_frame
                else YOCOResult(theta, joints)
            )
