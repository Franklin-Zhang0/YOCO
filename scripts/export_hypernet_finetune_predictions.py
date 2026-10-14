#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, ".")

from calibration_net.policy.refine_inferencer import RefineModelInferencer, resolve_torch_device
from calibration_net.policy.models.hyper_lora import lora_hook
from calibration_net.policy.model import load_model
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.utils import mano_recon, scale_hand_fingers, transform_joint3d
from scripts.run_finetune_baselines import compute_joint_metrics


DEFAULT_CANONICALIZER_EXTRACTOR = (
    "hand_keypoint_data/ckpts/combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel/"
    "combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel_20260518_132945/best"
)
TIP_JOINT_INDICES = [4, 8, 12, 16, 20]


def make_calibration_rng(
    seed: int,
    session_name: str,
    sample_idx: int,
    num_calib_actions: int,
) -> np.random.Generator:
    payload = f"calibration|{int(seed)}|{session_name}|{int(sample_idx)}|{int(num_calib_actions)}".encode(
        "utf-8"
    )
    child_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng(child_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HyperLoRA conditioned full-sequence predictions."
    )
    parser.add_argument(
        "--model-dir",
        default=(
            "hand_keypoint_data/ckpts/"
            "fair_ditflow_5ds_biasmix_round6_4pair_lr1e5_all_lora_worldgt_reject005_glvnd_20260524_104940/best"
        ),
        help="Trained HyperLoRA checkpoint dir",
    )
    parser.add_argument(
        "--data-path",
        default=(
            "hand_keypoint_data/Interhand_val/"
            "InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5"
        ),
        help="HDF5 dataset to export",
    )
    parser.add_argument(
        "--output-dir",
        default="expdata_annie/current_export_franklin_aligned_joint20cm_marked_right",
        help="Output root",
    )
    parser.add_argument("--num-calib-actions", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help=(
            "Seed Python, NumPy, and Torch before model initialization and "
            "conditioning. --seed independently controls calibration sampling."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--legacy-conditioning",
        action="store_true",
        help=(
            "Preserve the paper exporter's initial training-mode conditioning. "
            "This enables dropout on the first unit; use --inference-seed "
            "for repeatability. Normal exports "
            "use evaluation mode before conditioning."
        ),
    )
    parser.add_argument(
        "--calib-selection",
        default="random",
        choices=["random", "first", "uniform"],
        help="Calibration frame selection mode. Matches eval/evaluate_calibration.py.",
    )
    parser.add_argument(
        "--exclude-calib-from-eval",
        action="store_true",
        help="Exclude calibration frames from metrics. Default matches Franklin eval: include them.",
    )
    parser.add_argument(
        "--calibration-num",
        "--calibration_num",
        type=int,
        default=0,
        help=(
            "If >0, use frames [0,N) only for calibration selection and "
            "frames [N,T) only for evaluation. Example: --calibration-num 300."
        ),
    )
    parser.add_argument("--input-key", default="world_hand_joints_biased")
    parser.add_argument(
        "--true-key",
        default="world_hand_joints_true",
        help="Clean joint key used only for HyperLoRA conditioning pairs.",
    )
    parser.add_argument(
        "--paired-target-key",
        default="world_hand_joints",
        help=(
            "Key name used inside paired_poses for the clean calibration joints. "
            "Franklin eval reads --true-key but passes it to the inferencer as world_hand_joints."
        ),
    )
    parser.add_argument(
        "--metric-gt-key",
        default="world_hand_joints",
        help=(
            "GT joint key used for MPJPE/PA-MPJPE/tip MPJPE metrics. "
            "This is intentionally separate from --true-key: Franklin-style "
            "conditioning can use world_hand_joints_true while metrics use "
            "normal-beta world_hand_joints."
        ),
    )
    parser.add_argument(
        "--pred-beta-source",
        default="auto",
        choices=["auto", "group", "neutral"],
        help=(
            "MANO beta used for predicted-theta FK. auto uses neutral beta for "
            "world_hand_joints_true targets and group betas otherwise."
        ),
    )
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument(
        "--side-filter",
        default="right",
        choices=["all", "left", "right"],
        help="Only export/evaluate one MANO side. Default: right, matching Franklin eval.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["hypernet"],
        choices=["hypernet", "hypernet_lora_ft"],
        help="Which HyperLoRA variants to export/evaluate.",
    )
    parser.add_argument("--ft-steps", type=int, default=200)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--ft-weight-decay", type=float, default=0.0)
    parser.add_argument("--ft-grad-clip", type=float, default=1.0)
    parser.add_argument("--ft-train-gates", action="store_true")
    parser.add_argument(
        "--canonicalizer-extractor-path",
        default=DEFAULT_CANONICALIZER_EXTRACTOR,
        help=(
            "Override missing frozen MANO canonicalizer extractor path in old HyperLoRA configs. "
            "Use empty string to disable this override."
        ),
    )
    return parser.parse_args()


def decode_side(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "right"
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value).lower()


def read_sequence(group: h5py.Group, key: str, sample_idx: int = 0) -> np.ndarray:
    array = np.asarray(group[key])
    if array.ndim >= 4 and key.endswith("_biased"):
        sample_idx = min(max(sample_idx, 0), array.shape[0] - 1)
        array = array[sample_idx]
    return array.astype(np.float32, copy=False)


def biased_sample_indices(group: h5py.Group, input_key: str, sample_idx: int) -> list[int]:
    if input_key not in group:
        return [0]
    shape = group[input_key].shape
    if len(shape) >= 4 and input_key.endswith("_biased"):
        if sample_idx < 0:
            return list(range(shape[0]))
        return [min(sample_idx, shape[0] - 1)]
    return [0]


def read_bad_frame_mask(group: h5py.Group, length: int) -> np.ndarray | None:
    if "bad_frame" not in group:
        return None
    mask = np.asarray(group["bad_frame"], dtype=bool).reshape(-1)
    if len(mask) != length:
        raise ValueError(f"bad_frame length {len(mask)} does not match sequence length {length}")
    return mask


def read_valid_joint_mask(group: h5py.Group, length: int) -> np.ndarray | None:
    if "gt_joint_mask" not in group:
        return None
    bad = np.asarray(group["gt_joint_mask"], dtype=bool)
    if bad.shape != (length, 21):
        raise ValueError(f"gt_joint_mask shape {bad.shape} does not match expected {(length, 21)}")
    return ~bad


def select_mano_betas(
    group: h5py.Group,
    metric_gt_key: str,
    pred_beta_source: str,
) -> tuple[np.ndarray | None, str]:
    source = pred_beta_source
    if source == "auto":
        source = "neutral" if metric_gt_key == "world_hand_joints_true" else "group"
    if source == "neutral":
        return None, source
    if "betas" not in group:
        return None, "missing_group"
    return np.asarray(group["betas"], dtype=np.float32), source


def choose_calib_indices(
    length: int,
    num_calib: int,
    rng: np.random.Generator,
    mode: str,
    exclude_calib_from_eval: bool,
    bad_frame_mask: np.ndarray | None = None,
    calibration_num: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(length)
    good_indices = all_indices if bad_frame_mask is None else all_indices[~bad_frame_mask]
    if len(good_indices) == 0:
        raise ValueError("No non-bad frames available for HyperLoRA calibration/eval")
    eval_indices = None
    if calibration_num > 0:
        split_at = int(calibration_num)
        calib_candidates = good_indices[good_indices < split_at]
        eval_indices = all_indices[all_indices >= split_at]
        if len(calib_candidates) == 0:
            raise ValueError(f"No non-bad frames in [0,{split_at}) for HyperLoRA calibration")
        if len(eval_indices) == 0:
            raise ValueError(f"No frames in [{split_at},T) for HyperLoRA evaluation")
    else:
        calib_candidates = good_indices
    count = min(max(int(num_calib), 1), len(calib_candidates))

    if mode == "first":
        calib_idx = calib_candidates[:count]
    elif mode == "random":
        calib_idx = np.sort(rng.choice(calib_candidates, size=count, replace=False))
    elif mode == "uniform":
        if count == 1:
            calib_idx = calib_candidates[[0]]
        else:
            uniform_idx = np.linspace(0, len(calib_candidates) - 1, num=count).round().astype(np.int64)
            calib_idx = calib_candidates[uniform_idx]
    else:
        raise ValueError("calib-selection must be 'random', 'first', or 'uniform'")

    if calibration_num > 0:
        return calib_idx, eval_indices

    eval_mask = np.ones(length, dtype=bool)
    if exclude_calib_from_eval:
        eval_mask[calib_idx] = False
        if not eval_mask.any():
            raise ValueError("exclude-calib-from-eval leaves no frames to evaluate")
    return calib_idx, all_indices[eval_mask]


def _similarity_align_one(
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if fit_mask is None:
        src_fit = src
        tgt_fit = tgt
    else:
        valid = np.asarray(fit_mask, dtype=bool)
        if int(valid.sum()) < 3:
            return src.copy()
        src_fit = src[valid]
        tgt_fit = tgt[valid]

    src_mean = src_fit.mean(axis=0, keepdims=True)
    tgt_mean = tgt_fit.mean(axis=0, keepdims=True)
    src_centered = src_fit - src_mean
    tgt_centered = tgt_fit - tgt_mean
    src_var = float(np.sum(src_centered ** 2))
    if src_var < 1e-12:
        return np.broadcast_to(tgt_mean, src.shape).copy()

    cov = src_centered.T @ tgt_centered
    u, singular_values, vh = np.linalg.svd(cov)
    z = np.eye(cov.shape[0], dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        z[-1, -1] = -1.0
    rot = u @ z @ vh
    scale = float(np.sum(singular_values * np.diag(z)) / src_var)
    return scale * ((src - src_mean) @ rot) + tgt_mean


def batch_similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack(
        [_similarity_align_one(src, tgt) for src, tgt in zip(source, target)],
        axis=0,
    ).astype(np.float32)


def batch_similarity_align_masked(
    source: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    if valid_mask is None:
        return batch_similarity_align(source, target)
    return np.stack(
        [
            _similarity_align_one(src, tgt, valid)
            for src, tgt, valid in zip(source, target, valid_mask)
        ],
        axis=0,
    ).astype(np.float32)


def compute_theta_metrics(pred_theta: np.ndarray, gt_theta: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_theta, dtype=np.float32)
    gt = np.asarray(gt_theta, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred theta shape {pred.shape} does not match gt theta shape {gt.shape}")
    return {
        "theta_mse": float(np.mean((pred - gt) ** 2)),
    }


def reconstruct_joints(
    pred_theta_np: np.ndarray,
    inferencer: RefineModelInferencer,
    side: str,
    betas: np.ndarray | None,
) -> np.ndarray:
    pred_theta = torch.as_tensor(
        pred_theta_np,
        dtype=torch.float32,
        device=inferencer.device,
    )
    betas_tensor = None
    if betas is not None:
        betas_tensor = torch.as_tensor(
            betas,
            dtype=torch.float32,
            device=inferencer.device,
        )
        if betas_tensor.ndim == 1:
            betas_tensor = betas_tensor.unsqueeze(0).expand(pred_theta.shape[0], -1)

    with torch.no_grad():
        _, pred_joints = mano_recon(
            pred_theta,
            side=side,
            betas=betas_tensor,
            apply_left_shapedirs_fix=True,
        )
        model = inferencer.model
        if hasattr(model, "network") and hasattr(model.network, "finger_scaling"):
            pred_joints = scale_hand_fingers(pred_joints, model.network.finger_scaling)

        pred_local = transform_joint3d(pred_joints)
        pred_local = torch.as_tensor(
            pred_local,
            dtype=torch.float32,
            device=inferencer.device,
        )
    return pred_local.detach().cpu().numpy().astype(np.float32)


def save_predictions(
    predictions_path: Path,
    session_name: str,
    sample_idx: int,
    pred_theta: np.ndarray,
    pred_joints: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    attrs: dict[str, Any],
) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(predictions_path, "a") as f:
        group = f.require_group(session_name)
        sample_group = group.require_group(f"sample_{sample_idx}")
        for key in ("pred_theta", "pred_joints", "calib_idx", "eval_idx"):
            if key in sample_group:
                del sample_group[key]
        sample_group.create_dataset("pred_theta", data=pred_theta, compression="gzip")
        sample_group.create_dataset("pred_joints", data=pred_joints, compression="gzip")
        sample_group.create_dataset("calib_idx", data=calib_idx.astype(np.int64))
        sample_group.create_dataset("eval_idx", data=eval_idx.astype(np.int64))
        for key, value in attrs.items():
            sample_group.attrs[key] = value


class TrainableHyperLoRAAdapter(nn.Module):
    def __init__(self, model: nn.Module, hyper_input: torch.Tensor, train_gates: bool = False):
        super().__init__()
        self.model = model
        self.hyperlora = model.hyperlora
        self.rank = self.hyperlora.rank
        self.alpha = self.hyperlora.alpha
        self.drop_out = self.hyperlora.drop_out
        self.max_norm = self.hyperlora.max_norm
        self.train_gates = train_gates
        for param in self.model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            initial_weights = self.hyperlora.prepare_lora_weights(hyper_input)
        self.lora_params = nn.ParameterList(
            [nn.Parameter(weight.detach().clone()) for _layer, weight in initial_weights]
        )
        if self.hyperlora.use_gates and train_gates:
            for gate in self.hyperlora.gates:
                gate.requires_grad = True

        self._hooks = []
        self._register_hooks()
        self._assert_only_generated_lora_is_trainable()

    def _trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _assert_only_generated_lora_is_trainable(self) -> None:
        allowed_prefixes = ("lora_params.",)
        if self.train_gates:
            allowed_prefixes = allowed_prefixes + ("model.hyperlora.gates.",)
        unexpected = [
            name
            for name in self._trainable_parameter_names()
            if not name.startswith(allowed_prefixes)
        ]
        if unexpected:
            raise RuntimeError(
                "Unexpected trainable parameters in HyperLoRA finetune. "
                "Only generated LoRA tensors should be tuned, but got: "
                + ", ".join(unexpected[:20])
            )

    def _register_hooks(self) -> None:
        self.remove_hooks()
        for idx, ((layer, _module), lora_param) in enumerate(
            zip(self.hyperlora.modules_list, self.lora_params)
        ):
            gate = self.hyperlora.gates[idx] if self.hyperlora.use_gates else None
            handle = layer.register_forward_hook(
                lambda module, inputs, output, lora_param=lora_param, gate=gate: lora_hook(
                    module,
                    inputs,
                    output,
                    lora_weight=lora_param,
                    rank=self.rank,
                    alpha=self.alpha,
                    drop_out=self.drop_out,
                    gate=gate,
                    max_norm=self.max_norm,
                )
            )
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def train_forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_y = self.model.forward(x)
        if isinstance(pred_y, dict):
            pred_y = pred_y["pred_y"]
        return {"pred_y": pred_y, "loss": F.mse_loss(pred_y, y, reduction="none")}

    @torch.no_grad()
    def predict_theta(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.model.predict(x)

    @property
    def num_trainable_lora_tensors(self) -> int:
        return len(self.lora_params)


def train_hyper_lora_params(
    adapter: TrainableHyperLoRAAdapter,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
) -> tuple[list[float], float, float, float]:
    params = [p for p in adapter.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable HyperLoRA parameters")
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    x_tensor = torch.as_tensor(x_calib, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_calib, dtype=torch.float32, device=device)

    def keep_frozen_modules_eval() -> None:
        for child in adapter.modules():
            child_params = list(child.parameters(recurse=True))
            if child_params and not any(param.requires_grad for param in child_params):
                child.eval()
        # HyperLoRA dropout is implemented in lora_hook through the target
        # Linear module's training flag. Keep only those stateless Linear
        # layers in train mode; frozen BatchNorm/dropout parents stay eval.
        for layer, _module in adapter.hyperlora.modules_list:
            layer.train()

    @torch.no_grad()
    def eval_loss() -> float:
        adapter.eval()
        result = adapter.train_forward(x_tensor, y_tensor)
        return float(result["loss"].mean().detach().cpu())

    def clone_trainable_params() -> list[torch.Tensor]:
        return [param.detach().clone() for param in params]

    def restore_trainable_params(state: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for param, saved in zip(params, state):
                param.copy_(saved)

    losses: list[float] = []
    start = perf_counter()
    initial_loss = eval_loss()
    best_loss = initial_loss
    best_state = clone_trainable_params()
    adapter.train()
    keep_frozen_modules_eval()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        result = adapter.train_forward(x_tensor, y_tensor)
        loss = result["loss"].mean()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        optimizer.step()
        post_step_loss = eval_loss()
        losses.append(post_step_loss)
        if post_step_loss < best_loss:
            best_loss = post_step_loss
            best_state = clone_trainable_params()
        adapter.train()
        keep_frozen_modules_eval()
    restore_trainable_params(best_state)
    return losses, perf_counter() - start, initial_loss, best_loss


def main() -> None:
    args = parse_args()
    if args.inference_seed is not None:
        random.seed(args.inference_seed)
        np.random.seed(args.inference_seed)
        torch.manual_seed(args.inference_seed)
    device = resolve_torch_device(args.device)
    rows_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in args.variants}
    run_start_by_variant: dict[str, float] = {variant: perf_counter() for variant in args.variants}

    def fresh_loaded_model():
        loaded = load_model(args.model_dir, device=str(device))
        if args.canonicalizer_extractor_path:
            cfg = loaded[2]
            canonicalizer_cfg = cfg.get("dataset", {}).get("frozen_mano_canonicalizer", {})
            if canonicalizer_cfg.get("enabled", False):
                canonicalizer_cfg["extractor_path"] = args.canonicalizer_extractor_path
        return loaded

    inferencer: RefineModelInferencer | None = None

    def get_inferencer(side: str) -> RefineModelInferencer:
        nonlocal inferencer
        if inferencer is None:
            inferencer = RefineModelInferencer(
                model_dir=args.model_dir,
                side=side,
                device=device,
                paired_poses=None,
                fixed_noise=True,
                loaded_model=fresh_loaded_model(),
            )
        else:
            inferencer.set_side(side)
        return inferencer

    with h5py.File(args.data_path, "r") as f:
        session_names = list(f.keys())
        try:
            session_names = sorted(session_names, key=lambda item: int(item))
        except Exception:
            session_names = sorted(session_names)
        if args.max_sessions > 0:
            session_names = session_names[: args.max_sessions]

        total_units = 0
        for name in session_names:
            group = f[name]
            side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
            if args.side_filter != "all" and side != args.side_filter:
                continue
            total_units += len(biased_sample_indices(group, args.input_key, args.sample_idx))
        unit_idx = 0
        for session_name in tqdm(session_names, desc="hypernet export sessions"):
            group = f[session_name]
            side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
            if args.side_filter != "all" and side != args.side_filter:
                continue
            if args.input_key not in group:
                raise KeyError(f"{args.input_key!r} not found in session {session_name}")
            if args.true_key not in group:
                raise KeyError(f"{args.true_key!r} not found in session {session_name}")
            if args.metric_gt_key not in group:
                raise KeyError(f"{args.metric_gt_key!r} not found in session {session_name}")
            if "mano_angles" not in group:
                raise KeyError(f"'mano_angles' not found in session {session_name}")

            sample_indices = biased_sample_indices(group, args.input_key, args.sample_idx)
            for sample_idx in tqdm(
                sample_indices,
                desc=f"{session_name} samples",
                leave=False,
                disable=len(sample_indices) <= 1,
            ):
                unit_idx += 1
                print(f"[{unit_idx}/{total_units}] hypernet variants={','.join(args.variants)} k={args.num_calib_actions} session={session_name}/sample_{sample_idx}")

                biased_seq = read_sequence(group, args.input_key, sample_idx=sample_idx)
                true_seq = read_sequence(group, args.true_key, sample_idx=sample_idx)
                metric_gt_seq = read_sequence(group, args.metric_gt_key, sample_idx=0)
                gt_theta_seq = read_sequence(group, "mano_angles", sample_idx=0)
                if len(biased_seq) != len(true_seq) or len(biased_seq) != len(metric_gt_seq):
                    raise ValueError(
                        f"Length mismatch for {session_name}/sample_{sample_idx}: "
                        f"biased={len(biased_seq)} true={len(true_seq)} metric_gt={len(metric_gt_seq)}"
                    )
                if len(biased_seq) != len(gt_theta_seq):
                    raise ValueError(
                        f"Length mismatch for {session_name}/sample_{sample_idx}: "
                        f"biased={len(biased_seq)} mano_angles={len(gt_theta_seq)}"
                    )
                bad_frame_mask = read_bad_frame_mask(group, len(biased_seq))
                valid_joint_mask_all = read_valid_joint_mask(group, len(biased_seq))

                calib_rng = make_calibration_rng(
                    args.seed,
                    session_name,
                    sample_idx,
                    args.num_calib_actions,
                )
                calib_idx, eval_idx = choose_calib_indices(
                    len(biased_seq),
                    args.num_calib_actions,
                    calib_rng,
                    args.calib_selection,
                    args.exclude_calib_from_eval,
                    bad_frame_mask=bad_frame_mask,
                    calibration_num=args.calibration_num,
                )
                paired_poses = {
                    "world_hand_joints_biased": biased_seq[calib_idx],
                    args.paired_target_key: true_seq[calib_idx],
                }
                betas, beta_source = select_mano_betas(
                    group,
                    args.metric_gt_key,
                    args.pred_beta_source,
                )
                gt_joints = np.asarray(transform_joint3d(metric_gt_seq), dtype=np.float32)
                inferencer = get_inferencer(side)

                for variant in args.variants:
                    output_dir = Path(args.output_dir) / variant / f"k{args.num_calib_actions}"
                    predictions_path = output_dir / "predictions.hdf5"

                    if not args.legacy_conditioning:
                        inferencer.model.eval()
                    condition_start = perf_counter()
                    inferencer.update_condition(paired_poses=paired_poses)
                    conditioning_time = perf_counter() - condition_start
                    hyper_input = inferencer.last_hyper_input

                    adaptation_time = 0.0
                    losses: list[float] = []
                    initial_adapt_loss = None
                    best_adapt_loss = None
                    adapter = None
                    if variant == "hypernet":
                        pass
                    elif variant == "hypernet_lora_ft":
                        if hyper_input is None:
                            raise RuntimeError("HyperLoRA conditioning did not produce hyper_input")
                        inferencer.model.hyperlora.unmerge_weight()
                        adapter = TrainableHyperLoRAAdapter(
                            inferencer.model,
                            hyper_input,
                            train_gates=args.ft_train_gates,
                        )
                        x_calib_raw = inferencer._canonicalize_raw_joints(biased_seq[calib_idx])
                        x_calib = np.asarray(inferencer.process_x_fn(x_calib_raw), dtype=np.float32)
                        if "mano_angles" not in group:
                            raise KeyError(f"'mano_angles' not found in session {session_name}")
                        process_y_train = build_preprocessing(
                            inferencer.cfg["dataset"].get("preprocessing_y", []),
                            "y",
                            norm_stats=inferencer.norm_stats,
                        )
                        y_calib = np.asarray(
                            process_y_train(np.asarray(group["mano_angles"][calib_idx], dtype=np.float32)),
                            dtype=np.float32,
                        )
                        (
                            losses,
                            adaptation_time,
                            initial_adapt_loss,
                            best_adapt_loss,
                        ) = train_hyper_lora_params(
                            adapter,
                            x_calib=x_calib,
                            y_calib=y_calib,
                            steps=args.ft_steps,
                            lr=args.ft_lr,
                            weight_decay=args.ft_weight_decay,
                            grad_clip=args.ft_grad_clip,
                            device=inferencer.device,
                        )
                    else:
                        raise ValueError(f"Unknown variant: {variant}")

                    start = perf_counter()
                    if adapter is None:
                        pred_theta = np.asarray(
                            inferencer.inference(biased_seq, batched=True),
                            dtype=np.float32,
                        )
                    else:
                        all_raw = inferencer._canonicalize_raw_joints(biased_seq)
                        all_proc = np.asarray(inferencer.process_x_fn(all_raw), dtype=np.float32)
                        with torch.no_grad():
                            pred_proc = adapter.predict_theta(
                                torch.as_tensor(all_proc, dtype=torch.float32, device=inferencer.device)
                            )
                        pred_theta = np.asarray(
                            inferencer.process_y_fn(pred_proc.detach().cpu().numpy()),
                            dtype=np.float32,
                        )
                    inference_time = perf_counter() - start

                    start = perf_counter()
                    pred_joints = reconstruct_joints(pred_theta, inferencer, side, betas)
                    reconstruction_time = perf_counter() - start
                    joint_metrics = compute_joint_metrics(
                        pred_joints[eval_idx],
                        gt_joints[eval_idx],
                        valid_joint_mask_all[eval_idx] if valid_joint_mask_all is not None else None,
                    )
                    theta_metrics = compute_theta_metrics(
                        pred_theta[eval_idx],
                        gt_theta_seq[eval_idx],
                    )

                    save_predictions(
                        predictions_path,
                        session_name=session_name,
                        sample_idx=sample_idx,
                        pred_theta=pred_theta,
                        pred_joints=pred_joints,
                        calib_idx=calib_idx,
                        eval_idx=eval_idx,
                        attrs={
                            "method": variant,
                            "num_calib_actions": int(args.num_calib_actions),
                            "model_dir": str(args.model_dir),
                            "conditioning_mode": (
                                "legacy" if args.legacy_conditioning else "eval"
                            ),
                            "inference_seed": (
                                args.inference_seed
                                if args.inference_seed is not None
                                else -1
                            ),
                            "calib_selection": args.calib_selection,
                            "exclude_calib_from_eval": bool(args.exclude_calib_from_eval),
                            "calibration_num": int(args.calibration_num),
                            "input_key": args.input_key,
                            "true_key": args.true_key,
                            "paired_target_key": args.paired_target_key,
                            "metric_gt_key": args.metric_gt_key,
                            "pred_beta_source": beta_source,
                        },
                    )
                    rows_by_variant[variant].append(
                        {
                            "session": f"{session_name}/sample_{sample_idx}",
                            "group": session_name,
                            "side": side,
                            "sample_idx": int(sample_idx),
                            "method": variant,
                            "conditioning_mode": (
                                "legacy" if args.legacy_conditioning else "eval"
                            ),
                            "inference_seed": args.inference_seed,
                            "num_calib_actions": int(args.num_calib_actions),
                            "num_calib_samples": int(len(calib_idx)),
                            "num_eval_samples": int(len(eval_idx)),
                            "calib_selection": args.calib_selection,
                            "exclude_calib_from_eval": bool(args.exclude_calib_from_eval),
                            "calibration_num": int(args.calibration_num),
                            "input_key": args.input_key,
                            "true_key": args.true_key,
                            "paired_target_key": args.paired_target_key,
                            "metric_gt_key": args.metric_gt_key,
                            "pred_beta_source": beta_source,
                            "calib_idx": " ".join(str(int(i)) for i in calib_idx),
                            "conditioning_time_sec": float(conditioning_time),
                            "adaptation_time_sec": float(adaptation_time),
                            "inference_time_sec": float(inference_time),
                            "reconstruction_time_sec": float(reconstruction_time),
                            "ft_steps": int(args.ft_steps) if variant == "hypernet_lora_ft" else 0,
                            "initial_adapt_loss": (
                                float(initial_adapt_loss)
                                if initial_adapt_loss is not None
                                else None
                            ),
                            "final_adapt_loss": float(losses[-1]) if losses else None,
                            "best_adapt_loss": (
                                float(best_adapt_loss)
                                if best_adapt_loss is not None
                                else None
                            ),
                            "trainable_params": int(
                                sum(p.numel() for p in adapter.parameters() if p.requires_grad)
                            ) if adapter is not None else 0,
                            "trainable_lora_tensors": int(
                                adapter.num_trainable_lora_tensors
                            ) if adapter is not None else 0,
                            "trainable_param_names": " ".join(
                                adapter._trainable_parameter_names()
                            ) if adapter is not None else "",
                            **joint_metrics,
                            **theta_metrics,
                        }
                    )
                    if adapter is not None:
                        adapter.remove_hooks()
                        if hasattr(inferencer.model, "hyperlora"):
                            inferencer.model.hyperlora.unmerge_weight()

    for variant, rows in rows_by_variant.items():
        summary = {
            "num_units": len(rows),
            "wall_time_sec": float(perf_counter() - run_start_by_variant[variant]),
        }
        for key in (
            "conditioning_time_sec",
            "adaptation_time_sec",
            "inference_time_sec",
            "reconstruction_time_sec",
            "mpjpe",
            "mpjpe_mm",
            "pa_mpjpe",
            "pa_mpjpe_mm",
            "tip_mpjpe",
            "tip_mpjpe_mm",
            "theta_mse",
        ):
            values = [row[key] for row in rows if row.get(key) is not None]
            if values:
                summary[f"{key}_total"] = float(np.sum(values))
                summary[f"{key}_mean"] = float(np.mean(values))

        output_dir = Path(args.output_dir) / variant / f"k{args.num_calib_actions}"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"sessions": rows, "summary": summary}, f, indent=2, sort_keys=True)

        print(f"Saved {variant} predictions to {output_dir / 'predictions.hdf5'}")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
