#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


DEFAULT_CANONICALIZER_EXTRACTOR = (
    "hand_keypoint_data/ckpts/combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel/"
    "combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel_20260518_132945/best"
)
TIP_JOINT_INDICES = [4, 8, 12, 16, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HyperLoRA conditioned full-sequence predictions."
    )
    parser.add_argument(
        "--model-dir",
        default=(
            "hand_keypoint_data/ckpts/"
            "fair_ditflow_5ds_biasmix_round6_gttrue_no_canon_20260521_134310_20260521_134356/best"
        ),
        help="Trained HyperLoRA checkpoint dir",
    )
    parser.add_argument(
        "--data-path",
        default=(
            "hand_keypoint_data/Interhand_val/"
            "InterHand_retargeted_30fps_biased_super_shapedir_fix_bone50_only_filtered.hdf5"
        ),
        help="HDF5 dataset to export",
    )
    parser.add_argument(
        "--output-dir",
        default="expdata_annie/finetune_baselines_mac_bone50_filtered",
        help="Output root",
    )
    parser.add_argument("--num-calib-actions", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-key", default="world_hand_joints_biased")
    parser.add_argument("--true-key", default="world_hand_joints")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["hypernet", "hypernet_lora_ft"],
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


def choose_calib_indices(length: int, num_calib: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(length)
    if length <= num_calib:
        return all_indices, all_indices
    rng = np.random.default_rng(seed)
    calib_idx = np.sort(rng.choice(all_indices, size=num_calib, replace=False))
    eval_mask = np.ones(length, dtype=bool)
    eval_mask[calib_idx] = False
    return calib_idx, all_indices[eval_mask]


def batch_similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(source, dtype=np.float64)
    for i, (src, tgt) in enumerate(zip(source, target)):
        src_mean = src.mean(axis=0, keepdims=True)
        tgt_mean = tgt.mean(axis=0, keepdims=True)
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean
        src_var = float(np.sum(src_centered ** 2))
        if src_var < 1e-12:
            aligned[i] = np.broadcast_to(tgt_mean, src.shape)
            continue
        u, _, vh = np.linalg.svd(src_centered.T @ tgt_centered)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            vh[-1] *= -1
            rot = u @ vh
        scale = np.trace(rot.T @ src_centered.T @ tgt_centered) / src_var
        aligned[i] = scale * (src_centered @ rot) + tgt_mean
    return aligned.astype(np.float32)


def compute_joint_metrics(pred_joints: np.ndarray, gt_joints: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_joints, dtype=np.float32)
    gt = np.asarray(gt_joints, dtype=np.float32)
    pa_pred = batch_similarity_align(pred, gt)
    mpjpe = float(np.mean(np.linalg.norm(pred - gt, axis=-1)))
    pa_mpjpe = float(np.mean(np.linalg.norm(pa_pred - gt, axis=-1)))
    tip_mpjpe = float(
        np.mean(np.linalg.norm(pred[:, TIP_JOINT_INDICES] - gt[:, TIP_JOINT_INDICES], axis=-1))
    )
    return {
        "mpjpe": mpjpe,
        "mpjpe_mm": mpjpe * 1000.0,
        "pa_mpjpe": pa_mpjpe,
        "pa_mpjpe_mm": pa_mpjpe * 1000.0,
        "tip_mpjpe": tip_mpjpe,
        "tip_mpjpe_mm": tip_mpjpe * 1000.0,
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
        return self.model.predict(x)


def train_hyper_lora_params(
    adapter: TrainableHyperLoRAAdapter,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
) -> tuple[list[float], float]:
    params = [p for p in adapter.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable HyperLoRA parameters")
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    x_tensor = torch.as_tensor(x_calib, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_calib, dtype=torch.float32, device=device)
    losses: list[float] = []
    start = perf_counter()
    adapter.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        result = adapter.train_forward(x_tensor, y_tensor)
        loss = result["loss"].mean()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses, perf_counter() - start


def main() -> None:
    args = parse_args()
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

    with h5py.File(args.data_path, "r") as f:
        session_names = list(f.keys())
        try:
            session_names = sorted(session_names, key=lambda item: int(item))
        except Exception:
            session_names = sorted(session_names)
        if args.max_sessions > 0:
            session_names = session_names[: args.max_sessions]

        total_units = sum(
            len(biased_sample_indices(f[name], args.input_key, args.sample_idx))
            for name in session_names
        )
        unit_idx = 0
        for session_name in tqdm(session_names, desc="hypernet export sessions"):
            group = f[session_name]
            if args.input_key not in group:
                raise KeyError(f"{args.input_key!r} not found in session {session_name}")
            if args.true_key not in group:
                raise KeyError(f"{args.true_key!r} not found in session {session_name}")

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
                if len(biased_seq) != len(true_seq):
                    raise ValueError(
                        f"Length mismatch for {session_name}/sample_{sample_idx}: "
                        f"{len(biased_seq)} vs {len(true_seq)}"
                    )

                calib_idx, eval_idx = choose_calib_indices(
                    len(biased_seq),
                    args.num_calib_actions,
                    args.seed,
                )
                paired_poses = {
                    "world_hand_joints_biased": biased_seq[calib_idx],
                    "world_hand_joints": true_seq[calib_idx],
                }
                side = decode_side(np.asarray(group["mano_side"])) if "mano_side" in group else "right"
                betas = np.asarray(group["betas"], dtype=np.float32) if "betas" in group else None
                gt_joints = np.asarray(transform_joint3d(true_seq), dtype=np.float32)

                for variant in args.variants:
                    output_dir = Path(args.output_dir) / variant / f"k{args.num_calib_actions}"
                    predictions_path = output_dir / "predictions.hdf5"

                    start = perf_counter()
                    loaded_model = fresh_loaded_model()
                    inferencer = RefineModelInferencer(
                        model_dir=args.model_dir,
                        side=side,
                        device=device,
                        paired_poses=paired_poses,
                        fixed_noise=True,
                        loaded_model=loaded_model,
                    )
                    hyper_input = inferencer.last_hyper_input

                    adaptation_time = 0.0
                    losses: list[float] = []
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
                        losses, adaptation_time = train_hyper_lora_params(
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
                    conditioning_time = perf_counter() - start - adaptation_time

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
                        },
                    )
                    rows_by_variant[variant].append(
                        {
                            "session": f"{session_name}/sample_{sample_idx}",
                            "group": session_name,
                            "side": side,
                            "sample_idx": int(sample_idx),
                            "method": variant,
                            "num_calib_actions": int(args.num_calib_actions),
                            "num_calib_samples": int(len(calib_idx)),
                            "num_eval_samples": int(len(eval_idx)),
                            "conditioning_time_sec": float(conditioning_time),
                            "adaptation_time_sec": float(adaptation_time),
                            "inference_time_sec": float(inference_time),
                            "reconstruction_time_sec": float(reconstruction_time),
                            "ft_steps": int(args.ft_steps) if variant == "hypernet_lora_ft" else 0,
                            "initial_adapt_loss": float(losses[0]) if losses else None,
                            "final_adapt_loss": float(losses[-1]) if losses else None,
                            **joint_metrics,
                        }
                    )
                    if adapter is not None:
                        adapter.remove_hooks()

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
