#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/combined_cnn_mlp_leaky_global_mixer_scheduler_composite_20260308_205854/best}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/hand_keypoint_data/Interhand_val/InterHand_retargeted_30fps_biased_super_shapedir_fix_bone50_only_filtered.hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/expdata_annie/finetune_baselines_mac_bone50_filtered}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_DIR/hypernet_checkpoints}"
EXPNAME="${EXPNAME:-hypernet_k4}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-5}"
K="${K:-4}"
SAMPLE_IDX="${SAMPLE_IDX:--1}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
TRAIN="${TRAIN:-0}"
EXPORT="${EXPORT:-1}"
HYPERNET_MODEL_DIR="${HYPERNET_MODEL_DIR:-$ROOT_DIR/hand_keypoint_data/ckpts/fair_ditflow_5ds_biasmix_round6_gttrue_no_canon_20260521_134310_20260521_134356/best}"

echo "Root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Base model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "K: $K"
echo "Sample idx: $SAMPLE_IDX (-1 means all biased variants)"

if [[ "$TRAIN" == "1" ]]; then
  mkdir -p "$CHECKPOINT_ROOT"
  echo "=== Train HyperLoRA / hypernet ==="
  "$PYTHON_BIN" -m calibration_net.policy.train \
    trainer.expname="$EXPNAME" \
    config=config/hyperlora_calib.yaml \
    dataset.input_keys=world_hand_joints \
    dataset.output_keys=mano_angles \
    dataset.preprocessing_x=[joint_3d_world_to_local] \
    dataset.path="$DATA_PATH" \
    dataset.num_hypernet_pairs="$K" \
    dataset.pair_subsample=False \
    val_dataset.path="$DATA_PATH" \
    trainer.num_epochs="$EPOCHS" \
    trainer.batch_size="$BATCH_SIZE" \
    trainer.learning_rate="$LR" \
    trainer.lr_scheduler.enabled=False \
    trainer.save_path="$CHECKPOINT_ROOT" \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device="$DEVICE" \
    trainer.load_path="$MODEL_DIR" \
    trainer.loss_type=refine_composite \
    trainer.loss_weights.mse=0.01 \
    trainer.loss_weights.rel_distance=1000.0 \
    trainer.loss_weights.rel_vector=1.0 \
    trainer.loss_weights.global_keypoint=100.0 \
    model.iknet_cfg.enabled=False \
    model.type=Combined_CNN_MLP_MIXER \
    model.hypernet_cfg.rank=8 \
    model.hypernet_cfg.name_filter=["mano_regressor","fusion_attention","encoder"] \
    trainer.logger.type=[terminal,file]
fi

if [[ -z "$HYPERNET_MODEL_DIR" ]]; then
  best_model_file="$(find "$CHECKPOINT_ROOT" -type f -path "*/best/model.pth" | sort | tail -n 1)"
  if [[ -z "$best_model_file" ]]; then
    echo "Error: could not find */best/model.pth under $CHECKPOINT_ROOT"
    exit 1
  fi
  HYPERNET_MODEL_DIR="$(dirname "$best_model_file")"
fi

echo "HyperNet model: $HYPERNET_MODEL_DIR"

if [[ "$EXPORT" == "1" ]]; then
  echo "=== Export HyperNet predictions ==="
  "$PYTHON_BIN" scripts/export_hypernet_finetune_predictions.py \
    --model-dir "$HYPERNET_MODEL_DIR" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-calib-actions "$K" \
    --sample-idx "$SAMPLE_IDX" \
    --device "$DEVICE" \
    --max-sessions "$MAX_SESSIONS" \
    --variants hypernet hypernet_lora_ft

  "$PYTHON_BIN" scripts/summarize_finetune_times.py \
    --output-dir "$OUTPUT_DIR" \
    --k "$K"
fi

echo "Done."
echo "Predictions: $OUTPUT_DIR/hypernet/k$K/predictions.hdf5"
