#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

export MANO_PATH="${MANO_PATH:-/data/250010203/franklin/project/hand_teleop/manopth/mano/models}"
PYTHON_BIN="${PYTHON_BIN:-/data/250010203/franklin/envs/teleop/bin/python}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-expdata/fair_hyperlora_rounds/fair_ditflow_glovelike_sliced_round2_4pair_20260518_200916/best}"
SAVE_ROOT="${SAVE_ROOT:-expdata/fair_hyperlora_rounds}"
EXP_NAME="${EXP_NAME:-fair_ditflow_5ds_biasmix_round6_4pair_lr1e5}"
GLVND_RUNTIME_LIB="${GLVND_RUNTIME_LIB:-/data/250010203/franklin/drivers/glvnd-runtime/lib}"

if [[ -d "$GLVND_RUNTIME_LIB" ]]; then
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$GLVND_RUNTIME_LIB:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$GLVND_RUNTIME_LIB"
  fi
fi

DATA_PATHS="${DATA_PATHS:-/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_glovelike.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_biased_glovelike.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_super.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_concat3_900_biased_super.hdf5}"

EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/avp_glove_aligned/aligned_avp_glove.hdf5}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval/results/interhand_val_biased_glove_like_fair_ditflow_round6_5ds_biasmix_lr1e5_cuda_theta_baselines}"
CONFIRM_OUTPUT_DIR="${CONFIRM_OUTPUT_DIR:-eval/results/interhand_val_biased_glove_like_fair_ditflow_round6_5ds_biasmix_lr1e5_cuda_theta_baselines_confirm}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CONFIRM_EVAL="${RUN_CONFIRM_EVAL:-1}"
GLOBAL_KEYPOINT_GT_SOURCE="${GLOBAL_KEYPOINT_GT_SOURCE:-mano_recon}"
GLOBAL_KEYPOINT_BAD_ANNOTATION_THRESHOLD_M="${GLOBAL_KEYPOINT_BAD_ANNOTATION_THRESHOLD_M:-0.05}"

"$PYTHON_BIN" -m calibration_net.policy.train \
  config="$BASE_MODEL_PATH/config.yaml" \
  trainer.expname="$EXP_NAME" \
  dataset.path="$DATA_PATHS" \
  trainer.load_path="$BASE_MODEL_PATH" \
  trainer.num_epochs=6 \
  trainer.learning_rate=1e-5 \
  trainer.lr_scheduler.total_steps=1980 \
  trainer.lr_scheduler.warmup_steps=100 \
  trainer.lr_scheduler.final_lr=1e-6 \
  trainer.save_interval=1 \
  trainer.save_path="$SAVE_ROOT" \
  trainer.global_keypoint_gt_source="$GLOBAL_KEYPOINT_GT_SOURCE" \
  trainer.global_keypoint_bad_annotation_threshold_m="$GLOBAL_KEYPOINT_BAD_ANNOTATION_THRESHOLD_M" \
  trainer.logger.type='[terminal,file]'

RUN_DIR="$(find "$SAVE_ROOT" -maxdepth 1 -type d -name "${EXP_NAME}_*" | sort | tail -n 1)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/best" ]]; then
  echo "Could not find trained best checkpoint for ${EXP_NAME}" >&2
  exit 1
fi

echo "Best checkpoint: $RUN_DIR/best"

if [[ "$RUN_EVAL" == "1" ]]; then
  "$PYTHON_BIN" -B eval/evaluate_calibration.py \
    --dataset-path "$EVAL_DATASET_PATH" \
    --model-path "$RUN_DIR/best" \
    --num-calib-poses 4 \
    --max-aug-sets 2 \
    --baseline-space mano_theta \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --device cuda \
    --no-apply_left_shapedirs_fix
fi

if [[ "$RUN_CONFIRM_EVAL" == "1" ]]; then
  "$PYTHON_BIN" -B eval/evaluate_calibration.py \
    --dataset-path "$EVAL_DATASET_PATH" \
    --model-path "$RUN_DIR/best" \
    --num-calib-poses 4 \
    --max-aug-sets 2 \
    --baseline-space mano_theta \
    --output-dir "$CONFIRM_OUTPUT_DIR" \
    --device cuda \
    --no-apply_left_shapedirs_fix
fi
