#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

export MANO_PATH="${MANO_PATH:-/data/250010203/franklin/project/hand_teleop/manopth/mano/models}"
PYTHON_BIN="${PYTHON_BIN:-/data/250010203/franklin/envs/teleop/bin/python}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-expdata/fair_hyperlora_rounds/fair_ditflow_5ds_biasmix_round6_4pair_lr1e5_all_lora_worldgt_reject005_glvnd_dit8_e16_pa100_20260525_143310/best}"
SAVE_ROOT="${SAVE_ROOT:-expdata/fair_hyperlora_rounds}"
EXP_NAME="${EXP_NAME:-fair_ditflow_5ds_biasmix_round6_4pair_lr1e5_all_lora_worldgt_reject005_glvnd_dit8_e16_pa100_cont8_lr1e6_fullload}"
GLVND_RUNTIME_LIB="${GLVND_RUNTIME_LIB:-/data/250010203/franklin/drivers/glvnd-runtime/lib}"

if [[ -d "$GLVND_RUNTIME_LIB" ]]; then
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$GLVND_RUNTIME_LIB:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$GLVND_RUNTIME_LIB"
  fi
fi

DATA_PATHS="${DATA_PATHS:-/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_glovelike.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_biased_glovelike.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_super.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_concat3_900_biased_super.hdf5}"

EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/InterHand_retargeted_30fps_biased_super.hdf5}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval/results/interhand_val_biased_super_fair_ditflow_round6_worldgt_reject005_all_lora_dit8_e16_pa100_cont8_lr1e6_fullload_right_ours_worldcondtrue}"
RUN_EVAL="${RUN_EVAL:-1}"
LOGGER_TYPES="${LOGGER_TYPES:-[terminal,file]}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

"$PYTHON_BIN" -m calibration_net.policy.train \
  config="$BASE_MODEL_PATH/config.yaml" \
  trainer.expname="$EXP_NAME" \
  dataset.path="$DATA_PATHS" \
  trainer.load_path="$BASE_MODEL_PATH" \
  trainer.num_epochs=8 \
  trainer.learning_rate=1e-6 \
  trainer.lr_scheduler.total_steps=0 \
  trainer.save_interval="$SAVE_INTERVAL" \
  trainer.save_path="$SAVE_ROOT" \
  trainer.global_keypoint_gt_source=world_hand_joints \
  trainer.global_keypoint_bad_annotation_threshold_m=0.05 \
  trainer.load_ignore_mismatched_shapes=false \
  trainer.load_exclude_prefixes='[]' \
  trainer.save_legacy_torch_serialization=True \
  trainer.loss_weights.pa_mpjpe=100.0 \
  model.hypernet_cfg.name_filter='[]' \
  model.hypernet_cfg.max_modules=-1 \
  model.hypernet_cfg.dit_steps=8 \
  trainer.logger.type="$LOGGER_TYPES"

RUN_DIR="$(find "$SAVE_ROOT" -maxdepth 1 -type d -name "${EXP_NAME}_*" | sort | tail -n 1)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/best" ]]; then
  echo "Could not find trained best checkpoint for ${EXP_NAME}" >&2
  exit 1
fi

echo "Best checkpoint: $RUN_DIR/best"

if [[ "$RUN_EVAL" == "1" ]]; then
  "$PYTHON_BIN" -B eval/evaluate_calibration.py \
    --dataset_path "$EVAL_DATASET_PATH" \
    --model_path "$RUN_DIR/best" \
    --apply_left_shapedirs_fix \
    --side-filter right \
    --baseline_space mano_theta \
    --baseline_theta_source hdf5 \
    --methods ours \
    --target_key world_hand_joints \
    --conditioning_target_key world_hand_joints_true \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --device cuda
fi
