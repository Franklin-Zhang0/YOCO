#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-/Users/annie/miniconda3/envs/teleop/bin/python}"
export MANO_PATH="${MANO_PATH:-$ROOT_DIR/manopth/mano/models}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/hand_keypoint_data/Interhand_val/InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/expdata_annie/current_export_franklin_aligned_joint20cm_marked_right}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/combined_cnn_mlp_leaky_global_mixer_scheduler_composite_20260308_205854/best}"
HYPERNET_MODEL_DIR="${HYPERNET_MODEL_DIR:-$ROOT_DIR/hand_keypoint_data/ckpts/fair_ditflow_5ds_biasmix_round6_4pair_lr1e5_all_lora_worldgt_reject005_glvnd_20260524_104940/best}"
CANONICALIZER_EXTRACTOR_PATH="${CANONICALIZER_EXTRACTOR_PATH:-$ROOT_DIR/hand_keypoint_data/ckpts/combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel/combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel_20260518_132945/best}"

DEVICE="${DEVICE:-cpu}"
K="${K:-4}"
SEED="${SEED:-42}"
SAMPLE_IDX="${SAMPLE_IDX:--1}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
STEPS="${STEPS:-200}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_USE_GATES="${LORA_USE_GATES:-1}"
LORA_GATE_INIT="${LORA_GATE_INIT:-0.0}"
MAX_JOBS="${MAX_JOBS:-4}"
PORT="${PORT:-8087}"
SIDE_FILTER="${SIDE_FILTER:-right}"
HYPERNET_TRUE_KEY="${HYPERNET_TRUE_KEY:-${TRUE_JOINTS_KEY:-world_hand_joints_true}}"
METRIC_GT_KEY="${METRIC_GT_KEY:-world_hand_joints}"
BASELINE_TRUE_JOINTS_KEY="${BASELINE_TRUE_JOINTS_KEY:-}"
PAIRED_TARGET_KEY="${PAIRED_TARGET_KEY:-world_hand_joints}"
EXCLUDE_CALIB_FROM_EVAL="${EXCLUDE_CALIB_FROM_EVAL:-0}"
CALIBRATION_NUM="${CALIBRATION_NUM:-0}"
PRED_BETA_SOURCE="${PRED_BETA_SOURCE:-auto}"

RUN_BEST="${RUN_BEST:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_HYPERNET="${RUN_HYPERNET:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_VISUALIZE="${RUN_VISUALIZE:-1}"

METHODS=(best_world_hand_joints_biased best_world_hand_joints best_world_hand_joints_true gt_mano_fk lora full_ft residual_linear last_layer hypernet hypernet_lora_ft)

usage() {
  cat <<EOF
Usage: bash scripts/run_full_finetune_hypernet_visualize_mac.sh [options]

Options:
  --metric-gt-key KEY            Metric target joint key: world_hand_joints or world_hand_joints_true.
                                 Applies to best exports, baseline finetune metrics, HyperNet metrics,
                                 and Viser frame MPJPE. Default: $METRIC_GT_KEY
  --baseline-metric-gt-key KEY   Override baseline finetune metric target only. Default: follows --metric-gt-key
  --calibration-num N            Select calibration frames only from [0,N), evaluate only on [N,T).
  -h, --help                     Show this help.

Environment variables like RUN_BEST, RUN_BASELINES, RUN_HYPERNET, RUN_SUMMARY,
RUN_VISUALIZE, DATA_PATH, OUTPUT_DIR, SIDE_FILTER, and PRED_BETA_SOURCE are still supported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metric-gt-key|--metrics-gt-key)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value" >&2
        exit 2
      fi
      METRIC_GT_KEY="$2"
      shift 2
      ;;
    --baseline-metric-gt-key)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value" >&2
        exit 2
      fi
      BASELINE_TRUE_JOINTS_KEY="$2"
      shift 2
      ;;
    --calibration-num|--calibration_num)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value" >&2
        exit 2
      fi
      CALIBRATION_NUM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$METRIC_GT_KEY" in
  world_hand_joints|world_hand_joints_true) ;;
  *)
    echo "Error: --metric-gt-key must be world_hand_joints or world_hand_joints_true, got '$METRIC_GT_KEY'" >&2
    exit 2
    ;;
esac

if [[ -z "$BASELINE_TRUE_JOINTS_KEY" ]]; then
  BASELINE_TRUE_JOINTS_KEY="$METRIC_GT_KEY"
fi
case "$BASELINE_TRUE_JOINTS_KEY" in
  world_hand_joints|world_hand_joints_true) ;;
  *)
    echo "Error: --baseline-metric-gt-key must be world_hand_joints or world_hand_joints_true, got '$BASELINE_TRUE_JOINTS_KEY'" >&2
    exit 2
    ;;
esac

echo "Root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "MANO_PATH: $MANO_PATH"
echo "Dataset: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Best/base model: $MODEL_DIR"
echo "HyperNet model: $HYPERNET_MODEL_DIR"
echo "Canonicalizer extractor: $CANONICALIZER_EXTRACTOR_PATH"
echo "Device: $DEVICE"
echo "K: $K"
echo "Sample idx: $SAMPLE_IDX (-1 means all biased variants)"
echo "Max sessions: $MAX_SESSIONS (0 means all)"
echo "Side filter: $SIDE_FILTER"
echo "HyperNet calibration world_hand_joints_true key: $HYPERNET_TRUE_KEY"
echo "Baseline metric GT key: $BASELINE_TRUE_JOINTS_KEY"
echo "Metric GT key: $METRIC_GT_KEY"
echo "Pred beta source: $PRED_BETA_SOURCE"
echo "Calibration split num: $CALIBRATION_NUM"
echo "Baseline LoRA rank/alpha/dropout: $LORA_RANK / $LORA_ALPHA / $LORA_DROPOUT"
echo "Baseline LoRA use gates/gate init: $LORA_USE_GATES / $LORA_GATE_INIT"

mkdir -p "$OUTPUT_DIR/logs"

calibrate_args=()
if [[ "$CALIBRATION_NUM" != "0" ]]; then
  calibrate_args=(--calibration-num "$CALIBRATION_NUM")
fi

if [[ "$RUN_BEST" == "1" ]]; then
  echo "=== Export no-finetune baselines and GT MANO FK ==="
  if [[ "$EXCLUDE_CALIB_FROM_EVAL" == "1" ]]; then
    "$PYTHON_BIN" scripts/export_best_model_predictions.py \
      --model-dir "$MODEL_DIR" \
      --data-path "$DATA_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --num-calib-actions "$K" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --sample-idx "$SAMPLE_IDX" \
      --max-sessions "$MAX_SESSIONS" \
      --gt-key "$METRIC_GT_KEY" \
      --pred-beta-source "$PRED_BETA_SOURCE" \
      --side-filter "$SIDE_FILTER" \
      "${calibrate_args[@]}" \
      --exclude-calib-from-eval \
      --variants best_world_hand_joints_biased best_world_hand_joints best_world_hand_joints_true gt_mano_fk \
      2>&1 | tee "$OUTPUT_DIR/logs/export_best_model_k${K}.log"
  else
    "$PYTHON_BIN" scripts/export_best_model_predictions.py \
      --model-dir "$MODEL_DIR" \
      --data-path "$DATA_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --num-calib-actions "$K" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --sample-idx "$SAMPLE_IDX" \
      --max-sessions "$MAX_SESSIONS" \
      --gt-key "$METRIC_GT_KEY" \
      --pred-beta-source "$PRED_BETA_SOURCE" \
      --side-filter "$SIDE_FILTER" \
      "${calibrate_args[@]}" \
      --variants best_world_hand_joints_biased best_world_hand_joints best_world_hand_joints_true gt_mano_fk \
      2>&1 | tee "$OUTPUT_DIR/logs/export_best_model_k${K}.log"
  fi
fi

if [[ "$RUN_BASELINES" == "1" ]]; then
  echo "=== Run per-seq finetune baselines ==="
  PYTHON="$PYTHON_BIN" \
  MODEL_DIR="$MODEL_DIR" \
  DATA_PATH="$DATA_PATH" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  DEVICE="$DEVICE" \
  STEPS="$STEPS" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  LORA_RANK="$LORA_RANK" \
  LORA_ALPHA="$LORA_ALPHA" \
  LORA_DROPOUT="$LORA_DROPOUT" \
  LORA_USE_GATES="$LORA_USE_GATES" \
  LORA_GATE_INIT="$LORA_GATE_INIT" \
  SEED="$SEED" \
  MAX_JOBS="$MAX_JOBS" \
  MAX_SESSIONS="$MAX_SESSIONS" \
  SAMPLE_IDX="$SAMPLE_IDX" \
  SIDE_FILTER="$SIDE_FILTER" \
  TRUE_JOINTS_KEY="$BASELINE_TRUE_JOINTS_KEY" \
  EXCLUDE_CALIB_FROM_EVAL="$EXCLUDE_CALIB_FROM_EVAL" \
  CALIBRATION_NUM="$CALIBRATION_NUM" \
  PRED_BETA_SOURCE="$PRED_BETA_SOURCE" \
    scripts/run_all_finetune_baselines_mac.sh
fi

if [[ "$RUN_HYPERNET" == "1" ]]; then
  echo "=== Export HyperNet predictions ==="
  "$PYTHON_BIN" scripts/export_hypernet_finetune_predictions.py \
    --model-dir "$HYPERNET_MODEL_DIR" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-calib-actions "$K" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --sample-idx "$SAMPLE_IDX" \
    --max-sessions "$MAX_SESSIONS" \
    --side-filter "$SIDE_FILTER" \
    --true-key "$HYPERNET_TRUE_KEY" \
    --paired-target-key "$PAIRED_TARGET_KEY" \
    --metric-gt-key "$METRIC_GT_KEY" \
    --pred-beta-source "$PRED_BETA_SOURCE" \
    --variants hypernet \
    "${calibrate_args[@]}" \
    --canonicalizer-extractor-path "$CANONICALIZER_EXTRACTOR_PATH" \
    2>&1 | tee "$OUTPUT_DIR/logs/export_hypernet_k${K}.log"

  echo "=== Export HyperNet LoRA finetune predictions ==="
  "$PYTHON_BIN" scripts/export_hypernet_finetune_predictions.py \
    --model-dir "$HYPERNET_MODEL_DIR" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-calib-actions "$K" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --sample-idx "$SAMPLE_IDX" \
    --max-sessions "$MAX_SESSIONS" \
    --side-filter "$SIDE_FILTER" \
    --true-key "$HYPERNET_TRUE_KEY" \
    --paired-target-key "$PAIRED_TARGET_KEY" \
    --metric-gt-key "$METRIC_GT_KEY" \
    --pred-beta-source "$PRED_BETA_SOURCE" \
    --variants hypernet_lora_ft \
    "${calibrate_args[@]}" \
    --ft-steps "$STEPS" \
    --ft-lr "$LR" \
    --ft-weight-decay "$WEIGHT_DECAY" \
    --canonicalizer-extractor-path "$CANONICALIZER_EXTRACTOR_PATH" \
    2>&1 | tee "$OUTPUT_DIR/logs/export_hypernet_lora_ft_k${K}.log"
fi

if [[ "$RUN_SUMMARY" == "1" ]]; then
  echo "=== Summarize metrics by side ==="
  "$PYTHON_BIN" scripts/summarize_finetune_times.py \
    --output-dir "$OUTPUT_DIR" \
    --data-path "$DATA_PATH" \
    --k "$K" \
    --methods "${METHODS[@]}"
fi

if [[ "$RUN_VISUALIZE" == "1" ]]; then
  echo "=== Start Viser visualization ==="
  "$PYTHON_BIN" scripts/visualize_paired_poses_samples.py \
    "$DATA_PATH" \
    --model-path "$MODEL_DIR" \
    --finetune-output-dir "$OUTPUT_DIR" \
    --finetune-k "$K" \
    --finetune-methods "${METHODS[@]}" \
    --gt-key "$METRIC_GT_KEY" \
    --device "$DEVICE" \
    --port "$PORT" \
    "${calibrate_args[@]}"
fi

echo "Done."
echo "Metrics table: $OUTPUT_DIR/time_comparison_k${K}_by_side.csv"
echo "Predictions root: $OUTPUT_DIR"
