#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/combined_cnn_mlp_leaky_global_mixer_scheduler_composite_20260308_205854/best}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/hand_keypoint_data/Interhand_val/InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/expdata_annie/current_export_franklin_aligned_joint20cm_marked_right}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-200}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_USE_GATES="${LORA_USE_GATES:-1}"
LORA_GATE_INIT="${LORA_GATE_INIT:-0.0}"
SEED="${SEED:-42}"
MAX_JOBS="${MAX_JOBS:-4}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
SAMPLE_IDX="${SAMPLE_IDX:--1}"
SIDE_FILTER="${SIDE_FILTER:-right}"
TRUE_JOINTS_KEY="${TRUE_JOINTS_KEY:-world_hand_joints}"
EXCLUDE_CALIB_FROM_EVAL="${EXCLUDE_CALIB_FROM_EVAL:-0}"
PRED_BETA_SOURCE="${PRED_BETA_SOURCE:-auto}"

METHODS=(lora full_ft residual_linear last_layer)
KS=(4)

mkdir -p "$LOG_DIR"

echo "Root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Steps: $STEPS"
echo "LoRA rank: $LORA_RANK"
echo "LoRA alpha: $LORA_ALPHA"
echo "LoRA dropout: $LORA_DROPOUT"
echo "LoRA use gates: $LORA_USE_GATES"
echo "LoRA gate init: $LORA_GATE_INIT"
echo "Max jobs: $MAX_JOBS"
echo "Max sessions: $MAX_SESSIONS (0 means all)"
echo "Sample idx: $SAMPLE_IDX (-1 means all biased variants)"
echo "Side filter: $SIDE_FILTER"
echo "True joints key: $TRUE_JOINTS_KEY"
echo "Pred beta source: $PRED_BETA_SOURCE"

run_one() {
  local method="$1"
  local k="$2"
  local log_file="$LOG_DIR/${method}_k${k}.log"
  local lora_gate_args=()
  if [[ "$LORA_USE_GATES" == "1" ]]; then
    lora_gate_args=(--lora-use-gates)
  else
    lora_gate_args=(--no-lora-use-gates)
  fi

  echo "[start] method=$method k=$k log=$log_file"
  if [[ "$EXCLUDE_CALIB_FROM_EVAL" == "1" ]]; then
    "$PYTHON_BIN" scripts/run_finetune_baselines.py \
      --model-dir "$MODEL_DIR" \
      --data-path "$DATA_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --method "$method" \
      --num-calib-actions "$k" \
      --steps "$STEPS" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --lora-rank "$LORA_RANK" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-gate-init "$LORA_GATE_INIT" \
      "${lora_gate_args[@]}" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --sample-idx "$SAMPLE_IDX" \
      --max-sessions "$MAX_SESSIONS" \
      --side-filter "$SIDE_FILTER" \
      --true-joints-key "$TRUE_JOINTS_KEY" \
      --pred-beta-source "$PRED_BETA_SOURCE" \
      --exclude-calib-from-eval \
      2>&1 | tee "$log_file"
  else
    "$PYTHON_BIN" scripts/run_finetune_baselines.py \
      --model-dir "$MODEL_DIR" \
      --data-path "$DATA_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --method "$method" \
      --num-calib-actions "$k" \
      --steps "$STEPS" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --lora-rank "$LORA_RANK" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-gate-init "$LORA_GATE_INIT" \
      "${lora_gate_args[@]}" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --sample-idx "$SAMPLE_IDX" \
      --max-sessions "$MAX_SESSIONS" \
      --side-filter "$SIDE_FILTER" \
      --true-joints-key "$TRUE_JOINTS_KEY" \
      --pred-beta-source "$PRED_BETA_SOURCE" \
      2>&1 | tee "$log_file"
  fi
  echo "[done] method=$method k=$k"
}

wait_for_slot() {
  while (( "$(jobs -pr | wc -l | tr -d ' ')" >= MAX_JOBS )); do
    sleep 2
  done
}

run_phase() {
  local phase_name="$1"
  shift
  local ks=("$@")

  echo "=== $phase_name ==="
  for k in "${ks[@]}"; do
    for method in "${METHODS[@]}"; do
      wait_for_slot
      run_one "$method" "$k" &
    done
  done
  wait
  echo "=== $phase_name complete ==="
}

run_phase "K=4 finetune baselines" "${KS[@]}"

"$PYTHON_BIN" scripts/summarize_finetune_times.py \
  --output-dir "$OUTPUT_DIR" \
  --k 4 \
  --data-path "$DATA_PATH" \
  --methods "${METHODS[@]}"

echo "All baseline runs completed."
echo "Metrics: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
