#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/combined_cnn_mlp_leaky_global_mixer_scheduler_composite_20260308_205854/best}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/hand_keypoint_data/Interhand_Val_0511/hdf5/InterHand_retargeted_30fps_biased_complex.hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/expdata/finetune_baselines_mac}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-200}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"
MAX_JOBS="${MAX_JOBS:-4}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
SAMPLE_IDX="${SAMPLE_IDX:--1}"

METHODS=(lora full_ft residual_linear last_layer)
PHASE1_KS=(16)
PHASE2_KS=(4 8)

mkdir -p "$LOG_DIR"

echo "Root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Steps: $STEPS"
echo "Max jobs: $MAX_JOBS"
echo "Max sessions: $MAX_SESSIONS (0 means all)"
echo "Sample idx: $SAMPLE_IDX (-1 means all biased variants)"

run_one() {
  local method="$1"
  local k="$2"
  local log_file="$LOG_DIR/${method}_k${k}.log"

  echo "[start] method=$method k=$k log=$log_file"
  "$PYTHON_BIN" scripts/run_finetune_baselines.py \
    --model-dir "$MODEL_DIR" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --method "$method" \
    --num-calib-actions "$k" \
    --steps "$STEPS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --sample-idx "$SAMPLE_IDX" \
    --max-sessions "$MAX_SESSIONS" \
    2>&1 | tee "$log_file"
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

run_phase "phase 1: K=16 sanity/full pass" "${PHASE1_KS[@]}"
run_phase "phase 2: K=4 and K=8 sweep" "${PHASE2_KS[@]}"

echo "All baseline runs completed."
echo "Metrics: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
