# YOCO: You Only Calibrate Once!

**Fast Mocap Calibration for Dexterous Teleoperation, CoRL 2026**

Yu Zhang, Yunqi Li, Yushi Du, Yi Ma, Yanchao Yang

[Paper](docs/assets/papers/yoco.pdf) | [Project website](https://franklin-zhang0.github.io/YOCO/)

YOCO generates session-specific LoRA updates for a frozen MANO hand-pose
estimator from a few paired raw/reference poses, without test-time fine-tuning.
This repository provides model inference, calibration baselines, and offline
paper-result reproduction tools.

> **Paper correction:** "Tip-PA" in the main paper is a typo. The column should
> be **`tip_mpjpe_mm`**, the unaligned fingertip MPJPE in millimeters, not
> Procrustes-aligned fingertip error.

## Setup

Clone the release branch and use Python 3.10 with NumPy 1.23.5 for the legacy
upstream MANO/chumpy stack:

```bash
git clone --branch release/yoco-website https://github.com/Franklin-Zhang0/YOCO.git
cd YOCO
conda create -n yoco python=3.10 -y
conda activate yoco
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . 'numpy==1.23.5' 'scipy==1.10.1' pyyaml einops \
  'opencv-python-headless==4.8.1.78' 'huggingface_hub==2.2.0'
git clone https://github.com/hassony2/manopth.git
git -C manopth checkout 4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b
python -m pip install -e ./manopth
python -m pip install chumpy==0.70 --no-build-isolation
```

Obtain `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` under the
[MANO license](https://mano.is.tue.mpg.de/license.html), and place them in one directory:

```bash
export MANO_PATH=/absolute/path/to/mano_models
test -f "$MANO_PATH/MANO_LEFT.pkl"
test -f "$MANO_PATH/MANO_RIGHT.pkl"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export DEVICE=cpu
python -c 'import torch, manopth, h5py; print(torch.__version__)'
```

CPU is the reference runtime for the release checkpoint's accuracy checks.
For GPU evaluation, install the same PyTorch versions using the
`https://download.pytorch.org/whl/cu124` index, set `DEVICE=cuda`, and check
`torch.cuda.is_available()`. Headless OpenCV avoids an OpenGL system-library
dependency. The paper's adaptation-time measurements require one H100; CPU
timings are not comparable.
Offline reproduction does not require ROS, a glove, a robot URDF, or
`DEX_URDF_PATH`.

The repository is named YOCO; the Python package remains `calibration_net`.

## Download Models And Data

Model directories must contain **all three** files: `model.pth`, `config.yaml`,
and `norm_stats.pkl`. Keep the saved architecture and normalization unchanged.
Only load trusted checkpoints: the model and MANO files use Python pickle.

| Asset | Used For |
| --- | --- |
| `models/yoco/` | June 3 YOCO checkpoint for both tables: K=4, DiT depth 4, LoRA rank/alpha 32/32, 8 flow steps |
| `models/frozen_estimator/` | May 18 frozen estimator, canonicalizer, and baseline initialization |
| `models/historical_table1/` (optional) | May 26 depth-2, rank/alpha-8 YOCO checkpoint for the original Table 1 row |
| `InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5` | Table 1 prepared validation set, including 20 bias variants and masks |
| `aligned_avp_glove_val.hdf5` | Table 2 paired AVP/glove validation recordings |
| `aligned_avp_glove_train_new.hdf5` | Fourth recording for Table 2's legacy no-calibration/LoRA cohort |
| `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` | Hand geometry and forward kinematics |

The [YOCO reproduction collection](https://huggingface.co/collections/Franklin0/yoco-reproduction-6acf4f2f1580be300065ce25)
contains the [model repository](https://huggingface.co/Franklin0/YOCO) and
[dataset repository](https://huggingface.co/datasets/Franklin0/YOCO-Reproduction)
are **private for now**. Sign in with a Hugging Face account/token that has
access to both repositories; an unauthenticated request may return 404.

```bash
export HF_ENDPOINT=https://huggingface.co
python -c 'from huggingface_hub import login; login()'
python scripts/download_reproduction_assets.py --output-dir "$PWD/assets"
```

Alternatively, provide `HF_TOKEN` through your environment. The downloader pins
immutable Hub commits from the [release manifest](reproduction/huggingface_release.json)
on the official `huggingface.co` endpoint, and verifies each file's size and
SHA256 before use. Existing mismatched files
are not overwritten. Add `--include-historical` to also download the original
Table 1 checkpoint. The default payload is approximately 1.39 GB; the optional
checkpoint adds 0.36 GB.

MANO model files are not redistributed. The prepared InterHand derivative
retains the upstream [CC-BY-NC 4.0 terms and attribution](https://github.com/facebookresearch/InterHand2.6M/blob/main/LICENSE).
See the dataset card for preparation details and source attribution. This bundle
contains evaluation data and checkpoints, not the full training corpus.

Set absolute paths to your assets:

```bash
export ASSET_ROOT="$PWD/assets"
export BASE_MODEL_DIR="$ASSET_ROOT/models/frozen_estimator"
export YOCO_MODEL_DIR="$ASSET_ROOT/models/yoco"
export SYNTHETIC_DATA="$ASSET_ROOT/datasets/interhand/InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5"
export REAL_DATA="$ASSET_ROOT/datasets/avp/aligned_avp_glove_val.hdf5"
export REAL_TRAIN_NEW_DATA="$ASSET_ROOT/datasets/avp/aligned_avp_glove_train_new.hdf5"
export OUTPUT_ROOT="$PWD/eval/results/reproduction"
```

Raw InterHand downloads and the unmarked
`InterHand_retargeted_30fps_biased_super.hdf5` do not replace the prepared
validation set. No model training is needed for checkpoint evaluation; the
`train_new` file is used only to recover Table 2's older four-recording cohort.
`--canonicalizer-extractor-path` relocates the canonicalizer's saved path in
memory; it does not edit checkpoint files.

## Reproduction Protocol

| Setting | Calibration | Evaluation Units | Seeds |
| --- | --- | --- | --- |
| Table 1: augmented InterHand2.6M | K=4, sampled from frames `[0,200)` | 20 right-hand bias variants, frames `[200,T)` | 0, 1, 2 |
| Table 2: YOCO, Ridge, last-layer, full fine-tuning | K=4, sampled from frames `[0,200)` | 3 right-hand validation recordings, frames `[200,T)` | 0, 1, 2 |
| Table 2: no calibration, LoRA fine-tuning | K=4, sampled from frames `[0,200)` | Legacy 4-recording cohort, frames `[200,T)` | 0, 1, 2 |

Preserve HDF5 group names, augmentation order, the 21-joint ordering, group
MANO betas, and `gt_joint_mask`/`bad_frame` annotations. Calibration sampling is
deterministic and keyed by seed, group name, augmentation index, and K.
Average recording metrics within each seed, then average the three seeds.
Pooling frames across recordings gives a different result. Table 2 baseline
pinch scores are weighted by valid pinch-event counts within each seed; the
YOCO pinch score uses recording means. These method-specific conventions
reproduce the published table, but are not a uniform-cohort comparison.

Do not limit sessions or select a single augmentation for the final run. A
single-seed or reduced-data run is only a smoke test. The later five-recording
shared-calibration dataset uses a different protocol and is not Table 2.

Choose the setting before running the commands below:

```bash
# Table 1
export SETTING=interhand
export DATA_PATH="$SYNTHETIC_DATA"

# Table 2: select these instead. The same YOCO checkpoint is used.
# export SETTING=avp
# export DATA_PATH="$REAL_DATA"
```

Use a fresh output directory for each experiment to avoid mixing artifacts.

For Table 2, prepare the legacy cohort before running baselines. This copies
validation groups `1,2,3` and `train_new` group `0` into groups `0,1,2,3`,
respectively, without changing any input files. Group names affect the RNG:

```bash
export LEGACY_REAL_DATA="$OUTPUT_ROOT/assets/avp_glove_table2_legacy.hdf5"
python scripts/prepare_table2_legacy_data.py \
  --val-data "$REAL_DATA" --train-new-data "$REAL_TRAIN_NEW_DATA" \
  --output "$LEGACY_REAL_DATA"
```

## Run YOCO

Run from the repository root:

```bash
set -euo pipefail
for seed in 0 1 2; do
  python scripts/export_hypernet_finetune_predictions.py \
    --model-dir "$YOCO_MODEL_DIR" \
    --canonicalizer-extractor-path "$BASE_MODEL_DIR" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_ROOT/$SETTING/seed_$seed" \
    --variants hypernet --legacy-conditioning --inference-seed "$seed" \
    --num-calib-actions 4 --calibration-num 200 \
    --calib-selection random --seed "$seed" --sample-idx -1 \
    --side-filter right --input-key world_hand_joints_biased \
    --true-key world_hand_joints_true --paired-target-key world_hand_joints \
    --metric-gt-key world_hand_joints --pred-beta-source group \
    --device "$DEVICE"
done
```

`--seed` controls calibration-frame selection. `--inference-seed` independently
seeds Python, NumPy, and PyTorch before model initialization and conditioning.
Both must be set for the repeatable release recipe. `--legacy-conditioning`
retains the historical exporter's first-unit dropout and sequential LoRA state.
Omit it for evaluation-mode conditioning with dropout disabled, and verify
against `--conditioning-mode eval` instead. Do not mix the two modes in one
output directory. Both mode and inference seed are saved with the predictions.

## Run Baselines

All baselines start from the frozen estimator, not the YOCO checkpoint.
The published fine-tuning rows were validated on an H100 using the CUDA setup;
CPU fine-tuning can differ because its dropout RNG is different. Use that GPU
setup and `DEVICE=cuda` when checking those rows.
Use 200 optimization steps, learning rate `1e-4`, no weight decay, and the
same calibration frames and evaluation split:

```bash
for seed in 0 1 2; do
  for method in lora full_ft last_layer; do
    method_data="$DATA_PATH"
    if [ "$SETTING" = avp ] && [ "$method" = lora ]; then
      method_data="${LEGACY_REAL_DATA:?Prepare the Table 2 legacy cohort first}"
    fi
    python scripts/run_finetune_baselines.py \
      --model-dir "$BASE_MODEL_DIR" --data-path "$method_data" \
      --output-dir "$OUTPUT_ROOT/$SETTING/seed_$seed" --method "$method" \
      --num-calib-actions 4 --calibration-num 200 --seed "$seed" \
      --sample-idx -1 --side-filter right \
      --true-joints-key world_hand_joints --pred-beta-source group \
      --steps 200 --lr 1e-4 --weight-decay 0.0 \
      --lora-rank 8 --lora-alpha 8 --lora-dropout 0.1 \
      --lora-use-gates --lora-gate-init 0.0 --device "$DEVICE"
  done
done
```

Ridge fits a linear correction in MANO-theta space with an identity prior,
an unpenalized bias, and regularization `1e-3`. It uses a closed-form solve:

```bash
for seed in 0 1 2; do
  python scripts/run_finetune_baselines.py \
    --model-dir "$BASE_MODEL_DIR" --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_ROOT/$SETTING/seed_$seed" --method ridge \
    --num-calib-actions 4 --calibration-num 200 --seed "$seed" \
    --sample-idx -1 --side-filter right \
    --true-joints-key world_hand_joints --pred-beta-source group \
    --ridge-alpha 1e-3 --device "$DEVICE"
done
```

Export the no-calibration baseline:

```bash
for seed in 0 1 2; do
  method_data="$DATA_PATH"
  if [ "$SETTING" = avp ]; then
    method_data="${LEGACY_REAL_DATA:?Prepare the Table 2 legacy cohort first}"
  fi
  python scripts/export_best_model_predictions.py \
    --model-dir "$BASE_MODEL_DIR" --data-path "$method_data" \
    --output-dir "$OUTPUT_ROOT/$SETTING/seed_$seed" \
    --variants best_world_hand_joints_biased \
    --num-calib-actions 4 --calibration-num 200 --seed "$seed" \
    --sample-idx -1 --side-filter right \
    --gt-key world_hand_joints --pred-beta-source group --device "$DEVICE"
done
```

## Verify Results

Each method writes `seed_<N>/<method>/k4/metrics.json` and `predictions.hdf5`.
The latter retains all frame predictions plus calibration/evaluation indices.
Verify the selected release checkpoint's five accuracy columns:

```bash
python scripts/verify_paper_reproduction.py \
  --reference-profile release --conditioning-mode legacy \
  --output-root "$OUTPUT_ROOT/$SETTING" --setting "$SETTING" \
  --data-path "$DATA_PATH" --model-dir "$YOCO_MODEL_DIR" \
  --base-model-dir "$BASE_MODEL_DIR"
```

Verify all baseline rows:

```bash
set -euo pipefail
for method in lora full_ft last_layer ridge best_world_hand_joints_biased; do
  method_data="$DATA_PATH"
  if [ "$SETTING" = avp ] && \
    { [ "$method" = lora ] || [ "$method" = best_world_hand_joints_biased ]; }; then
    method_data="${LEGACY_REAL_DATA:?Prepare the Table 2 legacy cohort first}"
  fi
  python scripts/verify_paper_reproduction.py \
    --output-root "$OUTPUT_ROOT/$SETTING" --setting "$SETTING" \
    --method "$method" --data-path "$method_data" \
    --model-dir "$BASE_MODEL_DIR" --base-model-dir "$BASE_MODEL_DIR"
done
```

The verifier checks completeness, the calibration RNG, evaluation-frame arrays,
and joint masks, and recomputes metrics from saved predictions. It writes
`verification.json` or `verification-<method>.json` with model/data hashes and
runtime metadata, and exits nonzero on a mismatch. Default numerical tolerances
are 0.05 mm and 0.0005 AUCJ. They do not certify identical timing.
When historical exports are available, add `--reference-root /path/to/archive`
to also compare calibration indices, frame counts, and masked-joint counts.

| Metric | Definition |
| --- | --- |
| MPJPE | Unaligned mean joint error, mm |
| PA-MPJPE | Procrustes-aligned mean joint error, mm |
| `tip_mpjpe_mm` | Unaligned error at joints 4, 8, 12, 16, 20, mm |
| AUCJ | Normalized PCK area over 0-50 mm, 101 thresholds |
| Pinch | Predicted thumb-to-fingertip distance on valid GT pinches below 10 mm |

The pinch column is **not** absolute predicted-versus-GT distance error.
Adaptation/conditioning time is summed across evaluated units per seed, then
averaged across seeds; use the same hardware and unit count for comparisons.

Release YOCO expected values using the June 3 checkpoint and the seeded legacy
CPU recipe above:

| Setting | MPJPE | PA-MPJPE | `tip_mpjpe_mm` | AUCJ | Pinch (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Table 1 / InterHand | 15.34655 | 10.17906 | 22.58336 | 0.706600 | 34.63035 |
| Table 2 / AVP-glove | 17.75466 | 13.28029 | 31.18287 | 0.656464 | 41.63665 |

The release model improves all five Table 1 accuracy means. For Table 2, it
matches the paper's **rounded MPJPE only** (17.75 mm), not the full published
YOCO row. The original Table 2 evaluation-time checkpoint is unavailable; later
weights from that run are not a substitute. The `release` verification profile
checks the measured values above and the exact model, configuration,
normalization, dataset, mode, and inference-seed identities. It does not certify
an exact reproduction of the original Table 2 row.

Original Table 1 targets (baseline verification uses the `paper` profile):

| Method | MPJPE | PA-MPJPE | `tip_mpjpe_mm` | AUCJ | Pinch (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Calibration | 21.52 | 16.55 | 37.15 | 0.611 | 48.49 |
| Ridge | 17.62 | 13.00 | 28.28 | 0.676 | 45.35 |
| Last-layer-Finetune | 16.62 | 12.01 | 25.92 | 0.692 | 39.18 |
| Full-Finetune | 18.11 | 12.77 | 29.21 | 0.668 | 37.11 |
| LoRA-Finetune | 16.38 | 11.67 | 25.33 | 0.694 | 37.28 |
| YOCO (historical Table 1 model) | 15.64 | 10.45 | 23.29 | 0.702 | 34.90 |

Table 2 published targets, using the method-specific cohorts above:

| Method | MPJPE | PA-MPJPE | `tip_mpjpe_mm` | AUCJ | Pinch (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Calibration | 20.49 | 14.63 | 36.22 | 0.610 | 51.39 |
| Ridge | 24.71 | 19.21 | 45.50 | 0.573 | 57.54 |
| Last-layer-Finetune | 18.86 | 14.67 | 33.88 | 0.643 | 43.63 |
| Full-Finetune | 20.00 | 15.73 | 36.67 | 0.630 | 49.89 |
| LoRA-Finetune | 18.72 | 13.94 | 33.23 | 0.648 | 54.01 |
| YOCO (original snapshot required) | 17.75 | 13.14 | 31.73 | 0.657 | 42.79 |

Machine-readable reproduction evidence is kept separately under
[`reproduction/`](reproduction/), including the
[release checkpoint and Hub download checks](reproduction/release_results.json),
and historical
[Table 1](reproduction/table1_results.json) and
[Table 2](reproduction/table2_results.json) verification.

To check the original Table 1 YOCO row, download with `--include-historical`,
select `YOCO_MODEL_DIR="$ASSET_ROOT/models/historical_table1"`, use a fresh
output directory, and verify with `--reference-profile paper` instead of
`release`. Historical conditioning had unseeded first-unit dropout; it can vary
between runs. Do not use this checkpoint to certify the original Table 2 row.

## Other Experiments

For the pair-count experiment, use the checkpoint trained for each K in
`1, 2, 4, 8, 16`. Changing K on a K=4 checkpoint is not the same experiment.
Other learned calibration methods require their own checkpoints. Table 3
requires physical robot trials and cannot be reproduced by offline pose export.

Source layout: `calibration_net/policy/` contains models and inference,
`scripts/` contains reproduction entrypoints, `config/` contains training
configuration, and `calibration_net/teleop/` contains the hardware runtime.
