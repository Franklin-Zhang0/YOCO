# Reproduction Protocol

The complete release experiment is implemented in
[`scripts/reproduce_paper.py`](../scripts/reproduce_paper.py). It exports all
three seeds and then verifies their saved predictions. No training of the YOCO
checkpoint is needed. Follow the repository's environment and asset setup first.

## Entry Point

The runner defaults to both tables, YOCO only, CPU, and seeded legacy
conditioning. `--table 1` selects InterHand; `--table 2` selects AVP/glove.
`--methods all` additionally evaluates the five original paper baselines.
Individual methods are `yoco`, `ridge`, `last_layer`, `full_ft`, `lora`, and
`no_calibration`. Run `scripts/reproduce_paper.py --help` for all options.

After setup, evaluate the release checkpoint on both datasets:

```bash
uv run --frozen --extra cpu python scripts/reproduce_paper.py --table all
```

Select one table and an explicit fresh result directory as needed:

```bash
uv run --frozen --extra cpu python scripts/reproduce_paper.py --table 2 --output-dir eval/results/table2
```

For the original fine-tuning baselines, use the CUDA environment on an H100:

```bash
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
uv run --frozen --extra cu124 python scripts/reproduce_paper.py --table all --methods all --device cuda --conditioning-mode eval --output-dir eval/results/all-methods
```

Use a fresh `--output-dir` for every experiment. Existing directories, including
empty ones, are rejected. The runner checks each required model/configuration/
normalization/data file against the immutable release manifest before launching.
It never modifies source assets. `--dry-run` prints the commands without loading
local assets or running evaluations; it is not reproduction evidence.

The standard asset root is `assets/`; override it with `--asset-root`.
The pinned download entrypoint is
[`scripts/download_reproduction_assets.py`](../scripts/download_reproduction_assets.py).
The model bundles and prepared datasets are publicly downloadable on Hugging
Face; authentication with `HF_TOKEN` is optional. The collection remains private
for now. Download and verify the assets before running experiments.
`MANO_PATH` must contain your separately licensed `MANO_LEFT.pkl` and
`MANO_RIGHT.pkl`; neither is redistributed.

## Checkpoint And Data Contracts

The release provides one depth-4 YOCO checkpoint for both tables: K=4,
LoRA rank/alpha 32/32, and eight flow steps. The frozen estimator is the
canonicalizer and initialization for every baseline. Each model bundle requires
`model.pth`, `config.yaml`, and `norm_stats.pkl`, with their saved architecture
and normalization unchanged. Only load trusted pickle-based assets. The bundle
contains evaluation data and pretrained models, not the full training corpus.

| Asset under `assets/` | Purpose |
| --- | --- |
| `models/yoco/` | Release YOCO checkpoint for both tables |
| `models/frozen_estimator/` | Canonicalizer and baseline initialization |
| `datasets/interhand/InterHand_retargeted_30fps_biased_super_mano_fk20cm_bad_joint_marked.hdf5` | Table 1 prepared validation set |
| `datasets/avp/aligned_avp_glove_val.hdf5` | Table 2 three-recording validation cohort |
| `datasets/avp/aligned_avp_glove_train_new.hdf5` | Additional recording for legacy Table 2 baselines |

Raw InterHand downloads and the unmarked `InterHand_retargeted_30fps_biased_super.hdf5`
are not substitutes for the prepared validation set. Preserve group names,
augmentation order, the 21-joint ordering, group MANO betas, `gt_joint_mask`, and
`bad_frame` annotations. `--canonicalizer-extractor-path` relocates the saved
canonicalizer path in memory; checkpoint files are not edited.

## Sampling And Cohorts

| Setting | Evaluation units | Evaluation frames per seed |
| --- | --- | ---: |
| Table 1, all methods | 20 right-hand bias variants | 216,340 |
| Table 2, YOCO/Ridge/last-layer/full fine-tuning | 3 right-hand recordings | 4,128 |
| Table 2, no calibration/LoRA fine-tuning | 4-recording baseline cohort | 5,315 |

Every run uses seeds 0, 1, and 2. Each unit samples K=4 calibration frames
from `[0,200)` and evaluates all frames in `[200,T)`, applying `gt_joint_mask`
to metric joints. `bad_frame` restricts calibration candidates, not held-out
frames. The calibration RNG is keyed by seed, HDF5 group name, augmentation
index, and K. The runner does not truncate sessions or choose a single
augmentation.

When no calibration or LoRA is requested for Table 2, the runner invokes
[`scripts/prepare_table2_legacy_data.py`](../scripts/prepare_table2_legacy_data.py).
It copies validation groups `1,2,3` and `train_new` group `0` into output groups
`0,1,2,3`, respectively, under the new result directory. Renaming these groups
changes the calibration RNG and therefore the experiment. The source files
remain untouched. This method-specific cohort follows the original baseline
protocol; it is not a uniform-cohort comparison.

## Conditioning And Baselines

YOCO sets both calibration and inference seeds explicitly. The latter seeds
Python, NumPy, and PyTorch before model initialization and conditioning.
`--conditioning-mode legacy` retains the historical exporter's first-unit
dropout and sequential LoRA state. `--conditioning-mode eval` disables
conditioning dropout and selects the separate evaluation-mode references.
The saved prediction metadata records the mode and inference seed. Do not mix
the two modes in an output directory.

Use `legacy` on CPU for the CPU release references, including Table 2's rounded
MPJPE of 17.75 mm. Use `eval` for YOCO on CUDA and its evaluation-mode references.
Both modes use the same release checkpoint. Conditioning scores are
device-dependent; CPU legacy references do not certify a CUDA legacy run.
Conditioning mode changes only YOCO, not the five baseline protocols, and the
two modes are not the same experiment.

Ridge uses an identity prior in MANO-theta space, an unpenalized bias, and
regularization `1e-3`. The fine-tuning baselines use 200 steps, learning rate
`1e-4`, no weight decay, LoRA rank/alpha 8/8, dropout 0.1, and gates initialized
at 0. All baselines start from the frozen estimator, not from YOCO.

CPU is the reference runtime for the release YOCO accuracy checks. The original
fine-tuning baseline references were validated on an H100 with CUDA; CPU dropout
RNG can yield different scores. Select `--device cuda` for those comparisons.
GPU reproduction still has to pass verification; selecting CUDA by itself does
not certify an exact hardware/accuracy match. The tested NGC PyTorch 25.06
runtime uses the locked cuDNN 9.19.0.56 dependency,
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, matmul precision `high`, and TF32 enabled
for matmul and cuDNN. Preserve both the lockfile and the explicit precision
setting.
Keep this paper-export profile separate from the normal inference API's
temporary FP32 guard. The paper's adaptation-time measurements require one
H100 and matching unit counts. CPU timings are not comparable. Offline pose
evaluation does not need ROS, a glove, a robot URDF, or `DEX_URDF_PATH`.

## Verification

Each export writes `seed_<N>/<method>/k4/metrics.json` and `predictions.hdf5`.
The runner automatically invokes
[`scripts/verify_paper_reproduction.py`](../scripts/verify_paper_reproduction.py),
which checks complete units and seeds, calibration/evaluation indices, masked
joints, source hashes, and metrics recomputed from saved predictions.
Verification exits nonzero on a mismatch; aggregate accuracy tolerances are at
most 0.05 mm and 0.0005 AUCJ. The separate saved-versus-recomputed pose metric
check allows float32 reduction roundoff (`8 * float32 epsilon` relative,
`1e-5` mm absolute); this does not change the paper accuracy thresholds.
Hardware timing is outside its accuracy certificate.

Fine-tuning exports with saved predictions compute pose receipts from those
same held-out joints and the canonical NumPy ground-truth transform. Their
`paper_pose_metric_source` is `saved_predictions`; auxiliary losses and timings
retain the historical held-out pass and are not part of the pose certificate.

YOCO uses the measured `release` reference profile; baselines use the original
`paper` profile. The expected release values for both conditioning modes are
stored in [`release_reference.json`](release_reference.json). Method-specific
receipts are `verification.json` for YOCO and `verification-<method>.json` for
baselines. The runner additionally records its command plan, verified assets,
completed steps, and success/failure in `reproduction_run.json`. A launch,
dry run, or incomplete receipt is not a successful reproduction.

Pose and AUCJ metrics are averaged over units within each seed, then over seeds.
Table 2 baseline pinch scores are weighted by valid pinch-event counts within
each seed; YOCO uses unit means. Pooling frames across recordings gives a
different result.

| Metric | Definition |
| --- | --- |
| MPJPE | Unaligned mean joint error, mm |
| PA-MPJPE | Procrustes-aligned mean joint error, mm |
| `tip_mpjpe_mm` | Unaligned error at joints 4, 8, 12, 16, and 20, mm |
| AUCJ | Normalized PCK area over 0-50 mm, using 101 thresholds |
| Pinch | Predicted thumb-to-fingertip distance on valid GT pinches below 10 mm |

**Paper correction:** "Tip-PA" in the main paper is a typo for
**`tip_mpjpe_mm`**, not a Procrustes-aligned fingertip metric. Pinch is not the
absolute predicted-versus-GT distance error.

Passing the `release` profile certifies this model's measured results for the
selected conditioning mode and cohort. CPU legacy conditioning matches
Table 2's rounded MPJPE (17.75 mm), **not the full original YOCO row**. The
accuracy checks do not certify exact reproduction of every original paper
number or its hardware timings.

Release qualification and source/asset fingerprints are recorded in
[`release_results.json`](release_results.json),
[`uv_release_results.json`](uv_release_results.json), and
[`cci_release_results.json`](cci_release_results.json). Raw validation logs and
predictions are not part of the repository; running the scripts generates fresh
receipts.

## Other Experiments

Pair-count experiments require the checkpoint trained for each K in
`1,2,4,8,16`; changing K on the K=4 model is not the same experiment. Other
learned calibration methods require their own checkpoints. Table 3 requires
physical robot trials and cannot be reproduced by offline pose export.
