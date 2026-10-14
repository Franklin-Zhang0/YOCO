# Model Inference

The local checkpoint bundle provides YOCO and its frozen MANO canonicalizer.
Use trusted checkpoint files and set `MANO_PATH` to the absolute path of your
separately licensed MANO model directory before starting Python. Inference needs
neither ROS nor `DEX_URDF_PATH`.

## Python

```python
from calibration_net.yoco import YOCO

model = YOCO.from_pretrained(
    "assets/models/yoco",
    canonicalizer_dir="assets/models/frozen_estimator",
    side="right",
    device="cpu",
)
model.calibrate(raw_pairs, target_pairs)
result = model.predict(raw_frames, betas=betas)
theta, joints = result.theta, result.joints
```

`raw_pairs` and `target_pairs` both have shape `(4, 21, 3)` for the released
checkpoint. Each raw pose is paired with accurate reference keypoints from the
same pose. Inputs are in meters and follow MANO's 21-joint order: wrist, then
thumb, index, middle, ring, and pinky, each from the base joint to the fingertip.
Keep the same hand side and ordering across calibration and prediction.

`raw_frames` is either `(21, 3)` or `(N, 21, 3)`. `betas` is optional: omit it
for neutral MANO shape, pass `(10,)` for a shared shape, or `(N, 10)` for a shape
per frame. This controls output joint reconstruction, not calibration inputs.

`theta` contains 45 MANO axis-angle pose values in radians, without global
orientation. `joints` contains 21 wrist-origin, palm-aligned keypoints in meters,
not world coordinates. Single inputs return `(45,)` and `(21, 3)`; batched inputs
return `(N, 45)` and `(N, 21, 3)`.

Calling `calibrate` again starts an independent session. Normal inference
disables conditioning dropout and removes the previous adapter before deriving
the next one. This intentionally differs from legacy paper evaluation behavior;
use the reproduction scripts for table comparisons.

Normal CUDA calibration and prediction use full FP32 arithmetic, with TF32 and
CUDA autocast disabled only during each API call for consistent batching. Caller
precision settings are restored afterward, including on errors; CPU inference is
unchanged. Legacy reproduction scripts retain their original precision behavior.

## Local Example

Download the evaluation data as well as the models by running
`scripts/download_reproduction_assets.py` without `--models-only`. Then run a
short example on a right-hand AVP recording:

```bash
uv run --frozen --extra cpu python scripts/run_yoco.py \
  --data-path assets/datasets/avp/aligned_avp_glove_val.hdf5 \
  --group 1 --side right --output yoco_predictions.npz
```

The example chooses four valid calibration poses from the first 200 frames,
then predicts the next 32 frames. The output records `theta`, `joints`,
`calib_idx`, and `frame_idx`. It is an inference example, not an accuracy benchmark.
Use `--device cuda` with a CUDA installation or `--frames` to change its length.
The output path must not already exist.

For your own observations, provide an NPZ file with `raw_pairs`, `target_pairs`,
`raw_frames`, and optional `betas` arrays using the shapes above:

```bash
uv run --frozen --extra cpu python scripts/run_yoco.py --input observations.npz --output predictions.npz
```

The NPZ input does not need dataset-specific HDF5 fields. Use `--side left` for
left hands. `uv run --frozen --extra cpu python scripts/run_yoco.py --help` lists all options.
