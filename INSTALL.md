# Environment Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.8.22 or
newer. The locked release environment targets Linux x86_64 with Python 3.10.12.
uv downloads the matching Python runtime when needed; no separate Conda
environment or manual dependency installation is required.

## CPU

```bash
git clone --branch release/yoco-website https://github.com/Franklin-Zhang0/YOCO.git
cd YOCO
uv sync --frozen --extra cpu
```

This creates `.venv/` from `uv.lock`, including the editable YOCO package,
PyTorch 2.6.0, torchvision 0.21.0, NumPy 1.23.5, and the pinned MANO implementation.
The NumPy pin preserves compatibility with upstream chumpy. Its build dependencies
are managed by uv, and headless OpenCV avoids an OpenGL system dependency.

Run commands with `uv run --frozen --extra cpu`. Keep the same extra on every
invocation so uv retains the selected PyTorch build. There is no need to activate
the environment or set `PYTHONPATH`.

## CUDA

Use a compatible NVIDIA driver and the CUDA 12.4 profile:

```bash
uv sync --frozen --extra cu124
uv run --frozen --extra cu124 python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The CPU and CUDA extras are mutually exclusive. Use `--extra cu124` on subsequent
commands and pass `--device cuda` to inference or evaluation scripts. Requesting
CUDA on an unavailable device fails explicitly.

CPU accuracy checks have been validated in the clean locked environment. The
CUDA dependency profile is resolved and locked, but has not been execution-tested
in this release environment. The paper's fine-tuning baselines and timing
measurements additionally require the H100 protocol in the reproduction guide.

## MANO Assets

Obtain the model files from [MANO](https://mano.is.tue.mpg.de/) under its
[license](https://mano.is.tue.mpg.de/license.html). They are not redistributed:

```text
mano_models/
  MANO_LEFT.pkl
  MANO_RIGHT.pkl
```

```bash
export MANO_PATH=/absolute/path/to/mano_models
```

Use this environment variable for inference and reproduction commands. No ROS,
glove, or `DEX_URDF_PATH` is needed for offline use.

Return to the [README](README.md) to download pretrained models, or follow the
[inference](docs/inference.md) and [reproduction](reproduction/README.md) guides.
