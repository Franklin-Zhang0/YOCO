# Environment Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.8.22 or
newer. The locked release environment targets Linux x86_64 with Python 3.10.12.
uv downloads the matching Python runtime when needed; no separate Conda
environment or manual dependency installation is required.

If repository access is restricted, authenticate with GitHub before cloning
(HTTPS credentials or SSH). On clusters with restricted internet access, configure
the standard `HTTP_PROXY` and `HTTPS_PROXY` variables before setup and downloads.
The lockfile and asset manifest should remain unchanged.

For NGC PyTorch images, remove inherited system-PyTorch directories from
`LD_LIBRARY_PATH` before using this environment; otherwise Python 3.10 can load
incompatible libraries from the image's Python 3.12 installation. Preserve
NVIDIA driver library paths. A Python virtual environment does not isolate
these native-library search paths.

For the tested NGC PyTorch 25.06 image, keep only its driver-library paths:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

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

The CUDA lock pins `nvidia-cudnn-cu12==9.19.0.56` for the fine-tuning baseline
accuracy contract. Keep this pin: PyTorch 2.6.0's default cuDNN 9.1 dependency
does not pass every baseline reference. It does not add CUDA packages to the
CPU environment.

Both locked profiles have been execution-tested in a clean one-H100 CCI,
including package isolation, model loading, and the Python and CLI inference
paths. The paper's fine-tuning baselines and timing measurements additionally
require the H100 protocol in the reproduction guide. Inference checks alone
do not certify paper accuracy or adaptation times. Tested runtimes and scoped
receipts are recorded in the [CCI qualification](reproduction/cci_release_results.json).

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
glove, or `DEX_URDF_PATH` is needed for offline use. The CCI qualification lists
the tested MANO file fingerprints for comparison with your licensed copy.

Return to the [README](README.md) to download pretrained models, or follow the
[inference](docs/inference.md) and [reproduction](reproduction/README.md) guides.
