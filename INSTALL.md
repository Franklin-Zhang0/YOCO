# Installation Instructions
## Conda Environment Setup
```bash
git clone --branch release/yoco-website https://github.com/Franklin-Zhang0/YOCO.git
cd YOCO
git clone https://github.com/hassony2/manopth.git
```

Use Python 3.10 because older `manopth/chumpy` dependencies are not compatible
with Python 3.11.

```bash
conda create -n teleop python=3.10 -y
conda activate teleop

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
cd manopth
pip install -e .
cd ..
pip install h5py tyro opencv-python
pip install chumpy --no-build-isolation
```

Pin NumPy to `1.23.5` because `chumpy` still imports old NumPy aliases such as
`numpy.int`, `numpy.float`, and `numpy.bool`.

```bash
pip install --force-reinstall "numpy==1.23.5"
```

## Download MANO Model Files

Download the MANO model files according to the MANO license and place both files in the same directory:

```text
mano_models/
|- MANO_LEFT.pkl
|- MANO_RIGHT.pkl
```
Export `MANO_PATH` to the directory that contains those two files:

```bash
export MANO_PATH=/path/to/mano_models
```
