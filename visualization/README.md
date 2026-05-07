## Example Usage

### Keypoint reconstruction viewer
```bash
python \
  visualization/visualize_keypoint_recon.py \
  --dataset_path ./visualization/example_data_interhand_val.hdf5 \
  --model_path MODEL_DIR(e.g. ./best)
```

### InterHand retargeted viewer (human + robot in same frame)
```bash
# Requires DEX_URDF_PATH to be set.
python \
  visualization/visualize_inter.py \
  --dataset-path ./visualization/InterHand_retargeted_30fps_signed_aug_shape.hdf5 \
  --port 8091
```
