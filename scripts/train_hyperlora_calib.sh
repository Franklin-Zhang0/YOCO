#!/bin/bash
# Training script for HyperLoRA calibration with MANO-angle-space bias

# Set MANO_PATH environment variable (required for MANO forward kinematics)
# export MANO_PATH="${MANO_PATH:-/path/to/mano_v1_2/models}"
expname="hyperlora_8_pairs"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOAD_PATH="${LOAD_PATH:-$REPO_ROOT/expdata/test_combined_cnn_mlp_leaky_global_mixer/test_combined_cnn_mlp_leaky_global_mixer_20260111_134713/best}"
echo "Script directory: $SCRIPT_DIR"
# DATA_PATH=/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_aug_shape.hdf5 \
DATA_PATH=/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape_biased.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_aug_shape_biased.hdf5
# Check if MANO_PATH is set
if [ ! -d "$MANO_PATH" ]; then
    echo "Error: MANO_PATH not set or directory does not exist: $MANO_PATH"
    echo "Please set MANO_PATH to the directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl"
    exit 1
fi

# Run training
python -m calibration_net.policy.train \
    trainer.expname=$expname \
    config=config/hyperlora_calib.yaml \
    dataset.input_keys=world_hand_joints \
    dataset.output_keys=mano_angles \
    dataset.preprocessing_x=[joint_3d_world_to_local] \
    dataset.path=$DATA_PATH \
    dataset.num_hypernet_pairs=8 \
    val_dataset.path=/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/InterHand_retargeted_30fps_biased.hdf5\
    trainer.num_epochs=10 \
    trainer.batch_size=1024 \
    trainer.learning_rate=0.00005 \
    trainer.lr_scheduler.enabled=True \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    "trainer.load_path=$LOAD_PATH" \
    trainer.loss_type=mse \
    model.iknet_cfg.enabled=False \
    model.type=Combined_CNN_MLP_MIXER \
    trainer.logger.type=[terminal,file,wandb]

# Usage examples:
# 
# Basic training:
#   ./scripts/train_hyperlora_calib.sh
#
# Override dataset path:
#   ./scripts/train_hyperlora_calib.sh dataset.path=/path/to/dataset
#
# Resume from checkpoint:
#   ./scripts/train_hyperlora_calib.sh trainer.load_path=checkpoints/hyperlora_mano_bias_calib_20240115_120000/best
#
# Adjust bias strength:
#   ./scripts/train_hyperlora_calib.sh dataset.bias_cfg.per_finger_offset_std=0.2
#
# Change number of conditioning pairs:
#   ./scripts/train_hyperlora_calib.sh dataset.num_hypernet_pairs=8
