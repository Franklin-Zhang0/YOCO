#!/bin/bash
# Training script for HyperLoRA calibration with MANO-angle-space bias

# Set MANO_PATH environment variable (required for MANO forward kinematics)
# export MANO_PATH="${MANO_PATH:-/path/to/mano_v1_2/models}"
expname="New_DS_hyperlora_16_pairs_uniform-sub_peak3e-5"
# LOAD_PATH=/data/250010203/franklin/project/hand_teleop/Calibration_net/expdata/test_combined_cnn_mlp_leaky_global_mixer/test_combined_cnn_mlp_leaky_global_mixer_20260111_134713/best
LOAD_PATH=expdata/combined_cnn_mlp_leaky_global_mixer_scheduler_composite/combined_cnn_mlp_leaky_global_mixer_scheduler_composite_20260308_205854/best
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"
# DATA_PATH=/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_aug_shape.hdf5 \
DATA_PATH=/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_aug_shape_biased_complexer.hdf5
# DATA_PATH=/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/InterHand_retargeted_30fps_biased_complex.hdf5

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
    dataset.num_hypernet_pairs=16 \
    dataset.pair_subsample=True \
    dataset.frozen_mano_canonicalizer.enabled=True \
    dataset.frozen_mano_canonicalizer.extractor_path=$LOAD_PATH \
    val_dataset.path=/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/InterHand_retargeted_30fps_biased.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_complexer.hdf5\
    trainer.num_epochs=50 \
    trainer.batch_size=1024 \
    trainer.learning_rate=5e-5 \
    trainer.lr_scheduler.final_lr=2.0e-7 \
    trainer.lr_scheduler.enabled=True \
    trainer.lr_scheduler.total_steps=80000 \
    trainer.lr_scheduler.warmup_steps=2000 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    trainer.load_path=$LOAD_PATH \
    trainer.loss_type=refine_composite \
    trainer.loss_weights.mse=1.0 \
    trainer.loss_weights.rel_distance=1000.0 \
    trainer.loss_weights.rel_vector=1.0 \
    trainer.loss_weights.global_keypoint=100.0 \
    trainer.calib_finetune=False \
    model.iknet_cfg.enabled=False \
    model.type=Combined_CNN_MLP_MIXER \
    model.hypernet_cfg.rank=8 \
    model.hypernet_cfg.alpha=8.0 \
    'model.hypernet_cfg.name_filter=[]' \
    model.hypernet_cfg.drop_out=0.1 \
    model.hypernet_cfg.train_init=false \
    model.hypernet_cfg.init_method=xavier \
    model.hypernet_cfg.use_tanh=true \
    model.hypernet_cfg.scale_A=0.1 \
    model.hypernet_cfg.scale_B=0.2 \
    model.hypernet_cfg.use_gates=true \
    model.hypernet_cfg.gate_init=0.0 \
    model.hypernet_cfg.max_norm=null \
    model.hypernet_cfg.reg_weight=0.001 \
    model.hypernet_cfg.reg_type=frobenius \
    model.hypernet_cfg.head_type=dit_flow \
    trainer.logger.type=[terminal,file,wandb]

    # dataset.reference_poses_path=20260306_151100/paired_poses.hdf5 \


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
