#!/bin/bash
# Training script for HyperLoRA calibration with direct keypoint outputs.

expname="hyperlora_8_pairs_nosub_peak3e-5_keypoint_mixer_one_time_more_ds_norm"
LOAD_PATH=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

DATA_PATH=/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_aug_shape_biased_complexer.hdf5

python -m calibration_net.policy.train \
    trainer.expname=$expname \
    config=config/hyperlora_calib.yaml \
    dataset.input_keys=world_hand_joints \
    dataset.output_keys=world_hand_joints \
    dataset.preprocessing_x=[joint_3d_world_to_local] \
    dataset.preprocessing_y=[joint_3d_world_to_local] \
    dataset.path=$DATA_PATH \
    dataset.num_hypernet_pairs=8 \
    dataset.pair_subsample=False \
    val_dataset.path=/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/InterHand_retargeted_30fps_biased.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_biased_complexer.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced_biased_complexer.hdf5 \
    trainer.num_epochs=50 \
    trainer.batch_size=1024 \
    trainer.learning_rate=1e-4 \
    trainer.lr_scheduler.final_lr=1.0e-6 \
    trainer.lr_scheduler.enabled=True \
    trainer.lr_scheduler.total_steps=80000 \
    trainer.lr_scheduler.warmup_steps=2000 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    trainer.load_path=$LOAD_PATH \
    trainer.loss_type=refine_composite \
    trainer.loss_weights.mse=1.0 \
    trainer.loss_weights.rel_distance=0.0 \
    trainer.loss_weights.rel_vector=0.0 \
    trainer.loss_weights.global_keypoint=0.0 \
    trainer.calib_finetune=False \
    model.iknet_cfg.enabled=False \
    model.type=Combined_CNN_MLP_MIXER_KEYPOINT \
    model.hypernet_cfg.rank=8 \
    model.hypernet_cfg.name_filter=["keypoint_regressor","fusion_attention"] \
    trainer.logger.type=[terminal,file]
