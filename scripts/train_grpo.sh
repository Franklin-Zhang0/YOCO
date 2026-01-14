#!/bin/bash

# Default config path
CONFIG=${1:-config/grpo.yaml}
EXPNAME=lora_rank2-kl_1e-6
LOAD_PATH=/data/250010203/franklin/project/hand_teleop/Calibration_net/expdata/test_combined_cnn_mlp_leaky_global_mixer/test_combined_cnn_mlp_leaky_global_mixer_20260111_134713/best
DATASET_PATH=/data/250010203/franklin/data/hand_keypoint_data/pair_data/glove_franklin_right_pair.hdf5

# Standard GRPO training (without LoRA)
python -m calibration_net.policy.grpo_trainer config=$CONFIG \
    dataset.path=$DATASET_PATH \
    dataset.preprocessing_x=[joint_3d_world_to_local] \
    dataset.input_keys=world_hand_joints \
    trainer.load_path=$LOAD_PATH \
    trainer.expname=$EXPNAME \
    model.type=Combined_CNN_MLP_MIXER \
    trainer.logger.type=[terminal,file,wandb] \
    trainer.mano_kl_coeff=1e-6 \
    trainer.lora.enabled=true \
    trainer.lora.rank=2 \
    trainer.lora.alpha=4.0 \
    trainer.lora.dropout=0.1 \
    trainer.lora.target_modules=[mano_regressor,fusion_attention] \
    trainer.lora.train_finger_scaling=true

# To enable LoRA mode, add these parameters:
# trainer.lora.enabled=true \
# trainer.lora.rank=8 \
# trainer.lora.alpha=16.0 \
# trainer.lora.dropout=0.1 \
# trainer.lora.target_modules=[mano_regressor,fusion_attention] \
# trainer.lora.train_finger_scaling=true

# Example with LoRA enabled:
# python -m calibration_net.policy.grpo_trainer config=$CONFIG \
#     dataset.path=$DATASET_PATH \
#     dataset.preprocessing_x=[joint_3d_world_to_local] \
#     dataset.input_keys=world_hand_joints \
#     trainer.load_path=$LOAD_PATH \
#     trainer.expname=${EXPNAME}_lora \
#     model.type=Combined_CNN_MLP_MIXER \
#     trainer.logger.type=[terminal,file,wandb] \
#     trainer.mano_kl_coeff=1e-7 \
#     trainer.lora.enabled=true \
#     trainer.lora.rank=8 \
#     trainer.lora.alpha=16.0