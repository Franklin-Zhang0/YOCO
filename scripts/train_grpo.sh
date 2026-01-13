#!/bin/bash

# Default config path
CONFIG=${1:-config/grpo.yaml}
EXPNAME=GRPO_TUNE
LOAD_PATH=/data/250010203/franklin/project/hand_teleop/Calibration_net/expdata/test_combined_cnn_mlp_leaky_global_mixer/test_combined_cnn_mlp_leaky_global_mixer_20260111_134713/best
DATASET_PATH=/data/250010203/franklin/data/hand_keypoint_data/pair_data/glove_franklin_right_pair.hdf5

python -m calibration_net.policy.grpo_trainer config=$CONFIG \
    dataset.path=$DATASET_PATH \
    dataset.preprocessing_x=[joint_3d_world_to_local] \
    dataset.input_keys=world_hand_joints \
    trainer.load_path=$LOAD_PATH \
    trainer.expname=$EXPNAME \
    model.type=Combined_CNN_MLP_MIXER 