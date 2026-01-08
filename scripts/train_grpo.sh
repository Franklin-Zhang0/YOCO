#!/bin/bash

# Default config path
CONFIG=${1:-config/grpo.yaml}
LOAD_PATH=${2:-model/}
DATASET_PATH=/Users/franklin/Project/Hand_Teleop/code/Calibration_net/data/glove_right_franklin/glove_franklin_right_pair.hdf5

python -m calibration_net.policy.grpo_trainer config=$CONFIG \
    dataset.path=$DATASET_PATH \
    dataset.preprocessing_x=[joint_3d_world_to_local] 