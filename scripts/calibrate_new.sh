SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

python $SCRIPT_DIR/../calibration_net/data_collection/vr_pair_collect.py \
    --output_dir ./calibrate/data \
    --collector_type glove \
    --hand_model_path /home/franklin/project/hand_teleop/open_cyber_glove/models/hand_model.pkl \
    --glove_model_path /home/franklin/project/hand_teleop/open_cyber_glove/models/20250703_110909.onnx \
    --left_glove_port /dev/ttyUSB0

calib_data_dir="$SCRIPT_DIR/../calibrate/data"
data_dirs=( "$calib_data_dir"/* )
if ((${#data_dirs[@]} == 0)); then
    echo "No data directories found in $calib_data_dir" >&2
    exit 1
fi
# Pick the last element of the array
data_dir=${data_dirs[-1]}
echo "Using data directory: $data_dir"

python $SCRIPT_DIR/../calibration_net/data_collection/merge_dataset_hdf5.py \
    --source_dir $data_dir \
    --save_path $data_dir"/merged.hdf5"

expname=calibrate

# calibration finetuning
python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.path=$data_dir"/merged.hdf5" \
    dataset.input_keys=world_hand_joints \
    dataset.cat_side=True \
    dataset.output_keys=robot_joint_pose \
    dataset.preprocessing_x=[joint_3d_world_to_local,bone_normalize,flatten] \
    val_dataset.path=$data_dir"/merged.hdf5"  \
    trainer.num_epochs=10 \
    trainer.batch_size=256 \
    trainer.learning_rate=0.001 \
    trainer.load_path=$SCRIPT_DIR/../model \
    trainer.save_path="$SCRIPT_DIR/../calibrate/models" \
    trainer.device=cuda \
    trainer.logger.type=[terminal] \
    trainer.calib_finetune=True \
    model.iknet_cfg.enabled=True \
    model.type=Flow \
    model.hypernet_cfg.name_filter=['film','cond','invk_layers'] \
    model.hypernet_cfg.drop_out=0.3 \
    dataset.hypernet_input_keys=betas 
