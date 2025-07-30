CALIBRATOR=$1
GLOVE_PORT=$2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.data_collection.collect_pose \
    --left_glove_port $GLOVE_PORT \
    --save_path $SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_data \
    --type critical_continous \
    --round 1

python -m calibration_net.data_collection.merge_dataset \
    --source_dir $SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_data \
    --save_path $SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_data_merged \
    --is_continous

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/calib.yaml \
    trainer.expname=calib \
    dataset.path=$SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_data_merged \
    trainer.load_path=$SCRIPT_DIR/../model \
    trainer.num_epochs=3 \
    trainer.batch_size=64 \
    trainer.save_path=$SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_model \

model_name=$(ls $SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_model)
echo "Model name: $model_name"

python -m calibration_net.teleop.teleop --left_port=$GLOVE_PORT --model_dir=$SCRIPT_DIR/../calibrate/$CALIBRATOR/calib_model/$model_name/best