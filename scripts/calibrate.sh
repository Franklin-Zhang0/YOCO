CALIBRATOR=$1
python collect_pose.py \
    --left_glove_port /dev/ttyUSB0 \
    --save_path ./calibrate/$CALIBRATOR/calib_data \
    --type critical_continous \
    --round 1

python merge_dataset.py \
    --source_dir ./calibrate/$CALIBRATOR/calib_data \
    --save_path ./calibrate/$CALIBRATOR/calib_data_merged \
    --is_continous

python train.py \
    config=config/calib.yaml \
    trainer.expname=calib \
    dataset.path=./calibrate/$CALIBRATOR/calib_data_merged \
    trainer.load_path=model \
    trainer.num_epochs=3 \
    trainer.batch_size=64 \
    trainer.save_path=./calibrate/$CALIBRATOR/calib_model \

model_name=$(ls ./calibrate/$CALIBRATOR/calib_model)

python ./teleop.py --left_port /dev/ttyUSB0 --left_robot_port /dev/ttyUSB2 --model_dir ./calibrate/$CALIBRATOR/calib_model/$model_name/best