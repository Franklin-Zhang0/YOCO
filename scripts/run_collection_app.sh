#!/bin/bash -i

conda activate teleop
FILE_DIR=$(dirname -- "$0")
echo "Changing directory to $FILE_DIR/.."
DIR=$FILE_DIR/..
python $DIR/calibration_net/data_collection/vr_collect/web_app.py
 