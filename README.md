### Installation Guide
On ROS2 kilted, you can run:
```
git clone https://github.com/Franklin-Zhang0/Calibration_net.git
cd Calibration_net
conda creeate -n teleop python=3.11
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
If you are running on older ROS2 versions, and need the ROS2 node, run:
```
git clone https://github.com/Franklin-Zhang0/Calibration_net.git
cd Calibration_net
conda creeate -n teleop python=3.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

After Installing the torch, run:
```
git clone https://github.com/CyberOrigin2077/open_cyber_glove.git
pip install -e open_cyber_glove
git clone https://github.com/Sentdex/inspire_hands.git
pip install -e inspire_hands
```

### Teleoperation Guide
To see the port of gloves on your device, type
```
# Windows
usbipd list  # your might need to install usbipd first
# Linux
ls /dev/ttyUSB*
```

To set the permission of the glove port:
```
sudo chmot +777 /dev/ttyUSBx
```

```
python /home/franklin/Project/Hand_teleop/code/calibration_net/teleop.py \
    --left_port XXX \
    --right_port XXX \
    --control_frequency 60 \
    --robot_speed 0.4 \
    --calibration \
    --model_dir ./model
```

### Calibration and teleop Guide
This will ask you to wear glove and do severl poses to train a fintuned, calibrated model.
The model will be saved in calibrate/XXX/calib_model/best
XXX can be any string, which is used as a tag.
```
bash ./script/calibration.sh XXX
```


### ROS 2 node
```
python teleop_glove_node.py --ros-args -p left_port:=/dev/ttyUSBx -p model_dir:=XXX
```