### Installation Guide
```
git clone https://github.com/Franklin-Zhang0/Calibration_net.git
cd Calibration_net
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/CyberOrigin2077/open_cyber_glove.git
pip install git+https://github.com/Sentdex/inspire_hands.git
```

### Teleoperation Guide
To see the port of gloves on your device, type
```
# Windows
usbipd list  # your might need to install usbipd first
# Linux
ls /dev/ttyUSB*
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