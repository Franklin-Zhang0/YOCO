expname=test

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.path=/mnt/sda/yuzhang/data/dex_ycb_retargeted/val_huggingface \
    dataset.input_keys=left_convention_mano_angles \
    dataset.output_keys=inspire_joint_pose \
    dataset.hypernet_input_keys=betas \
    dataset.preprocessing_x=[normalize] \
    trainer.num_epochs=10 \
    trainer.batch_size=8 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.device=cuda \
    model.type=Flow 

    # model.type=Flow 