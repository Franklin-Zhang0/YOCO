expname=test

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.path=/mnt/sda/yuzhang/data/InterHand/train/InterHand_retargeted_30fps_huggingface/ \
    dataset.input_keys=mano_angles \
    dataset.output_keys=inspire_joint_pose \
    dataset.hypernet_input_keys=betas \
    val_dataset.path=/mnt/sda/yuzhang/data/InterHand/val/InterHand_retargeted_30fps_huggingface/ \
    trainer.num_epochs=20 \
    trainer.batch_size=128 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.device=cuda \
    model.type=Flow 

    # model.type=Flow 
    # dataset.path=/mnt/sda/yuzhang/data/dex_ycb_retargeted/val_huggingface \
    # dataset.path=/mnt/sda/yuzhang/data/InterHand/InterHand_retargeted_30fps_huggingface/ \


