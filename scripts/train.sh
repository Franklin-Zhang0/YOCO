expname=test_flow_all_data_pca_shuffle

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.path=/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_aug_pose_shape.hdf5,/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_aug_shape.hdf5,/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5 \
    dataset.input_keys=pca_mano_pose \
    dataset.output_keys=inspire_joint_pose \
    dataset.hypernet_input_keys=betas \
    val_dataset.path=/mnt/sda/yuzhang/data/InterHand/val/hdf5/InterHand_retargeted_30fps \
    trainer.num_epochs=20 \
    trainer.batch_size=256 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    trainer.logger.type=[wandb] \
    model.hypernet_cfg.name_filter=['film','cond'] \
    model.type=Flow 

    # model.type=MLP 
    # model.type=Flow 
    # dataset.preprocessing_x=[normalize] \
    # dataset.obs_horizon=3 \
    # dataset.path=/mnt/sda/yuzhang/data/dex_ycb_retargeted/val_huggingface \
    # dataset.path=/mnt/sda/yuzhang/data/InterHand/InterHand_retargeted_30fps_huggingface/ \


