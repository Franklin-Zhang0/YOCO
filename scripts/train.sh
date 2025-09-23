expname=test_hyper_iknet_double_dropout_webdata_only_weight_decay
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.path=/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_aug_shape.hdf5,/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5 \
    dataset.input_keys=world_hand_joints \
    dataset.cat_side=True \
    dataset.output_keys=inspire_joint_pose \
    dataset.preprocessing_x=[joint_3d_world_to_local,bone_normalize,flatten] \
    val_dataset.path=/mnt/sda/yuzhang/data/vr_collected/hdf5/20250909.hdf5 \
    trainer.num_epochs=20 \
    trainer.batch_size=256 \
    trainer.learning_rate=0.0003 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    trainer.logger.type=[wandb] \
    model.iknet_cfg.enabled=True \
    model.type=Flow \
    model.hypernet_cfg.name_filter=['film','cond','invk_layers'] \
    model.hypernet_cfg.drop_out=0.3 \
    dataset.hypernet_input_keys=betas