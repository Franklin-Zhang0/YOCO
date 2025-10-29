expname=test_mlp_omni_composite
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.input_keys=world_hand_joints \
    dataset.cat_side=True \
    dataset.output_keys=robot_joint_pose \
    dataset.preprocessing_x=[joint_3d_world_to_local,bone_normalize] \
    dataset.path=/mnt/sda/yuzhang/data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced,/mnt/sda/yuzhang/data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape,/mnt/sda/yuzhang/data/vr_collected/hdf5/avp_15min_20251020_000547.hdf5 \
    val_dataset.path=/mnt/sda/yuzhang/data/InterHand/val/hdf5/InterHand_retargeted_30fps_omni \
    trainer.num_epochs=100 \
    trainer.batch_size=4096 \
    trainer.learning_rate=0.0001 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    trainer.logger.type=[terminal] \
    trainer.loss_type=composite \
    model.iknet_cfg.enabled=False \
    model.type=MLP \
    model.hypernet_cfg.name_filter=['layers'] \
    dataset.hypernet_input_keys=betas \
    model.hypernet_cfg.drop_out=0.5 \
    # model.hypernet_cfg.name_filter=['layers','invk_layers'] \
    # model.hypernet_cfg.drop_out=0.3 \
    # dataset.hypernet_input_keys=betas

    # model.type=MLP 
    # model.type=Flow 
    # model.hypernet_cfg.name_filter=['film','cond','invk_layers'] \
    # model.hypernet_cfg.name_filter=['layers','invk_layers'] \
    # dataset.preprocessing_x=[normalize] \
    # dataset.path=/mnt/sda/yuzhang/data/vr_collected/hdf5/20250912_aug_shape.hdf5,/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_aug_shape.hdf5,/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5 \
    # dataset.path=/mnt/sda/yuzhang/data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced,/mnt/sda/yuzhang/data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape,/mnt/sda/yuzhang/data/vr_collected/hdf5/avp_15min_20251020_000547.hdf5 \
    # dataset.preprocessing_y=[original_and_delta] \





