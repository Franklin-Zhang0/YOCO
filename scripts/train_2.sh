expname=combined_cnn_mlp_leaky_global_mixer_scheduler_no_rel
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"

python -m calibration_net.policy.train \
    config=$SCRIPT_DIR/../config/config.yaml \
    trainer.expname=$expname \
    dataset.input_keys=world_hand_joints \
    dataset.cat_side=True \
    dataset.output_keys=mano_angles \
    dataset.preprocessing_x=[joint_3d_world_to_local] \
    dataset.path=/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced.hdf5,/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_signed_aug_shape,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547_aug_shape.hdf5,/data/250010203/franklin/data/hand_keypoint_data/vr_collected/hdf5/avp_15min_20251020_000547.hdf5 \
    val_dataset.path=/data/250010203/franklin/data/hand_keypoint_data/InterHand/val/hdf5/InterHand_retargeted_30fps_omni.hdf5\
    trainer.num_epochs=100 \
    trainer.batch_size=1024 \
    trainer.learning_rate=0.0001 \
    trainer.lr_scheduler.enabled=True \
    trainer.lr_scheduler.warmup_steps=500 \
    trainer.lr_scheduler.peak_lr=3e-4 \
    trainer.lr_scheduler.final_lr=1.0e-6 \
    trainer.lr_scheduler.total_steps=3413 \
    trainer.save_path=$SCRIPT_DIR/../expdata/$expname \
    trainer.calibnet_warmup_epochs=0 \
    trainer.device=cuda \
    trainer.logger.type=[terminal,file,wandb] \
    trainer.loss_type=refine_composite \
    trainer.loss_weights.mse=1.0 \
    trainer.loss_weights.rel_distance=0.0 \
    trainer.loss_weights.rel_vector=0.0 \
    trainer.loss_weights.global_keypoint=0.5 \
    model.iknet_cfg.enabled=False \
    model.type=Combined_CNN_MLP_MIXER \
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





