from calibration_net.teleop.mano_fitter import ManoSequenceFitter
from calibration_net.data_collection.pose_collector import BaseCollector
import os
import torch

MANO_PATH = os.environ.get("MANO_PATH", "")
def calibration(pose_collector: BaseCollector, n_comps:int, sides:list[str], timesteps:int=200):
    poses = pose_collector.collect(timesteps)
    betas = {}
    for side in sides:
        mano_fitter = ManoSequenceFitter(
                mano_model_dir=MANO_PATH,
                mano_side=side,
                use_pca=n_comps < 45,                      
                ncomps=n_comps,
                iters=200,
                w_joint=1.0,
                w_pose_prior=0.001,
                w_shape_prior=0.01,
                w_trans_smooth=0.1,          
                w_pose_smooth=0.01
            )
        out = mano_fitter.fit_sequence(poses[side], optimize_scale=False)
        betas[side] = out['betas']
        
    return betas

def initialize_fitter(n_comps:int, sides:list[str]):
    mano_fitter = {}
    for side in sides:
        mano_fitter[side] = ManoSequenceFitter(
            mano_model_dir=MANO_PATH,
            mano_side=side,
            use_pca=n_comps < 45,                         
            ncomps=n_comps,
        )
        mano_fitter[side].initialize_single_frame_fitting()
    
    return mano_fitter