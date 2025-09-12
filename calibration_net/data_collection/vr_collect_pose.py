from calibration_net.teleop.mano_fitter import PoseCollector, ManoSequenceFitter
import tyro
import os
import pickle
import datetime

MANO_PATH = os.environ.get("MANO_PATH", "")

def main(save_path:str, n_sec:int=300, frame_rate:int=30, ncomps=21, slice_len:int=100, sides=['left','right']):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    pose_collector = PoseCollector(frame_rate)
    poses = pose_collector.collect(n_sec*frame_rate)

    for side in sides:
        result = {'world_hand_joints': [], 'mano_side': [side], 'mano_angles': [], 'pca_mano_pose': [], 'betas': [], 'trans': [], 'global_orient': []}

        mano_fitter = ManoSequenceFitter(
                mano_model_dir=os.path.join(MANO_PATH, 'models'),
                mano_side=side,
                use_pca=True,                      
                ncomps=ncomps,
                iters=400,
                w_joint=1.0,
                w_pose_prior=0.001,
                w_shape_prior=0.1,
        )
        out = mano_fitter.fit_sequence(poses[side], optimize_scale=False)
        result['betas'] = out['betas']
        result['pca_mano_pose'] = out['hand_pose']
        result['trans'] = out['transl']
        result['global_orient'] = out['global_orient']
        result['mano_angles'] = result['pca_mano_pose'] @ mano_fitter.mano_layer.th_selected_comps.cpu().numpy()
        result['world_hand_joints'] = poses[side]

        for i in range(0, len(result['world_hand_joints']), slice_len):
            slice_result = {}
            for k in ['world_hand_joints', 'mano_angles', 'pca_mano_pose', 'trans', 'global_orient']:
                slice_result[k] = result[k][i:i+slice_len]
            slice_result['betas'] = result['betas']
            slice_result['mano_side'] = result['mano_side']
            with open(os.path.join(save_path, f'{timestamp}_{side}_{i//slice_len}.pkl'), 'wb') as f:
                pickle.dump(slice_result, f)
if __name__ == "__main__":
    tyro.cli(main)

