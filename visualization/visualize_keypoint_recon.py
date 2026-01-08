from calibration_net.policy.model import load_model
import tyro
import h5py
import torch
import viser
import numpy as np
from calibration_net.utils import mano_recon, transform_joint3d
import time

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def main(dataset_path: str, model_path: str):
    model, _, _ = load_model(model_path)
    data = h5py.File(dataset_path, 'r')
    
    
    server = viser.ViserServer()
    
    server.scene.add_grid("/grid", 1.0, 1.0)
    for traj_id in data.keys():
        pose_name = data[traj_id]['pose_name'][()].decode('utf-8')
        print(f"Processing trajectory {traj_id} with pose {pose_name}")
        keypoint_traj = data[traj_id]['world_hand_joints'][:]  # (T, 21, 3)
        keypoint_traj = transform_joint3d(keypoint_traj)
        pred_theta = model.predict(keypoint_traj, side='right')  # (T, 48)
        _, recon_joint3d = mano_recon(pred_theta)
        recon_joint3d = transform_joint3d(recon_joint3d)
        
        recon_joint3d = recon_joint3d.numpy()
        
        for frame_idx in range(keypoint_traj.shape[0]):
            with server.atomic():
                
                #pred joints
                server.scene.add_point_cloud(
                    name=f"/joints",
                    points=recon_joint3d[frame_idx],
                    colors=np.array([0.0, 0.0, 1.0]),
                    point_size=0.002,
                )
                lines = []
                for start, end in MANO_CONNECTIONS:
                    lines.append(np.array([recon_joint3d[frame_idx][start],recon_joint3d[frame_idx][end]]))
                
                server.scene.add_line_segments(
                    name=f"/bones",
                    points=np.array(lines),
                    colors=np.array([0.0, 0.0, 1.0]) ,
                    line_width=1.0,
                )
                
                # ground truth joints
                server.scene.add_point_cloud(
                    name=f"/joints_gt",
                    points=keypoint_traj[frame_idx],
                    colors=np.array([1.0, 0.0, 0.0]),
                    point_size=0.002,
                )
                lines_gt = []
                for start, end in MANO_CONNECTIONS:
                    lines_gt.append(np.array([keypoint_traj[frame_idx][start],keypoint_traj[frame_idx][end]]))
                
                server.scene.add_line_segments(
                    name=f"/bones_gt",
                    points=np.array(lines_gt),
                    colors=np.array([1.0, 0.0, 0.0]),
                    line_width=1.0,
                )
            time.sleep(0.1)
        
        
if __name__ == "__main__":
    tyro.cli(main)