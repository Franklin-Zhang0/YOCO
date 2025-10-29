import viser
import os
from pathlib import Path
import yourdfpy
from viser.extras import ViserUrdf
import numpy as np
import tyro
import time
from calibration_net.utils import AngleNormalizer
from calibration_net.policy.models.fk_model import FKModel, TIP_INDICIES
from calibration_net.policy.preprocessing import transform_joint3d
from datetime import datetime
import pickle
import torch

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')

def load_urdf(hand_type: str, side: str):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)    
    return urdf

keypoint_joints = [[0,1,2,3],[4,5,6],[7,8,9],[10],[11]]

def main(fk_model_path: str, path:str, hand_type: str='omni'):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    
    robot_pose = data['robot_joint_pose']
    keypoints = data['world_hand_joints']
    keypoints = transform_joint3d(keypoints)
    
    right_urdf = load_urdf(hand_type, 'right')

    angle_normalizer = AngleNormalizer(Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_left.urdf")
    
    fk_model = FKModel(keypoint_joints)
    fk_model.load_state_dict(torch.load(fk_model_path))
    fk_model.eval()

    server = viser.ViserServer()

    server.scene.add_frame(
        "/scene_offset_right",
        show_axes=False,
        position=(0.0, 0.0, 0.),
        wxyz=(1., 0., 0., 0.),
    )
    
    right_base_frame = server.scene.add_frame("/scene_offset_right/base", show_axes=False)
    right_urdf_vis = ViserUrdf(server, right_urdf, root_node_name="/scene_offset_right/base")
    server.scene.add_grid("/grid", 1.0, 1.0)

    right_base_frame.wxyz = np.zeros(4)
    right_base_frame.position = np.zeros(3)

    i = 0
    while True:
        joints = robot_pose[i%len(robot_pose)]
        if hand_type == 'inspire':
            joints = 1 - np.array(joints[::-1])
            
        with torch.no_grad():
            fingertip_pos = fk_model(torch.from_numpy(joints).unsqueeze(0).float()).cpu().numpy()[0]
                        
        # joints = joints * (joint_upper_limits - joint_lower_limits) + joint_lower_limits
        joints = angle_normalizer.denormalize(joints)
        
        
        with server.atomic():
            right_urdf_vis.update_cfg(np.array(joints))
            server.scene.add_point_cloud(
                "/scene_offset/fk_fingertip_keypoints",
                (fingertip_pos).reshape(-1, 3),
                np.array((0, 0, 255))[None]
                .repeat(fingertip_pos.reshape(-1,3).shape[0], axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="sparkle",
            )
            
            current_keypoint = keypoints[i%len(keypoints)][TIP_INDICIES]
            server.scene.add_point_cloud(
                "/scene_offset/keypoints",
                (current_keypoint).reshape(-1, 3),
                np.array((0, 255, 0))[None]
                .repeat(current_keypoint.reshape(-1,3).shape[0], axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="sparkle",
            )
        i+=1
        time.sleep(0.1)
            


if __name__ == "__main__":
    tyro.cli(main)