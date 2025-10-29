import viser
import os
from pathlib import Path
import yourdfpy
from viser.extras import ViserUrdf
import numpy as np
import tyro
import time
from calibration_net.data_collection.poses import FIX_POSE_REGISTER
from calibration_net.utils import AngleNormalizer
from datetime import datetime
import pyroki as pk
import pickle

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')

def load_urdf(hand_type: str, side: str):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)    
    return urdf


def main(hand_type: str='inspire'):
    global POSE_REGISTER
    POSE_REGISTER = FIX_POSE_REGISTER[hand_type]

    left_urdf = load_urdf(hand_type, 'left')   
    right_urdf = load_urdf(hand_type, 'right')

    hand = pk.Robot.from_urdf(right_urdf)

    angle_normalizer = AngleNormalizer(Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_left.urdf")

    server = viser.ViserServer()

    # We will transform everything by the transform below, for aesthetics.
    server.scene.add_frame(
        "/scene_offset_left",
        show_axes=False,
        position=(-0.0, -0.3, 0.1),
        wxyz=(0.5, -0.5, -0.5, 0.5)
    )
    
    server.scene.add_frame(
        "/scene_offset_right",
        show_axes=False,
        position=(0.0, 0.3, 0.1),
        wxyz=(-0.5, -0.5, 0.5, 0.5),
    )
    
    left_base_frame = server.scene.add_frame("/scene_offset_left/base", show_axes=False)
    right_base_frame = server.scene.add_frame("/scene_offset_right/base", show_axes=False)
    left_urdf_vis = ViserUrdf(server, left_urdf, root_node_name="/scene_offset_left/base")
    right_urdf_vis = ViserUrdf(server, right_urdf, root_node_name="/scene_offset_right/base")
    server.scene.add_grid("/grid", 1.0, 1.0)


    left_base_frame.wxyz = np.zeros(4)
    left_base_frame.position = np.zeros(3)
    right_base_frame.wxyz = np.zeros(4)
    right_base_frame.position = np.zeros(3)


    for key, value in POSE_REGISTER.items():
        print(f'Pose {key}: {key}, values: {value}')
        joints = 1 - np.array(value)
        if hand_type == 'inspire':
            joints = joints[::-1]
            
        # joints = joints * (joint_upper_limits - joint_lower_limits) + joint_lower_limits
        joints = angle_normalizer.denormalize(joints)
        wxyz_pos = hand.forward_kinematics(cfg=joints)
        finger_pos = wxyz_pos[[4, 8, 12, 16, 20], -3:]
        with server.atomic():
            left_urdf_vis.update_cfg(np.array(joints))
            right_urdf_vis.update_cfg(np.array(joints))
        input("Press Enter to continue...")
            


if __name__ == "__main__":
    tyro.cli(main)