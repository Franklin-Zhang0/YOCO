import viser
import os
from pathlib import Path
import yourdfpy
from viser.extras import ViserUrdf
import numpy as np
import tyro
import time
from calibration_net.data_collection.poses import FIX_POSE_REGISTER
from calibration_net.data_collection.pose_collector import get_pose_collector
from calibration_net.teleop.vr_utils import calibration
from calibration_net.utils import AngleNormalizer
from pynput import keyboard
from datetime import datetime
import pickle

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')

def load_urdf(hand_type: str, side: str):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)    
    return urdf

def save_data(poses, betas, output_dir, idx, hand_type, angle_normalizer):
    global POSE_REGISTER
    broadcast_pose = np.repeat(1-np.array(POSE_REGISTER[ list(POSE_REGISTER.keys())[idx]])[np.newaxis, ...], len(poses['left']), axis=0)
    if hand_type == 'inspire':
        broadcast_pose = broadcast_pose[:, ::-1]
    denormed_broadcast_pose = angle_normalizer.denormalize(broadcast_pose)
    for side in ['left', 'right']:
        if side in poses and side in betas:
            pose = poses[side]
            save_dict = {
                'world_hand_joints': pose,
                'betas': betas[side],
                'mano_side': [side],
                'robot_joint_pose': denormed_broadcast_pose,
                'robot_type': hand_type,
                'pose_name': list(POSE_REGISTER.keys())[idx],
                'normalized_robot_joint_pose': broadcast_pose
            }
            save_path = os.path.join(output_dir, f'{side}_{idx}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
            print(f'Saved {side} hand pose to {save_path}')

def main(output_dir: str, 
         hand_type: str='inspire', 
         collector_type: str='avp', 
         ip: str='', 
         left_glove_port: str='', 
         right_glove_port: str='', 
         calibrate: bool=False,
         hand_model_path: str='',
         glove_model_path: str='',
         frame_rate: int=60,
         ):
    if collector_type == 'avp' and ip == '':
        raise ValueError("AVP IP address must be provided when collector_type is 'avp'.")
    if collector_type == 'glove' and left_glove_port == '' and right_glove_port == '':
        raise ValueError("At least one glove port must be provided when collector_type is 'glove'.")
    
    global POSE_REGISTER
    POSE_REGISTER = FIX_POSE_REGISTER[hand_type]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    left_urdf = load_urdf(hand_type, 'left')   
    right_urdf = load_urdf(hand_type, 'right')

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

    collect_next_button = server.gui.add_button("Start Collect Current Pose")
    collect_next_button.on_click(lambda _: collect_next())

    collect_prev_button = server.gui.add_button("Restore to Previous Pose")
    collect_prev_button.on_click(lambda _: restore_prev())

    left_base_frame.wxyz = np.zeros(4)
    left_base_frame.position = np.zeros(3)
    right_base_frame.wxyz = np.zeros(4)
    right_base_frame.position = np.zeros(3)
    pose_list = list(POSE_REGISTER.keys())
    stop_flag = False

    pose_collector = get_pose_collector(collector_type, avp_ip=ip, frame_rate=60, left_glove_port=left_glove_port, right_glove_port=right_glove_port, hand_model_path=hand_model_path, glove_model_path=glove_model_path, calibrate=calibrate)
    print('Fitting betas, please randomly move your hands for 10 seconds...')
    betas = calibration(pose_collector, n_comps=21, sides=['left'], timesteps=frame_rate*10)

    key_idx = 0
    poses = None
    sleep_flag = False
    from colorama import Fore, Style
    def collect_next():
        nonlocal key_idx, poses, sleep_flag
        #show sleeping in the gui
        print('\nPlease hold the pose for 3 seconds after the start signal.')
        sleep_flag = True
        time.sleep(2)
        print(Fore.RED+'\nCollecting pose...' + Style.RESET_ALL)
        poses = pose_collector.collect(timesteps=frame_rate * 3, sides=['left'])
        save_data(poses, betas, output_dir, key_idx, hand_type, angle_normalizer)
        key_idx += 1
        sleep_flag = False
        print(f'\nPose {key_idx}/{len(pose_list)} collected. Press n to collect next pose, r to restore previous pose, q to quit.')

    def restore_prev():
        nonlocal key_idx
        key_idx -= 1
        print(f'\nRestored to pose {key_idx}/{len(pose_list)}. Press n to collect next pose, r to restore previous pose, q to quit.')

    listener = keyboard.Listener(
        on_press=lambda key: on_press(key)
    )
    def on_press(key):
        try:
            nonlocal stop_flag
            if key.char == 'n':
                collect_next()
            elif key.char == 'r':
                restore_prev()
            elif key.char == 'q':
                stop_flag = True
                return False
        except AttributeError:
            pass
    listener.start()

    print(f'Press n to collect next pose, r to restore previous pose, q to quit.')
    while True:
        joints = np.array(POSE_REGISTER[pose_list[key_idx]])
        if hand_type == 'inspire':
            joints = 1 - joints[::-1]
            
        # joints = joints * (joint_upper_limits - joint_lower_limits) + joint_lower_limits
        joints = angle_normalizer.denormalize(joints)
        with server.atomic():
            left_urdf_vis.update_cfg(np.array(joints))
            right_urdf_vis.update_cfg(np.array(joints))
            if stop_flag:
                break
        if key_idx >= len(pose_list):
            print('All poses collected. Exiting in 2 seconds...')
            time.sleep(2)
            break
        if sleep_flag:
            time.sleep(4.9)

    listener.stop()
            


if __name__ == "__main__":
    tyro.cli(main)