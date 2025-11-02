import viser
import os
from pathlib import Path
import yourdfpy
from calibration_net.policy.inferencer import ModelInferencer
from viser.extras import ViserUrdf
import numpy as np
from calibration_net.teleop.vr_utils import initialize_fitter
from calibration_net.data_collection.pose_collector import get_pose_collector
import tyro
import time
import multiprocessing as mp
from calibration_net.utils import AngleNormalizer
from pynput import keyboard

mp.set_start_method('spawn', force=True)
DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')


def retargeting(model_dir, side, queue, calibration):
    inferencer = ModelInferencer(model_dir=model_dir, side=side)

    pose_collector = get_pose_collector(collector_type='avp', avp_ip='10.212.36.171', frame_rate=60)
    fitter = initialize_fitter(inferencer.n_comps, [side])
    if calibration:
        print("Starting calibration...")
        poses = pose_collector.collect(400)
        betas = fitter[side].fit_sequence(poses[side], optimize_scale=False)['betas']
        print("Calibration done. Betas:", betas)

    max_steps = 200
    prev_result = None
    i=0
    import time

    start = time.time()
    while True:
        i+=1
        joint_pose = pose_collector.collect(1, sides=[side])[side][0]
        model_input = joint_pose
        joints = inferencer.inference(model_input, batched=False)
        queue.put(joints)
        if i % 30 == 0:
            print(f"FPS: {30/(time.time()-start):.2f}")
            start = time.time()

def main(model_dir:str, hand_type: str='inspire', side='right', calibration:bool=False):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    queue = mp.Queue(maxsize=1)
    retarget_process = mp.Process(target=retargeting, args=(model_dir, side, queue, calibration), daemon=True)
    retarget_process.start()

    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)  
    angle_normalizer = AngleNormalizer(robot_urdf_path)

    server = viser.ViserServer()

    # We will transform everything by the transform below, for aesthetics.
    server.scene.add_frame(
        "/scene_offset",
        show_axes=False,
        position=(-0.0, -0.0, 0.0),
        wxyz=(-0.0, 0.0, 0.0, 0.0),
    )

    pause_flag = False
    def on_press(key):
        nonlocal pause_flag
        if key == keyboard.Key.space:
            pause_flag = True
            print("Paused")

    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    base_frame = server.scene.add_frame("/scene_offset/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/scene_offset/base")
    server.scene.add_grid("/grid", 1.0, 1.0)
    
    while True:
        with server.atomic():
            base_frame.wxyz = np.zeros(4)
            base_frame.position = np.zeros(3)
            
            try:
                joints = queue.get(block=False)
            except:
                continue
            # print(joints)
            if pause_flag:
                breakpoint()
                pause_flag = False
            joints = angle_normalizer.denormalize(joints)

            urdf_vis.update_cfg(joints)


if __name__ == "__main__":
    tyro.cli(main)