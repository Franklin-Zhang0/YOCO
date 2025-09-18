import viser
import os
from pathlib import Path
import yourdfpy
from calibration_net.policy.inferencer import ModelInferencer
from viser.extras import ViserUrdf
import numpy as np
from calibration_net.teleop.vr_utils import initialize_fitter
from calibration_net.teleop.mano_fitter import PoseCollector
import tyro
import time
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')


def retargeting(model_dir, side, queue, calibration):
    inferencer = ModelInferencer(model_dir=model_dir)

    pose_collector = PoseCollector(30)
    fitter = initialize_fitter(inferencer.n_comps, [side])
    if calibration:
        print("Starting calibration...")
        poses = pose_collector.collect(400)
        betas = fitter[side].fit_sequence(poses[side], optimize_scale=False)['betas']
        print("Calibration done. Betas:", betas)
        
    max_steps = 200
    prev_result = None
    while True:
        joint_pose = pose_collector.collect(1, sides=[side])[side][0]
        if not hasattr(inferencer.model, 'iknet'):
            result = fitter[side].fit_single_frame(joint_pose, prev_result=prev_result, max_steps=max_steps)
            model_input = result['hand_pose']
            max_steps = 200
            prev_result = result
        else:
            model_input = joint_pose
        joints = inferencer.inference(model_input, batched=False)
        queue.put(joints)

def main(model_dir:str, hand_type: str='inspire', side='right', calibration:bool=True):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    queue = mp.Queue()
    retarget_process = mp.Process(target=retargeting, args=(model_dir, side, queue, calibration), daemon=True)
    retarget_process.start()

    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)    
    
    server = viser.ViserServer()

    # We will transform everything by the transform below, for aesthetics.
    server.scene.add_frame(
        "/scene_offset",
        show_axes=False,
        position=(-0.0, -0.0, 0.2),
        wxyz=(-0.0, 0.0, 0.0, 0.0),
    )
    
    base_frame = server.scene.add_frame("/scene_offset/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/scene_offset/base")
    server.scene.add_grid("/grid", 1.0, 1.0)
    
    prev_joints = np.zeros(6)
    while True:
        with server.atomic():
            base_frame.wxyz = np.zeros(4)
            base_frame.position = np.zeros(3)
            
            try:
                joints = queue.get(block=False)
            except:
                joints = prev_joints

            urdf_vis.update_cfg(np.array(joints))
            prev_joints = joints


if __name__ == "__main__":
    tyro.cli(main)