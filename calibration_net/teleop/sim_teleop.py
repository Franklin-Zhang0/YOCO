import viser
import os
from pathlib import Path
import yourdfpy
from calibration_net.policy.inferencer import ModelInferencer
from viser.extras import ViserUrdf
import numpy as np
from calibration_net.teleop.vr_utils import initialize_fitter
from calibration_net.teleop.mano_fitter import PoseCollector

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')


def main(model_dir:str, hand_type: str='inspire', side='left'):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_right.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
    
    inferencer = ModelInferencer(model_dir=model_dir)
    
    
    server = viser.ViserServer()

    # We will transform everything by the transform below, for aesthetics.
    server.scene.add_frame(
        "/scene_offset",
        show_axes=False,
        position=(-0.15415953, -0.73598871, 0.93434792),
        wxyz=(-0.381870867, 0.92421569, 0.0, 2.0004992e-32),
    )
    
    base_frame = server.scene.add_frame("/scene_offset/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/scene_offset/base")
    playing = server.gui.add_checkbox("playing", True)
    # object_handle = server.scene.add_mesh_trimesh("/scene_offset/object", mesh)
    server.scene.add_grid("/grid", 2.0, 2.0)
    
    pose_collector = PoseCollector(30)
    fitter = initialize_fitter(inferencer.n_comps.input_size, [side])
    max_steps = 200
    prev_result = None
    
    while True:
        with server.atomic():
            base_frame.wxyz = np.zeros(4)
            base_frame.position = np.zeros(3)
            
            joint_pose = pose_collector.collect(1)[side][0]
            result = fitter[side].fit_single_frame(joint_pose, prev_result=prev_result, max_steps=max_steps)
            
            mano_pose = result['hand_pose']
            joints = inferencer.inference(mano_pose, batched=False)
            
            urdf_vis.update_cfg(np.array(joints))
            max_steps=20
            prev_result = result