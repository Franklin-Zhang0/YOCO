import viser
import os
from pathlib import Path
import yourdfpy
from viser.extras import ViserUrdf
import numpy as np
import tyro
import time
import pickle
from calibration_net.utils import AngleNormalizer

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def load_urdf(hand_type: str, side: str):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)    
    return urdf

class PlaybackNode:
    def __init__(self, file_path, viser_port):
        self.file_path = file_path
        self.viser_port = viser_port
        
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.hand_type = self.data.get('robot_type', 'inspire')
        self.sides = self.data.get('mano_side', ['left'])
        
        # Load URDFs
        self.urdfs = {}
        for side in ['left', 'right']:
            try:
                self.urdfs[side] = load_urdf(self.hand_type, side)
            except Exception as e:
                print(f"Could not load URDF for {side}: {e}")

        self.server = viser.ViserServer(port=viser_port)
        
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (0.0, 0.0, -0.5)
            client.camera.look_at = (0.0, 0.0, 0.0)

        # Setup scene
        self.server.scene.add_frame(
            "/scene_offset_left",
            show_axes=False,
            position=(-0.0, -0.3, 0.1),
            wxyz=(0.5, -0.5, -0.5, 0.5)
        )
        
        self.server.scene.add_frame(
            "/scene_offset_right",
            show_axes=False,
            position=(0.0, 0.3, 0.1),
            wxyz=(-0.5, -0.5, 0.5, 0.5),
        )
        
        self.urdf_vis = {}
        for side in ['left', 'right']:
            if side in self.urdfs:
                base_frame = self.server.scene.add_frame(f"/scene_offset_{side}/base", show_axes=False)
                self.urdf_vis[side] = ViserUrdf(self.server, self.urdfs[side], root_node_name=f"/scene_offset_{side}/base")
        
        self.server.scene.add_grid("/grid", 1.0, 1.0)
        
        # GUI
        self.play_button = self.server.gui.add_button("Play/Pause")
        self.playing = True
        self.play_button.on_click(lambda _: self.toggle_play())
        
        self.slider = self.server.gui.add_slider("Frame", min=0, max=len(self.data['world_hand_joints'])-1, step=1, initial_value=0)
        
        self.frame_idx = 0
        self.total_frames = len(self.data['world_hand_joints'])

    def toggle_play(self):
        self.playing = not self.playing

    def update(self):
        if self.playing:
            self.frame_idx = (self.frame_idx + 1) % self.total_frames
            self.slider.value = self.frame_idx
        else:
            self.frame_idx = int(self.slider.value)

        # Update Human Hand
        # world_hand_joints is (N, 21, 3)
        joints = self.data['world_hand_joints'][self.frame_idx]
        
        # We need to know which side this data belongs to. 
        # The pickle seems to be per-side based on input_node.py save_data:
        # save_path = os.path.join(output_dir, f'{side}_{idx}_{device_id}.pkl')
        # And 'mano_side' is a list [side]
        
        side = self.sides[0] # Assuming single side per file for now based on save_data
        
        self.server.scene.add_point_cloud(
            name=f"/{side}/joints",
            points=joints,
            colors=np.array([1.0, 0.0, 0.0]) if side == 'right' else np.array([0.0, 0.0, 1.0]),
            point_size=0.002,
        )
        
        points = []
        for start, end in MANO_CONNECTIONS:
            points.append(np.array([joints[start],joints[end]]))
        
        self.server.scene.add_line_segments(
            name=f"/{side}/bones",
            points=np.array(points),
            colors=np.array([1.0, 0.0, 0.0]) if side == 'right' else np.array([0.0, 0.0, 1.0]),
            line_width=1.0,
        )
        
        # Update Robot Hand
        # denormed_robot_joint_pose is (N, num_joints)
        if 'denormed_robot_joint_pose' in self.data:
            robot_pose = self.data['denormed_robot_joint_pose'][self.frame_idx]
            if side in self.urdf_vis:
                self.urdf_vis[side].update_cfg(robot_pose)

    def run(self):
        while True:
            self.update()
            time.sleep(0.033) # ~30fps

def main(file_path: str, viser_port: int=8090):
    node = PlaybackNode(file_path, viser_port)
    node.run()

if __name__ == "__main__":
    tyro.cli(main)
