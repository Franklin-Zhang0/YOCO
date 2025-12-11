import viser
import os
from pathlib import Path
import yourdfpy
from viser.extras import ViserUrdf
import numpy as np
import tyro
import time
import zmq
from calibration_net.data_collection.poses import FIX_POSE_REGISTER
from calibration_net.utils import AngleNormalizer

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')

def load_urdf(hand_type: str, side: str):
    robot_urdf_path = Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_{side}.urdf"
    
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)    
    return urdf

class RobotNode:
    def __init__(self, hand_type, viser_port, zmq_port):
        self.hand_type = hand_type
        self.viser_port = viser_port
        self.zmq_port = zmq_port
        
        self.pose_register = FIX_POSE_REGISTER[hand_type]
        self.pose_list = list(self.pose_register.keys())
        self.key_idx = 0
        self.stop_flag = False

        self.left_urdf = load_urdf(hand_type, 'left')   
        self.right_urdf = load_urdf(hand_type, 'right')

        self.angle_normalizer = AngleNormalizer(Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_left.urdf")

        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        print(f"Robot Node ZMQ REP socket bound to port {zmq_port}")

        self.server = viser.ViserServer(port=viser_port)

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
        
        self.left_base_frame = self.server.scene.add_frame("/scene_offset_left/base", show_axes=False)
        self.right_base_frame = self.server.scene.add_frame("/scene_offset_right/base", show_axes=False)
        self.left_urdf_vis = ViserUrdf(self.server, self.left_urdf, root_node_name="/scene_offset_left/base")
        self.right_urdf_vis = ViserUrdf(self.server, self.right_urdf, root_node_name="/scene_offset_right/base")
        self.server.scene.add_grid("/grid", 1.0, 1.0)

        self.left_base_frame.wxyz = np.zeros(4)
        self.left_base_frame.position = np.zeros(3)
        self.right_base_frame.wxyz = np.zeros(4)
        self.right_base_frame.position = np.zeros(3)
        
        # Add buttons to Viser GUI for local control if needed
        self.next_button = self.server.gui.add_button("Next Pose")
        self.next_button.on_click(lambda _: self.trigger_next())

        self.prev_button = self.server.gui.add_button("Prev Pose")
        self.prev_button.on_click(lambda _: self.trigger_prev())

    def handle_command(self, command):
        cmd = command.get('cmd')
        if cmd == 'next':
            self.trigger_next()
            return {'status': 'success', 'idx': self.key_idx, 'pose_name': self.pose_list[self.key_idx]}
        elif cmd == 'prev':
            self.trigger_prev()
            return {'status': 'success', 'idx': self.key_idx, 'pose_name': self.pose_list[self.key_idx]}
        elif cmd == 'get_state':
            return {'status': 'success', 'idx': self.key_idx, 'pose_name': self.pose_list[self.key_idx]}
        elif cmd == 'quit':
            self.stop_flag = True
            return {'status': 'success'}
        elif cmd == 'ping':
            return {'status': 'alive'}
        else:
            return {'status': 'error', 'message': 'Unknown command'}

    def trigger_next(self):
        if self.key_idx < len(self.pose_list) - 1:
            self.key_idx += 1
        print(f"Robot Pose: {self.key_idx}/{len(self.pose_list)}")

    def trigger_prev(self):
        if self.key_idx > 0:
            self.key_idx -= 1
        print(f"Robot Pose: {self.key_idx}/{len(self.pose_list)}")

    def run(self):
        while not self.stop_flag:
            # Check ZMQ
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
                response = self.handle_command(message)
                self.socket.send_json(response)
            except zmq.Again:
                pass
            
            if self.key_idx < len(self.pose_list):
                joints = np.array(self.pose_register[self.pose_list[self.key_idx]])
                if self.hand_type == 'inspire':
                    joints = 1 - joints[::-1]
                    joints = self.angle_normalizer.denormalize(joints)
                
                with self.server.atomic():
                    self.left_urdf_vis.update_cfg(np.array(joints))
                    self.right_urdf_vis.update_cfg(np.array(joints))
            
            time.sleep(0.03) # ~30fps update

def main(hand_type: str='inspire', 
         viser_port: int=8080,
         zmq_port: int=6000
         ):
    
    node = RobotNode(hand_type, viser_port, zmq_port)
    node.run()

if __name__ == "__main__":
    tyro.cli(main)
