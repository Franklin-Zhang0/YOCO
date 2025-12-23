import os
from pathlib import Path
import numpy as np
import tyro
import time
import zmq
import pickle
import viser
from colorama import Fore, Style
from calibration_net.data_collection.poses import FIX_POSE_REGISTER
from calibration_net.data_collection.pose_collector import get_pose_collector
from calibration_net.teleop.vr_utils import calibration
from calibration_net.utils import AngleNormalizer
from datetime import datetime

DEX_URDF_PATH = os.environ.get('DEX_URDF_PATH', '')

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def save_data(poses, betas, output_dir, idx, pose_name, hand_type, angle_normalizer, pose_register, device_id, sides=['left','right']):
    # We need the robot pose to save alongside the human pose
    # We can reconstruct it from the pose_register and idx
    
    # Determine the number of frames from the first available side
    num_frames = 0
    for side in sides:
        if side in poses and len(poses[side]) > 0:
            num_frames = len(poses[side])
            break
            
    if num_frames == 0:
        print(f"No data to save for device {device_id}")
        return

    broadcast_pose = np.repeat(np.array(pose_register[pose_name])[np.newaxis, ...], num_frames, axis=0)
    if hand_type == 'inspire':
        broadcast_pose = 1 - broadcast_pose[:, ::-1]
        denormed_broadcast_pose = angle_normalizer.denormalize(broadcast_pose)
    else:
        denormed_broadcast_pose = broadcast_pose
        broadcast_pose = angle_normalizer.normalize(broadcast_pose)
        
    for side in sides:
        if side in poses and side in betas:
            pose = poses[side]
            save_dict = {
                'world_hand_joints': pose,
                'betas': betas[side],
                'mano_side': [side],
                'denormed_robot_joint_pose': denormed_broadcast_pose,
                'robot_type': hand_type,
                'pose_name': pose_name,
                'robot_joint_pose': broadcast_pose,
                'device_id': device_id
            }
            # Include device_id in filename to avoid collisions
            save_path = os.path.join(output_dir, f'{side}_{idx}_{device_id}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
            print(f'Saved {side} hand pose to {save_path}')

class InputNode:
    def __init__(self, output_dir, hand_type, collector_type, ip, left_glove_port, right_glove_port, calibrate, hand_model_path, glove_model_path, frame_rate, zmq_port, device_id, vuer_zmq_port, viser_port=None, camera_index=0, sides='both', vuer_port=8012):
        self.output_dir = output_dir
        self.hand_type = hand_type
        self.frame_rate = frame_rate
        self.zmq_port = zmq_port
        self.device_id = device_id
        self.viser_port = viser_port
        self.viser_server = None
        
        if sides == 'both':
            self.sides = ['left', 'right']
        else:
            self.sides = [sides]
        
        if viser_port:
            self.viser_server = viser.ViserServer(port=viser_port)
            print(f"Input Node {device_id} Viser server started on port {viser_port}")
            
            @self.viser_server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                client.camera.position = (0.0, 0.0, -0.5)
                client.camera.look_at = (0.0, 0.0, 0.0)
        
        if collector_type == 'avp' and ip == '':
            raise ValueError("AVP IP address must be provided when collector_type is 'avp'.")
        if collector_type == 'glove' and left_glove_port == '' and right_glove_port == '':
            raise ValueError("At least one glove port must be provided when collector_type is 'glove'.")

        self.pose_register = FIX_POSE_REGISTER[hand_type]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, timestamp)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        self.angle_normalizer = AngleNormalizer(Path(DEX_URDF_PATH) / "robots" / 'hands' / f'{hand_type}_hand' / f"{hand_type}_hand_left.urdf")

        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        print(f"Input Node {device_id} ZMQ REP socket bound to port {zmq_port}")

        self.stop_flag = False
        self.poses = None
        self.betas = None
        
        # Pass vuer_zmq_port if collector is vuer
        kwargs = {}
        if collector_type == 'vuer':
            kwargs['zmq_port'] = vuer_zmq_port
            kwargs['vuer_port'] = vuer_port
        if collector_type == 'mediapipe':
            kwargs['camera_index'] = camera_index

        self.pose_collector = get_pose_collector(collector_type, avp_ip=ip, frame_rate=60, left_glove_port=left_glove_port, right_glove_port=right_glove_port, hand_model_path=hand_model_path, glove_model_path=glove_model_path, calibrate=calibrate, **kwargs)
        
        self.betas = {'left':np.zeros(10), 'right':np.zeros(10)} # Dummy betas for now if calibration is skipped/mocked

    def handle_command(self, command):
        cmd = command.get('cmd')
        if cmd == 'collect':
            idx = command.get('idx')
            pose_name = command.get('pose_name')
            self.trigger_collect(idx, pose_name)
            return {'status': 'success'}
        elif cmd == 'quit':
            self.stop_flag = True
            return {'status': 'success'}
        elif cmd == 'calibrate':
            self.trigger_calibration()
            return {'status': 'success'}
        elif cmd == 'set_output_dir':
            new_dir = command.get('output_dir')
            try:
                if new_dir:
                    self.output_dir = new_dir
                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir)
                    print(f"[Device {self.device_id}] Output directory changed to: {self.output_dir}")
                    return {'status': 'success'}
                else:
                    return {'status': 'error', 'message': 'No output_dir provided'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        elif cmd == 'ping':
            return {'status': 'alive'}
        else:
            return {'status': 'error', 'message': 'Unknown command'}

    def trigger_collect(self, idx, pose_name):
        print(f'\n[Device {self.device_id}] Collecting pose {idx} ({pose_name})...')
        # We assume the user is already holding the pose, or we wait a bit?
        # The robot node handles the "wait 3 seconds" logic or the user presses "Collect" when ready.
        # Let's just collect immediately upon request.
        
        self.poses = self.pose_collector.collect(timesteps=self.frame_rate * 3, sides=self.sides)
        save_data(self.poses, self.betas, self.output_dir, idx, pose_name, self.hand_type, self.angle_normalizer, self.pose_register, self.device_id, sides=self.sides)
        print(f'[Device {self.device_id}] Pose collected.')

    def trigger_calibration(self):
        print(f'\n[Device {self.device_id}] Starting calibration...')
        print('Fitting betas, please randomly move your hands for 10 seconds...')
        poses, self.betas = calibration(self.pose_collector, n_comps=21, sides=self.sides, timesteps=self.frame_rate*10)
        
        # Save calibration data
        for side in self.sides:
            if side in poses and side in self.betas:
                save_dict = {
                    'world_hand_joints': poses[side],
                    'betas': self.betas[side],
                    'mano_side': [side],
                    # 'denormed_robot_joint_pose': np.repeat(dummy_robot_pose[np.newaxis, ...], num_frames, axis=0),
                    'robot_type': self.hand_type,
                    'pose_name': 'calibration',
                    # 'robot_joint_pose': np.repeat(dummy_robot_pose[np.newaxis, ...], num_frames, axis=0),
                    'device_id': self.device_id
                }
                save_path = os.path.join(self.output_dir, f'{side}_calibration_{self.device_id}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(save_dict, f)
                print(f'Saved {side} calibration data to {save_path}')

        print(f'[Device {self.device_id}] Calibration done.')

    def update_visualization(self, pose):
        for side in self.sides:
            if side in pose and len(pose[side]) > 0:
                # pose[side] is (21, 3)
                joints = pose[side] - pose[side][0:1, :]  # Normalize to wrist at origin
                if self.viser_server:
                    self.viser_server.scene.add_point_cloud(
                        name=f"/{side}/joints",
                        points=joints,
                        colors=np.array([1.0, 0.0, 0.0]) if side == 'right' else np.array([0.0, 0.0, 1.0]),
                        point_size=0.002,
                    )
                    
                    # Add bone lines
                    points = []
                    for start, end in MANO_CONNECTIONS:
                        points.append(np.array([joints[start],joints[end]]))
                    
                    self.viser_server.scene.add_line_segments(
                        name=f"/{side}/bones",
                        points=np.array(points),
                        colors=np.array([1.0, 0.0, 0.0]) if side == 'right' else np.array([0.0, 0.0, 1.0]),
                        line_width=1.0,
                    )

    def run(self):
        while not self.stop_flag:
            # Check ZMQ
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
                response = self.handle_command(message)
                self.socket.send_json(response)
            except zmq.Again:
                pass
            
            if self.viser_server:
                pose = self.pose_collector.get_latest_pose(sides=self.sides)
                if pose:
                    self.update_visualization(pose)
            
            time.sleep(0.01)

def main(output_dir: str, 
         hand_type: str='inspire', 
         collector_type: str='avp', 
         ip: str='', 
         left_glove_port: str='', 
         right_glove_port: str='', 
         calibrate: bool=False,
         hand_model_path: str='',
         glove_model_path: str='',
         frame_rate: int=30,
         zmq_port: int=5555,
         device_id: str='dev0',
         vuer_zmq_port: int=5555,
         viser_port: int=None,
         camera_index: int=0,
         sides: str='both',
         vuer_port: int=8012
         ):
    
    node = InputNode(output_dir, hand_type, collector_type, ip, left_glove_port, right_glove_port, calibrate, hand_model_path, glove_model_path, frame_rate, zmq_port, device_id, vuer_zmq_port, viser_port, camera_index, sides, vuer_port)
    node.run()

if __name__ == "__main__":
    tyro.cli(main)
