import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from manopth.manolayer import ManoLayer
import zmq
import tqdm
from calibration_net.utils import FrameRateContext

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------
# Utilities
# ----------------------------
MEDIAPIPE_21_ORDER = list(range(21))  # MP order = [0..20]; MANO-21 commonly uses the same order.
FINGERTIP_IDXS = [4, 8, 12, 16, 20]   # MP fingertip indices
KEEP16_FROM_21 = [i for i in range(21) if i not in FINGERTIP_IDXS]  # drop tips to match MANO-16

WEBXR_TO_SMPLX = [
     0,   # wrist                                          → 0
     1, 2, 3, 4,                                           # thumb_1-4
     6, 7, 8, 9,                                           # index_1-4  (skip 5 = metacarpal)
    11,12,13,14,                                           # middle_1-4 (skip 10)
    16,17,18,19,                                           # ring_1-4   (skip 15)
    21,22,23,24                                            # pinky_1-4  (skip 20)
]

JOINT_WEIGHT = [
    1.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
]

def webxr_to_smplx(joints_xyz: np.ndarray) -> np.ndarray:
    """
    Convert a (25, 3) WebXR-ordered hand to the (21, 3) MANO ordering.

    Parameters
    ----------
    joints_xyz : np.ndarray
        Hand joint coordinates in WebXR order, shape (25, 3).

    Returns
    -------
    np.ndarray
        Same coordinates reordered for MANO, shape (21, 3).
    """
    if joints_xyz.shape[0] != 25 or joints_xyz.shape[1] != 3:
        raise ValueError("Expected (25,3) array in WebXR joint order.")
    return joints_xyz[WEBXR_TO_SMPLX]

class BaseCollector():
    def __init__(self):
        raise NotImplementedError()
    def collect(self, timesteps:int=1, sides:list[str]=['left','right']):
        raise NotImplementedError()

class PoseCollector(BaseCollector):
    def __init__(self, frame_rate=60):
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE,   1) 
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect("tcp://localhost:5555")
        self.frame_rate=frame_rate
        self.frame_context = FrameRateContext(frame_rate=frame_rate)

    def collect(self, timesteps:int=1, sides:list[str]=['left','right']):
        assert sides in [['left'], ['right'], ['left', 'right']]
        hand_pose={
            'left':[],
            'right':[]
        }
        while True:
            with self.frame_context:
                stop_flag = True
                for side in sides:
                    if len(hand_pose[side]) < timesteps:
                        stop_flag = False
                if stop_flag:
                    break

                msg = self._sub.recv_json()
                obj = msg['data']
                sides = ['left', 'right']
                for side in sides:
                    matrix = obj[side]
                    if not isinstance(matrix, list):
                        print(f"Invalid joint data for {side} hand, skipping")
                        continue
                    joints = np.asarray(matrix, dtype=np.float32) 
                    if joints.shape != (400,):
                        print(f"Bad joint array shape for {side} hand, skipping")
                        continue
                    joints = joints.reshape(25, 4, 4)
                    joints_xyz = joints[:, 3, :3]  # Extract XYZ positions from the 4x4 matrices
                    joints_xyz = webxr_to_smplx(joints_xyz)  # Reorder to MANO format
                    hand_pose[side].append(joints_xyz)
                print(f"Number of frames collected, left:{len(hand_pose['left'])}/{timesteps}, right:{len(hand_pose['right'])}/{timesteps}", end='\r')

        for side in sides:
            hand_pose[side] = np.stack(hand_pose[side], axis=0)
        return hand_pose


class GlovePoseCollector(BaseCollector):
    def __init__(self, hand_model_path, left_glove_port=None, right_glove_port=None, calibrate=False):
        from calibration_net.utils import initilize_glove
        from open_cyber_glove.visualizer import HandVisualizer
        glove_sdk = initilize_glove(left_port=left_glove_port, 
                                    right_port=right_glove_port,
                                    calibrate=calibrate)
        visualizer = HandVisualizer(hand_model_path)
        self.glove_sdk = glove_sdk
        self.visualizer = visualizer

    def collect(self, timesteps:int=1, sides:list[str]=['left','right']):
        assert sides in [['left'], ['right'], ['left', 'right']]
        hand_pose={
            'left':[],
            'right':[]
        }
        while True:
            stop_flag = True
            for side in sides:
                if len(hand_pose[side]) < timesteps:
                    stop_flag = False
            if stop_flag:
                break

            sides = ['left', 'right']
            for side in sides:
                if side == 'left':
                    if self.glove_sdk.left_hand is None:
                        continue
                    angles = self.glove_sdk.left_hand.get_angles(method='model')
                    joints = self.visualizer.get_joints(angles, hand_type='left')
                else:
                    if self.glove_sdk.right_hand is None:
                        continue
                    angles = self.glove_sdk.right_hand.get_angles(method='model')
                    joints = self.visualizer.get_joints(angles, hand_type='right')
                hand_pose[side].append(joints)
            print(f"Number of frames collected, left:{len(hand_pose['left'])}/{timesteps}, right:{len(hand_pose['right'])}/{timesteps}", end='\r')

        for side in sides:
            hand_pose[side] = np.stack(hand_pose[side], axis=0)
        return hand_pose
    
class AvpPoseCollector(BaseCollector):
    def __init__(self, avp_ip, frame_rate=60):
        from avp_stream import VisionProStreamer
        self.streamer = VisionProStreamer(ip=avp_ip, record=True)
        self.frame_rate = frame_rate
        self.frame_rate_context = FrameRateContext(frame_rate=frame_rate)

    def collect(self, timesteps:int=1, sides:list[str]=['left','right']):
        assert sides in [['left'], ['right'], ['left', 'right']]
        hand_pose={
            'left':[],
            'right':[]
        }
        while True:
            with self.frame_rate_context:
                stop_flag = True
                for side in sides:
                    if len(hand_pose[side]) < timesteps:
                        stop_flag = False
                if stop_flag:
                    break

                data = self.streamer.latest

                sides = ['left', 'right']
                for side in sides:
                    finger_data = data[f'{side}_fingers']
                    wrist_data = data[f'{side}_wrist']
                    local_finger_joints_xyz = finger_data[:, :3, 3]
                    global_wrist_xyz = wrist_data[0, :3, 3]
                    global_joints_xyz = local_finger_joints_xyz + global_wrist_xyz
                    global_smplx_joints_xyz = webxr_to_smplx(global_joints_xyz)
                    hand_pose[side].append(global_smplx_joints_xyz)
                print(f"Number of frames collected, left:{len(hand_pose['left'])}/{timesteps}, right:{len(hand_pose['right'])}/{timesteps}", end='\r')

        for side in sides:
            hand_pose[side] = np.stack(hand_pose[side], axis=0)
        return hand_pose

def get_pose_collector(collector_type, **kwargs) -> BaseCollector:
    """
    Factory function to create a pose collector based on the specified type.
    Parameters
    ----------
    collector_type : str
        Type of pose collector to create. Options are 'vuer', 'avp', 'glove'.
    **kwargs
        Additional keyword arguments to pass to the collector's constructor.
        For 'vuer': frame_rate (int)
        For 'avp': avp_ip (str), frame_rate (int)
        For 'glove': hand_model_path (str), left_glove_port (str or None), right_glove_port (str or None), calibrate (bool)
    Returns
    """
    if collector_type == 'vuer':
        return PoseCollector(kwargs.get('frame_rate', 60))
    
    elif collector_type == 'avp':
        assert 'avp_ip' in kwargs, "avp_ip must be provided for AvpPoseCollector"
        return AvpPoseCollector(kwargs['avp_ip'], kwargs.get('frame_rate', 60))
    
    elif collector_type == 'glove':
        return GlovePoseCollector(**kwargs)
    else:
        raise ValueError(f"Unknown collector type: {collector_type}")
