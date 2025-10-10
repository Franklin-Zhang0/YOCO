import numpy as np
import time
from typing import Union
from enum import Enum

class Side(Enum):
    LEFT = 1
    RIGHT = -1

def initilize_glove(left_port=None, right_port=None, model_path=None, calibrate=True):
    from open_cyber_glove.sdk import OpenCyberGlove
    sdk = OpenCyberGlove(left_port=left_port, 
                         right_port=right_port,
                         model_path=model_path)
    
    # Check if SDK is properly initialized
    sdk.start()
    if calibrate:
        print("Calibrating glove...")
        sdk.calibrate()
    return sdk

def initialize_robot(port=None, speed=1.0):
    from inspire_hand import InspireHand
    print(f"Initializing robot on port {port}")
    robot = InspireHand(port=port, baudrate=115200)
    robot.open()
    # Reset the robot
    input("Press Enter to reset the robot...")
    robot.reset()
    robot.set_all_finger_speeds(int(speed*1000))
    time.sleep(2)
    robot.open_all_fingers()
    time.sleep(2)
    return robot

def apply_angles(robot, angles, wait_for_response: bool = True):
    from inspire_hand.hand import FingerID, Register
    """
    Apply angles to the robot.
    Inputs:
        robot: InspireHand instance
        angles: list of float angles for each finger, range [0,1]
    """
    robot_angle = []
    if isinstance(angles, list):
        robot_angle = [int(np.clip(angle, 0, 1.0) * 1000) for angle in angles]
    elif isinstance(angles, dict):
        for finger_id, angle in angles.items():
            # print(finger_id.name, angle, end="  ")
            angle = int(np.clip(angle, 0, 1) * 1000)
            if finger_id == FingerID.THUMB_BEND:
                angle = 1000 - angle
            robot_angle.append(angle)
    else:
        raise TypeError(f"Type of the angle is not valid: {type(angles)}. It should be either a dict or a list.")
    robot.modbus.write_multiple_registers(Register.ANGLE_SET, robot_angle, wait_for_response=wait_for_response)
    
def walk_dict(d: dict, prefix: str = ''):
    """
    Walk through a dictionary and yield key-value pairs with full path.
    Args:
        d (dict): Dictionary to walk through.
        prefix (str): Current prefix for keys.
    Yields:
        tuple: (full_key, value)
    """
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from walk_dict(v, full_key)
        else:
            yield full_key, v
            
def index_dict(input_dict:dict, flat_key:str):
    """
    Index a dictionary using a flat key.
    Args:
        input_dict (dict): Dictionary to index.
        flat_key (str): Key in dot notation.
    Returns:
        value: Value corresponding to the flat key.
    """
    keys = flat_key.split('.')
    value = input_dict
    for key in keys:
        try:
            value = value[key]
        except KeyError:
            raise KeyError(f"Key '{flat_key}' not found in {input_dict}.")
    return value
            
def create_dict_from_flattened(flat_dict: dict) -> dict:
    """
    Create a nested dictionary from a flattened dictionary.
    Args:
        flat_dict (dict): Flattened dictionary with keys in dot notation.
    Returns:
        dict: Nested dictionary.
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split('.')
        d = nested_dict
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested_dict

import time



class FrameRateContext:
    def __init__(self, frame_rate, verbose=False):
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self.verbose = verbose
        self.start_time = time.monotonic()
        self.iter = 0
        # Compensation for sleep overhead
        self.sleep_overhead = 0.000  # Initial estimate, will be adjusted
        self.alpha = 0.0  # Learning rate for sleep overhead adjustment

    def __enter__(self):
        self.frame_start_time = time.monotonic()
        self.this_iter_end_time = self.start_time + (self.iter + 1) * self.dt
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = time.monotonic()

        if now < self.this_iter_end_time:
            target_sleep = self.this_iter_end_time - now
            # Compensate for sleep overhead
            adjusted_sleep = max(0, target_sleep - self.sleep_overhead)

            if self.verbose:
                print(
                    f"Target sleep: {target_sleep:.6f}, Adjusted sleep: {adjusted_sleep:.6f}"
                )

            sleep_start = time.monotonic()
            time.sleep(adjusted_sleep)
            actual_sleep = time.monotonic() - sleep_start

            # Update sleep overhead estimate
            if actual_sleep > 0:
                # measured_overhead = actual_sleep - adjusted_sleep
                self.sleep_overhead = (
                    1 - self.alpha
                ) * self.sleep_overhead + self.alpha * actual_sleep
        else:
            if self.verbose:
                print("Frame took longer than expected")

        self.iter += 1

        if self.verbose:
            frame_time = time.monotonic() - self.frame_start_time
            current_fps = 1.0 / frame_time
            print("iteration:", self.iter)
            print(
                f"Frame time: {frame_time:.6f}s, Current FPS: {current_fps:.1f}, Sleep overhead: {self.sleep_overhead:.6f}"
            )



def build_axes_from_joints3d(joints_xyz: np.ndarray):
    """
    Batched wrist–palm frame from 3-D hand joints.

    Parameters
    ----------
    joints_xyz : ndarray with shape (..., 21, 3)  (any leading batch dims)
        Landmark order must match MANO / MediaPipe so that
        0=wrist, 5=index-MCP, 9=middle-MCP, 13=ring-MCP.

    Returns
    -------
    T      : ndarray, shape (..., 4, 4)   – forward transform (world→hand-base)
    T_inv  : ndarray, shape (..., 4, 4)   – inverse (hand-base→world)
    """
    joints_xyz = np.asarray(joints_xyz)
    if joints_xyz.shape[-2:] != (21, 3):
        raise ValueError("expected (..., 21, 3) input, but got shape " + str(joints_xyz.shape))

    eps = 1e-8

    z_axis = joints_xyz[..., 9, :] - joints_xyz[..., 0, :]          # middle_MCP − wrist
    z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True) + eps

    y_aux  = joints_xyz[..., 13, :] - joints_xyz[..., 5, :]         # ring_MCP − index_MCP
    y_aux /= np.linalg.norm(y_aux, axis=-1, keepdims=True) + eps

    x_axis = np.cross(y_aux, z_axis)
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True) + eps

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1, keepdims=True) + eps

    # ------------------------------------------------------------------ #
    # rotation & translation                                             #
    # ------------------------------------------------------------------ #
    # stack as columns → (..., 3, 3)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)

    t = joints_xyz[..., 0, :]  # (..., 3)

    batch_shape = R.shape[:-2]

    T = np.broadcast_to(np.eye(4, dtype=R.dtype), batch_shape + (4, 4)).copy()
    T[..., :3, :3] = R
    T[..., :3,  3] = t

    # inverse: Rᵀ and −Rᵀ·t
    R_T = np.swapaxes(R, -2, -1)
    t_inv = -(R_T @ t[..., None])[..., 0]

    T_inv = np.broadcast_to(np.eye(4, dtype=R.dtype), batch_shape + (4, 4)).copy()
    T_inv[..., :3, :3] = R_T
    T_inv[..., :3, 3]  = t_inv

    return T, T_inv

def apply_transformation(joint_xyz: np.ndarray, transformation_matrix: np.ndarray):
    """
    Apply a transformation matrix to a joint pose.
    Args:
        joint_pose (np.ndarray): Joint pose of shape (N, 3).
        transformation_matrix (np.ndarray): Transformation matrix of shape (4, 4).
    Returns:
        np.ndarray: Transformed joint pose of shape (N, 3).
    """
    joint_xyz = np.concatenate((joint_xyz, np.ones((*joint_xyz.shape[:-1], 1))), axis=-1)
    transformed_joint_xyz = joint_xyz @ np.swapaxes(transformation_matrix, -2, -1)
    return transformed_joint_xyz[..., :3]

def transform_joint3d(joint_xyz: np.ndarray):
    tran, inv_tran = build_axes_from_joints3d(joint_xyz)
    transformed_joint_xyz = apply_transformation(joint_xyz, inv_tran)
    return transformed_joint_xyz
    
class AngleNormalizer():
    def __init__(self, urdf_path):
        import yourdfpy
        from pathlib import Path
        robot_urdf_path = Path(urdf_path)

        def filename_handler(fname: str) -> str:
            base_path = robot_urdf_path.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)
        
        urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler) 
        act_indices = urdf.actuated_joint_indices
        joints = [urdf.robot.joints[i] for i in act_indices]
        self.joint_upper_limits = np.array([joint.limit.upper for joint in joints])
        self.joint_lower_limits = np.array([joint.limit.lower for joint in joints])

    def normalize(self, angles: Union[np.ndarray, list]):
        angles = np.asarray(angles)
        if angles.ndim == 1:
            assert angles.shape[0] == len(self.joint_upper_limits), f"Expected {len(self.joint_upper_limits)} angles, but got {angles.shape[0]}"
            normalized_angles = (angles - self.joint_lower_limits) / (self.joint_upper_limits - self.joint_lower_limits)
        elif angles.ndim == 2:
            assert angles.shape[1] == len(self.joint_upper_limits), f"Expected {len(self.joint_upper_limits)} angles, but got {angles.shape[1]}"
            normalized_angles = (angles - self.joint_lower_limits[None, :]) / (self.joint_upper_limits - self.joint_lower_limits)[None, :]
        else:
            raise ValueError(f"Expected angles to be 1D or 2D array, but got {angles.ndim}D array")
        normalized_angles = np.clip(normalized_angles, 0.0, 1.0)
        return normalized_angles
    
    def denormalize(self, normalized_angles: Union[np.ndarray, list]):
        normalized_angles = np.asarray(normalized_angles)
        if normalized_angles.ndim == 1:
            assert normalized_angles.shape[0] == len(self.joint_upper_limits), f"Expected {len(self.joint_upper_limits)} angles, but got {normalized_angles.shape[0]}"
            angles = normalized_angles * (self.joint_upper_limits - self.joint_lower_limits) + self.joint_lower_limits
        elif normalized_angles.ndim == 2:
            assert normalized_angles.shape[1] == len(self.joint_upper_limits), f"Expected {len(self.joint_upper_limits)} angles, but got {normalized_angles.shape[1]}"
            angles = normalized_angles * (self.joint_upper_limits - self.joint_lower_limits)[None, :] + self.joint_lower_limits[None, :]
        else:
            raise ValueError(f"Expected angles to be 1D or 2D array, but got {normalized_angles.ndim}D array")
        return angles
