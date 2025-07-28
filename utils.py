from open_cyber_glove.sdk import OpenCyberGlove
from inspire_hand.hand import FingerID, Register
from inspire_hand import InspireHand
import numpy as np
import time
from typing import Union

def initilize_glove(left_port=None, right_port=None, model_path=None, calibrate=True):
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

def apply_angles(robot, angles:list[float]|dict[FingerID, float], wait_for_response: bool = True):
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
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Key '{flat_key}' not found in dictionary.")
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
