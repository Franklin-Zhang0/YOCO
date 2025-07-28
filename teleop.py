from open_cyber_glove.sdk import OpenCyberGlove
from inspire_hand.hand import FingerID, Register
from inspire_hand import InspireHand
import argparse
from tqdm import tqdm
from model import load_model
import numpy as np
import torch

from utils import initilize_glove, initialize_robot, apply_angles, FrameRateContext


finger_mapping_inspire_to_glove = {
    FingerID.LITTLE: 12,
    FingerID.RING: 8,
    FingerID.MIDDLE: 4,
    FingerID.INDEX: 0,
    FingerID.THUMB_BEND: 19,
    FingerID.THUMB_ROTATE: 16,
}

def calibrate_glove(sdk, hand_type='left', samples_min_max=1000):
    tensile_data = []
    print(f"[{hand_type}] Calibration Pose 1: Make a fist and open your hand, multiple times. Press Enter to continue...")
    input("")
    for i in tqdm(range(samples_min_max), desc=f"[{hand_type}] min/max calibration"):
        data = sdk.get_data(hand_type=hand_type).tensile_data
        tensile_data.append(data)
    tensile_data = np.array(tensile_data)
    mean = np.mean(tensile_data, axis=0)
    std = np.std(tensile_data, axis=0)
    max = np.max(tensile_data, axis=0)
    min = np.min(tensile_data, axis=0)
    norm_stat = {
        'x': {
            'mean': mean,
            'std': std,
            'min': min,
            'max': max
        }
    }
    return norm_stat

def normalize_angles(angles, statistics):
    normalized_angles = {}
    for finger, angle in angles.items():
        if finger in statistics:
            min_angle = statistics[finger]['min']
            max_angle = statistics[finger]['max']
            normalized_angle = (angle - min_angle) / (max_angle - min_angle)  # Scale to 0-1
            normalized_angles[finger] = normalized_angle
        else:
            print(f"Warning: No statistics available for {finger}. Using raw angle.")
            normalized_angles[finger] = angle
    return normalized_angles


def main():
    print("starting plain teleop")
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_port', type=str, default=None)
    parser.add_argument('--right_port', type=str, default=None)
    parser.add_argument('--left_robot_port', type=str, default=None)
    parser.add_argument('--right_robot_port', type=str, default=None)
    parser.add_argument('--robot_speed', type=float, default=0.4)
    parser.add_argument('--control_frequency', type=int, default=30)  # Hz
    parser.add_argument('--calibration', action='store_true', help="Use new calibration method")
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()

    glove_sdk = initilize_glove(left_port=args.left_port, 
                                right_port=args.right_port,
                                calibrate=False
                                )
    
    if args.left_robot_port:
        left_robot = initialize_robot(port=args.left_robot_port, speed=args.robot_speed)
        
    if args.right_robot_port:
        right_robot = initialize_robot(port=args.right_robot_port, speed=args.robot_speed)
    
    
    model, norm_stats, _ = load_model(args.model_dir, device='cpu')
    
    if args.calibration:
        if glove_sdk.left_glove is not None:
            new_norm_stats = calibrate_glove(glove_sdk, hand_type='left')
            print(f"Left glove angle statistics: {new_norm_stats}")
        if glove_sdk.right_glove is not None:
            new_norm_stats = calibrate_glove(glove_sdk, hand_type='right')
            print(f"Right glove angle statistics: {new_norm_stats}")
        norm_stats['x'] = new_norm_stats['x']
    
    input("Press enter to start teleop")
    try:
        while True:
            with FrameRateContext(
                    frame_rate=args.control_frequency, verbose=False
                ) as frame_manager:
                # Update poses with timeout protection
                if glove_sdk.left_glove is not None:
                    data = glove_sdk.get_data(hand_type='left').tensile_data
                    # print(data)
                    
                    angles = model.predict(np.array(data), norm_stats, norm_type='meanstd').tolist()
                    # print(angles)
                    apply_angles(left_robot, angles, wait_for_response=False)
                if glove_sdk.right_glove is not None:
                    data = glove_sdk.get_angles(hand_type='right', method='model')
                    angles = model.predict(np.array(data), norm_stats, norm_type='meanstd').tolist()
                    # print(angles)
                    apply_angles(right_robot, angles, wait_for_response=False)

    except KeyboardInterrupt:
        print("Stopping teleoperation...")
    finally:
        glove_sdk.stop()
        
    if args.left_robot_port:
        left_robot.close()
    if args.right_robot_port:
        right_robot.close()

if __name__ == "__main__":
    main()