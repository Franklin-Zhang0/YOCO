from open_cyber_glove.sdk import OpenCyberGlove
from inspire_hand.hand import FingerID, Register
from inspire_hand import InspireHand
from poses import FIX_POSE_REGISTER, CONTINOUS_POSE_REGISTER, CONTINOUS_CRITICAL_POSE_REGISTER
import sys
import os
import argparse
import time
import pickle
import datetime
import numpy as np
from tqdm import tqdm

from utils import initilize_glove, initialize_robot, apply_angles, walk_dict, create_dict_from_flattened, FrameRateContext


def collect_pose_data(glove, hand_type):
    glove_data = glove.get_data(hand_type=hand_type)
    glove_data = glove_data.__dict__
    for k, v in glove_data.items():
        if isinstance(v, list) or isinstance(v, tuple):
            glove_data[k] = np.array(v)
        else:
            glove_data[k] = v
            
    return glove_data

def collect_fix_pose_data(glove, robot, angles, hand_type, open_fingers=True, wait_input=True, wait_for_response=True):
    if open_fingers:
        robot.open_all_fingers()
        time.sleep(0.5)
    if robot is not None:
        apply_angles(robot, angles, wait_for_response=wait_for_response)
    if wait_input:
        input(f"Press Enter to collect glove data for {angles}...")
    glove_data = collect_pose_data(glove, hand_type)
    return glove_data

def interpolate_poses(start_data, end_data, steps=1):
    for t in np.linspace(0, 1, steps + 2):
        interpolated_data = {}
        for (key, start), (key, end) in zip(walk_dict(start_data), walk_dict(end_data)):
            if isinstance(start, np.ndarray):
                interpolated = (1 - t) * start + t * end
                interpolated_data[key] = interpolated.astype(start.dtype)
            elif isinstance(start, int):
                interpolated_data[key] = int((1 - t) * start + t * end)
            elif isinstance(start, float):
                interpolated_data[key] = (1 - t) * start + t * end
            else:
                interpolated_data[key] = start
        interpolated_data['interpolated'] = True
        yield interpolated_data
        
def apply_continous_pose(robot, pose_name):
    robot.open_all_fingers()
    time.sleep(0.5)
    start, end = pose_name.split('-')
    start_angles, end_angles = FIX_POSE_REGISTER[start], FIX_POSE_REGISTER[end]
    print(f"Applying continuous pose from {start} to {end}")
    apply_angles(robot, start_angles)
    apply_angles(robot, end_angles)
    
    
    
def data_loop(glove_sdk, left_robot, right_robot, args):
    global CONTINOUS_POSE_REGISTER
    data = {}
    if args.type == 'critical_continous':
        args.type = 'continous'
        CONTINOUS_POSE_REGISTER = CONTINOUS_CRITICAL_POSE_REGISTER
        
    try:
        for round in range(args.rounds):
            print(f"Round {round + 1}/{args.rounds}")
            
            # Collect data for discrete poses
            if args.type == 'discrete':
                for pose_name, angles in FIX_POSE_REGISTER.items():
                    data['robot'] = {}
                    print(f"Applying pose: {pose_name}")
                    glove_data, robot_data = {}, {}
                    if args.left_glove_port:
                        glove_data['left'] = collect_fix_pose_data(glove_sdk, left_robot, angles, hand_type='left')
                        robot_data['left'] = {'angles': np.array(left_robot.get_finger_angles()), 'force': np.array(left_robot.get_finger_forces()), 'temperature': np.array(left_robot.get_finger_temperatures()), 'target_angles': np.array(angles)}
                    if args.right_glove_port:
                        glove_data['right'] = collect_fix_pose_data(glove_sdk, right_robot, angles, hand_type='right')
                        robot_data['right'] = {'angles': np.array(right_robot.get_finger_angles()), 'force': np.array(right_robot.get_finger_forces()), 'temperature': np.array(right_robot.get_finger_temperatures()), 'target_angles': np.array(angles)}

                    robot_data['pose_name'] = pose_name
                    
                    data['glove'] = glove_data
                    data['robot'] = robot_data
                    data['interpolated'] = False
                yield data
                
            # Collect data for continuous poses
            elif args.type == 'continous':
                for pose_name in CONTINOUS_POSE_REGISTER:
                    start, end = pose_name.split('-')
                    start_angles, end_angles = FIX_POSE_REGISTER[start], FIX_POSE_REGISTER[end]
                    print(f"Applying continuous pose from {start} to {end}")
                    
                    if args.left_robot_port:
                        apply_angles(left_robot, start_angles, wait_for_response=False)
                    if args.right_robot_port:
                        apply_angles(right_robot, start_angles, wait_for_response=False)
                    time.sleep(0.5)
                    
                    print(f"Collecting data for {pose_name} with {args.collect_frequency} points")
                    input(f"Press Enter to start collecting data for {pose_name}...")
                    
                    last = None
                    while True:
                        if glove_sdk.left_glove:
                            current = glove_sdk.get_data(hand_type='left').tensile_data
                            if (last is not None) and (not glove_sdk.left_glove._sensors_still(current, last, threshold=10)):
                                break
                            last = current
                        
                    traj_data = []
                    static = False
                    for t in tqdm(np.linspace(0, 1, args.collect_frequency*args.sec_per_pose), desc=f"Collecting data for {pose_name}"):
                        with FrameRateContext(args.collect_frequency):
                            angles = [(1 - t) * start_angle + t * end_angle for start_angle, end_angle in zip(start_angles, end_angles)]
                            data['robot'] = {}
                            glove_data, robot_data = {}, {}
                            if args.left_glove_port:
                                glove_data['left'] = collect_fix_pose_data(glove_sdk, left_robot, angles, hand_type='left', open_fingers=False, wait_input=False, wait_for_response=False)
                                current = glove_sdk.get_data(hand_type='left').tensile_data
                            if args.right_glove_port:
                                glove_data['right'] = collect_fix_pose_data(glove_sdk, right_robot, angles, hand_type='right', open_fingers=False, wait_input=False, wait_for_response=False)
                                current = glove_sdk.get_data(hand_type='right').tensile_data

                            robot_data['pose_name'] = pose_name
                            
                            data['glove'] = glove_data
                            # data['robot'] = robot_data
                            data['interpolated'] = False
                            traj_data.append(data)
                            
                            # if static for continous 2 frames, stop collecting
                            if static:
                                if glove_sdk.left_glove._sensors_still(current, last, threshold=10):
                                    print(f"Glove sensors are static. Stopping data collection for {pose_name}.")
                                    break
                                else:
                                    static = False
                            else:
                                if glove_sdk.left_glove._sensors_still(current, last, threshold=10):
                                    static = True
                            last = current
                    
                    for i, data in enumerate(traj_data):
                        np_start_angles = np.array(start_angles, dtype=np.float32)
                        np_end_angles = np.array(end_angles, dtype=np.float32)
                        ratio = i / (len(traj_data) - 1)
                        if args.left_glove_port:
                            data['robot']['left'] = {'target_angles': np_start_angles + ratio * (np_end_angles - np_start_angles)}
                        if args.right_glove_port:
                            data['robot']['right'] = {'target_angles': np_start_angles + ratio * (np_end_angles - np_start_angles)}
                            
                        
                    if args.interpolate > 0:
                        print(f"Interpolating")
                        all_interpolated_data = []
                        for i in range(len(traj_data) - 1):
                            start_data = traj_data[i]
                            end_data = traj_data[i + 1]
                            for interpolated_data in interpolate_poses(start_data, end_data, steps=args.interpolate):
                                interpolated_data = create_dict_from_flattened(interpolated_data)
                                all_interpolated_data.append(interpolated_data)
                        traj_data.extend(all_interpolated_data)
                        traj_data.sort(key=lambda x: x['glove']['left']['timestamp'] if 'left' in x['glove'] else x['glove']['right']['timestamp'])
                    yield traj_data
                    
            elif args.type == 'glove_only':
                print("Collecting glove data only")
                input("Press Enter to start collecting glove data...")
                print("Collecting glove data...")
                try:
                    i = 0
                    while True:
                        with FrameRateContext(args.collect_frequency, verbose=False):
                            glove_data = {}
                            if args.left_glove_port:
                                glove_data['left'] = collect_pose_data(glove_sdk, hand_type='left')
                            if args.right_glove_port:
                                glove_data['right'] = collect_pose_data(glove_sdk, hand_type='right')
                            data['glove'] = glove_data
                            if i% args.collect_frequency == 0:
                                print(f"Collected {i} glove data points")
                                print("Collected glove data:", glove_data, end='\r')
                            i+=1
                            data['robot'] = {}
                            data['interpolated'] = False
                            yield data
                except KeyboardInterrupt:
                    print("Keyboard Interrupted. Stop Collecting")
            else:
                raise ValueError(f"Unknown type: {args.type}. Supported types are 'discrete', 'continous', 'glove_only'.")
                        
    except KeyboardInterrupt:
        print("Keyboard Interrupted. Stop Collecting")
        
    return
            

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_glove_port', type=str, default=None)
    parser.add_argument('--right_glove_port', type=str, default=None)
    parser.add_argument('--left_robot_port', type=str, default=None)
    parser.add_argument('--right_robot_port', type=str, default=None)
    parser.add_argument('--robot_speed', type=float, default=0.2)
    parser.add_argument('--collect_frequency', type=int, default=20)  # points collected per traj
    parser.add_argument('--interpolate', type=int, default=2, help="Whether to interpolate between poses in continuous mode")
    parser.add_argument('--sec_per_pose', type=int, default=1, help="Seconds to collect data for each pose")
    parser.add_argument('--type', type=str, choices=['critical_continous', 'continous', 'discrete', 'glove_only'], default='discrete', help="Type of data collection")
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    if os.path.exists(args.save_path):
        input(f"Data already exists in {args.save_path}. Press Enter to overwrite or Ctrl+C to exit.")
    else:
        os.makedirs(args.save_path, exist_ok=True)
        
    glove_sdk = initilize_glove(left_port=args.left_glove_port, 
                                right_port=args.right_glove_port,
                                calibrate=False)
    
    if args.left_robot_port:
        left_robot = initialize_robot(port=args.left_robot_port, speed=args.robot_speed)
        left_robot.calibrate_force_sensors()
    else:
        left_robot = None
        
    if args.right_robot_port:
        right_robot = initialize_robot(port=args.right_robot_port, speed=args.robot_speed)
        right_robot.calibrate_force_sensors()
    else:
        right_robot = None
        
    for data in data_loop(glove_sdk, left_robot, right_robot, args):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))
        with open(os.path.join(args.save_path, f'{timestamp}.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {os.path.join(args.save_path, f'{timestamp}.pkl')}")
    
    
            
if __name__ == "__main__":
    main()