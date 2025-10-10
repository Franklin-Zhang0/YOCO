from inspire_hand.hand import FingerID, Register
from inspire_hand import InspireHand
import argparse
from tqdm import tqdm
from calibration_net.policy.model import load_model
import numpy as np
import torch
from calibration_net.policy.preprocessing import build_preprocessing

from calibration_net.utils import initilize_glove, initialize_robot, apply_angles, FrameRateContext
from calibration_net.data_collection.pose_collector import PoseCollector
from calibration_net.teleop.vr_utils import calibration, initialize_fitter
import os
from contextlib import nullcontext

MANO_PATH = os.environ.get("MANO_PATH", "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sides = []

finger_mapping_inspire_to_glove = {
    FingerID.LITTLE: 12,
    FingerID.RING: 8,
    FingerID.MIDDLE: 4,
    FingerID.INDEX: 0,
    FingerID.THUMB_BEND: 19,
    FingerID.THUMB_ROTATE: 16,
}

class Ensembler:
    def __init__(self, horizon=5, tao=0.5):
        self.horizon = horizon
        self.tao = tao
        self.history = []
    
    def add(self, new_data):
        self.history.append(new_data)
        if len(self.history) > self.horizon:
            self.history.pop(0)

    def predict(self, x):
        self.add(x)
        weight = np.array([self.tao**i for i in range(len(self.history)-1, -1, -1)])
        weighted_sum = sum([w * h for w, h in zip(weight, self.history)])
        return weighted_sum / weight.sum()

def main():
    print("starting plain teleop")
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_robot_port', type=str, default=None)
    parser.add_argument('--right_robot_port', type=str, default=None)
    parser.add_argument('--side', type=str, default='left', choices=['left', 'right', 'both'])
    parser.add_argument('--robot_speed', type=float, default=0.4)
    parser.add_argument('--control_frequency', type=int, default=30)  # Hz
    parser.add_argument('--calibration', action='store_true', help="If set, enter calibration mode to collect data for fitting the MANO model")
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    global sides
    sides = [args.side] if args.side in ['left', 'right'] else ['left', 'right']
    
    robot = {}
    if args.left_robot_port:
        robot['left'] = initialize_robot(port=args.left_robot_port, speed=args.robot_speed)
    else:
        robot['left'] = None
        
    if args.right_robot_port:
        robot['right'] = initialize_robot(port=args.right_robot_port, speed=args.robot_speed)
    else:
        robot['right'] = None

    model, norm_stats, cfg = load_model(args.model_dir, device=device)
    n_comps = cfg['model'].get('input_size')

    pose_collector = PoseCollector(args.control_frequency)

    if args.calibration:
        betas = calibration(pose_collector, n_comps, sides)
    else:
        betas = {'left': torch.zeros((10,), dtype=torch.float32).to(device), 
                 'right': torch.zeros((10,), dtype=torch.float32).to(device)}
    print(f"Using betas {betas}")

    hyper_lora_ctx = {}
    hyper_lora_ctx['left'] = model.hyperlora.use(torch.tensor(betas['left']).unsqueeze(0))
    hyper_lora_ctx['right'] = model.hyperlora.use(torch.tensor(betas['right']).unsqueeze(0))


    mano_fitter = initialize_fitter(n_comps, sides)

    process_x = cfg['dataset']['preprocessing_x']
    process_y = cfg['dataset']['preprocessing_y']
    process_x_fn = build_preprocessing(process_x, 'x', norm_stats=norm_stats)
    process_y_fn = build_preprocessing(process_y, 'y', norm_stats=norm_stats)

    input("Press enter to start teleop")
    prev_results = {'left': None, 'right': None}
    is_first_frame = True
    ensembler = {'left': Ensembler(horizon=5, tao=0.5), 'right': Ensembler(horizon=5, tao=0.5)}
    fixed_noise = torch.randn(1, cfg['model']['output_size']).to(device)
    
    try:
        while True:
            with FrameRateContext(
                    frame_rate=args.control_frequency, verbose=False
                ) as frame_manager:
                # Update poses with timeout protection
                # collect pose
                joint_pose = pose_collector.collect(1)

                if not hasattr(model, 'iknet'):
                    max_steps = 200 if is_first_frame else 50
                    results = {side: mano_fitter[side].fit_single_frame(joint_pose[side][-1], prev_results[side], betas[side], max_steps) for side in sides}
                    # breakpoint()
                    hand_pose = {side: ensembler[side].predict(results[side]['hand_pose']) for side in sides}
                    # print(hand_pose['left'][:5])
                    input = {side: hand_pose for side in sides}
                    prev_results = results
                else:
                    input = {side: joint_pose[side][-1] for side in sides}

                for side in sides:
                    with hyper_lora_ctx[side]:
                        processed_input = torch.tensor(process_x_fn(input[side]), dtype=torch.float32).to(device)
                        pred = model.predict(processed_input.unsqueeze(0), noisy_act=fixed_noise).cpu().numpy().squeeze(0)
                        pred = process_y_fn(pred)
                        if side == 'left':
                            print(f"{side}: {pred[:4]}")
                        if robot[side] is not None:
                            apply_angles(robot[side], pred, wait_for_response=False)
                is_first_frame = False

    except KeyboardInterrupt:
        print("Stopping teleoperation...")
    finally:
        pass
        
    if args.left_robot_port:
        robot['left'].close()
    if args.right_robot_port:
        robot['right'].close()

if __name__ == "__main__":
    main()