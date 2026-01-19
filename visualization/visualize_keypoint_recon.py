from calibration_net.policy.model import load_model
import tyro
import h5py
import torch
import viser
import numpy as np
from calibration_net.utils import mano_recon, transform_joint3d, Side, scale_hand_fingers
from calibration_net.policy.preprocessing import get_mano_pca, MANO_PCA
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_mano_pose_geometric(target_joints, rest_joints):
    """
    Computes MANO pose parameters (axis-angle) from 3D joint locations using Geometric IK.
    
    Args:
        target_joints (np.array): (21, 3) Current observed joints.
        rest_joints (np.array): (21, 3) Template MANO joints in rest pose (flat hand).
        
    Returns:
        pose_params (np.array): (16, 3) Axis-angle rotations for MANO.
                                Index 0 is global orientation.
                                Indices 1-15 are finger joint rotations.
    """
    
    # ---------------------------------------------------------
    # 1. Define Topology
    # ---------------------------------------------------------
    # Standard MANO Joint Order (usually):
    # 0: Wrist
    # 1,2,3: Index (MCP, PIP, DIP)
    # 4,5,6: Middle
    # 7,8,9: Pinky
    # 10,11,12: Ring
    # 13,14,15: Thumb
    
    # We define the kinematic tree: (Joint Index, Parent Index, Child Vector Tip Index)
    # Note: The "Child Vector Tip" is the point we use to form the bone vector.
    # For DIP joints, the tip is the fingertip (which is not in the 16 pose params but in the 21 joints).
    
    # Assuming input is strictly 21 joints in standard SMPL/MANO order:
    # 0=Wrist, 1=Index1, 2=Index2, 3=Index3, 16=IndexTip
    # 4=Mid1, 5=Mid2, 6=Mid3, 17=MidTip
    # 7=Pinky1, 8=Pinky2, 9=Pinky3, 18=PinkyTip
    # 10=Ring1, 11=Ring2, 12=Ring3, 19=RingTip
    # 13=Thumb1, 14=Thumb2, 15=Thumb3, 20=ThumbTip

    chain_definitions = [
        # Index
        (1, 0, 2), (2, 1, 3), (3, 2, 16),
        # Middle
        (4, 0, 5), (5, 4, 6), (6, 5, 17),
        # Pinky
        (7, 0, 8), (8, 7, 9), (9, 8, 18),
        # Ring
        (10, 0, 11), (11, 10, 12), (12, 11, 19),
        # Thumb
        (13, 0, 14), (14, 13, 15), (15, 14, 20)
    ]
    
    # Sort by joint index to ensure parents are processed before children
    chain_definitions.sort(key=lambda x: x[0])

    # ---------------------------------------------------------
    # 2. Helper: Construct Orientation Frame
    # ---------------------------------------------------------
    def build_local_frame(wrist, index_mcp, pinky_mcp):
        """
        Builds a coordinate frame from the palm plane.
        X: Wrist -> Index
        Y: Normal to palm (Wrist-Index-Pinky)
        Z: Cross(X, Y)
        """
        x_axis = index_mcp - wrist
        x_axis /= np.linalg.norm(x_axis)
        
        vec_diag = pinky_mcp - wrist
        vec_diag /= np.linalg.norm(vec_diag)
        
        y_axis = np.cross(x_axis, vec_diag)
        y_axis /= np.linalg.norm(y_axis)
        
        z_axis = np.cross(x_axis, y_axis)
        
        return np.column_stack([x_axis, y_axis, z_axis])

    # ---------------------------------------------------------
    # 3. Compute Global Root Rotation (Wrist)
    # ---------------------------------------------------------
    # Using Wrist(0), IndexMCP(1), PinkyMCP(7) to define the palm frame.
    
    # Rest Frame
    R_rest_frame = build_local_frame(rest_joints[0], rest_joints[1], rest_joints[7])
    
    # Target Frame
    R_target_frame = build_local_frame(target_joints[0], target_joints[1], target_joints[7])
    
    # Global Rotation = Target * Rest^T
    R_global_mat = R_target_frame @ R_rest_frame.T
    
    # Store results
    pose_params = np.zeros((16, 3))
    pose_params[0] = R.from_matrix(R_global_mat).as_rotvec()
    
    # We need to keep track of the Global Rotation of every parent to process the children
    # Initialize dictionary with root rotation
    global_rotations = {0: R_global_mat}

    # ---------------------------------------------------------
    # 4. Compute Finger Joint Rotations
    # ---------------------------------------------------------
    for j_idx, p_idx, tip_idx in chain_definitions:
        
        # A. Get vectors in Rest and Target
        # Vector from current joint to the next joint (bone vector)
        vec_rest = rest_joints[tip_idx] - rest_joints[j_idx]
        vec_rest /= np.linalg.norm(vec_rest)
        
        vec_target = target_joints[tip_idx] - target_joints[j_idx]
        vec_target /= np.linalg.norm(vec_target)
        
        # B. Transform Target Vector into Parent's Local Frame
        # We need to undo the parent's global rotation to isolate the local change
        parent_R = global_rotations[p_idx]
        
        # Invert parent rotation (transpose for rotation matrix)
        vec_target_in_parent = parent_R.T @ vec_target
        
        # C. Calculate Rotation to align Rest Vector to Target (in parent frame)
        # We use cross product to find the axis and dot product for the angle
        cross = np.cross(vec_rest, vec_target_in_parent)
        dot = np.dot(vec_rest, vec_target_in_parent)
        
        # Clip dot product to handle numerical errors
        dot = np.clip(dot, -1.0, 1.0)
        
        angle = np.arccos(dot)
        
        # Handle small angles (avoid division by zero)
        if np.linalg.norm(cross) < 1e-6:
            # Vectors are already aligned
            r_local_vec = np.zeros(3)
            r_local_mat = np.eye(3)
        else:
            axis = cross / np.linalg.norm(cross)
            r_local_vec = axis * angle
            r_local_mat = R.from_rotvec(r_local_vec).as_matrix()
            
        pose_params[j_idx] = r_local_vec
        
        # D. Update Global Rotation for this joint
        # This becomes the parent rotation for the next joint in the chain
        global_rotations[j_idx] = parent_R @ r_local_mat

    return pose_params

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def main(dataset_path: str, model_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = load_model(model_path, device=device)
    # model2, _, _ =load_model("/data/250010203/franklin/project/hand_teleop/Calibration_net/expdata/test_combined_cnn_mlp_leaky_global_mixer/test_combined_cnn_mlp_leaky_global_mixer_20260111_134713/best" ,device=device)
    data = h5py.File(dataset_path, 'r')
    
    
    server = viser.ViserServer(port='8087')
    
    server.scene.add_grid("/grid", 1.0, 1.0)
    for traj_id in data.keys():
        if 'pose_name' in data[traj_id].keys():
            pose_name = data[traj_id]['pose_name'][()].decode('utf-8')
            print(f"Processing trajectory {traj_id} with pose {pose_name}")
            
        keypoint_key = "world_hand_joints"
        if 'world_hand_joints_biased' in data[traj_id].keys():
            print("Using biased keypoints input")
            keypoint_key='world_hand_joints_biased'
            
        keypoint_traj = data[traj_id][keypoint_key][:]  # (T, 21, 3)
        keypoint_traj = transform_joint3d(keypoint_traj)
        side = data[traj_id]['mano_side'][0].decode('utf-8')
        betas = torch.tensor(data[traj_id]['betas'][:]).broadcast_to((keypoint_traj.shape[0],10))
        
        pred_theta = model.predict(keypoint_traj, side='right').to('cpu')  # (T, 45)
        # pred_theta = torch.tensor(get_mano_pca(data[traj_id]['mano_angles'][:], side=side), dtype=torch.float32)  # (T, 45)
        # pred_theta = torch.tensor(data[traj_id]['pca_mano_pose'][:], dtype=torch.float32)  # (T, 45)
        
        _, recon_joint3d = mano_recon(pred_theta, side=data[traj_id]['mano_side'][0].decode('utf-8'))  # (T, 21, 3)
        
        if hasattr(model.network, 'finger_scaling'):
            with torch.no_grad():
                recon_joint3d = scale_hand_fingers(recon_joint3d, model.network.finger_scaling)
        
        recon_joint3d = transform_joint3d(recon_joint3d)
        
        recon_joint3d = recon_joint3d.cpu().numpy()
        recon_loss = np.mean((recon_joint3d - keypoint_traj)**2)
        print(f"Reconstruction MSE Loss: {recon_loss:.6f}")
        
        for frame_idx in range(keypoint_traj.shape[0]):
            with server.atomic():
                
                #pred joints
                server.scene.add_point_cloud(
                    name=f"/joints",
                    points=recon_joint3d[frame_idx],
                    colors=np.array([0.0, 0.0, 1.0]), #blue
                    point_size=0.002,
                )
                lines = []
                for start, end in MANO_CONNECTIONS:
                    lines.append(np.array([recon_joint3d[frame_idx][start],recon_joint3d[frame_idx][end]]))
                
                server.scene.add_line_segments(
                    name=f"/bones",
                    points=np.array(lines),
                    colors=np.array([0.0, 0.0, 1.0]) ,
                    line_width=1.0,
                )
                
                # ground truth joints
                server.scene.add_point_cloud(
                    name=f"/joints_gt",
                    points=keypoint_traj[frame_idx],
                    colors=np.array([1.0, 0.0, 0.0]), #red
                    point_size=0.002,
                )
                lines_gt = []
                for start, end in MANO_CONNECTIONS:
                    lines_gt.append(np.array([keypoint_traj[frame_idx][start],keypoint_traj[frame_idx][end]]))
                
                server.scene.add_line_segments(
                    name=f"/bones_gt",
                    points=np.array(lines_gt),
                    colors=np.array([1.0, 0.0, 0.0]),
                    line_width=1.0,
                )
            time.sleep(0.1)
        
        
if __name__ == "__main__":
    tyro.cli(main)