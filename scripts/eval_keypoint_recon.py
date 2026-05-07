from calibration_net.policy.model import load_model
import tyro
import h5py
import torch
import viser
import numpy as np
from calibration_net.utils import mano_recon, transform_joint3d, Side, scale_hand_fingers
from calibration_net.policy.preprocessing import get_mano_pca, MANO_PCA
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Any

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


def _make_bone_segments(joints: np.ndarray) -> np.ndarray:
    """(21,3) -> (E,2,3) for viser line segments."""
    segs = []
    for a, b in MANO_CONNECTIONS:
        segs.append(np.stack([joints[a], joints[b]], axis=0))
    return np.stack(segs, axis=0)


@dataclass
class Args:
    dataset_path: str
    model_path: str
    port: int = 8087
    
    # Playback
    fps: float = 30.0
    
    # Rendering
    show_grid: bool = True
    point_size: float = 0.002
    line_width: float = 1.0


def main(args: Args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, norm_stats, cfg = load_model(args.model_path, device=device)
    model.eval()
    data = h5py.File(args.dataset_path, 'r')
    
    # Get trajectory keys
    traj_keys = list(data.keys())
    try:
        traj_keys = sorted(traj_keys, key=lambda k: int(k))
    except Exception:
        traj_keys = sorted(traj_keys)
    
    if len(traj_keys) == 0:
        raise ValueError(f"No trajectories found in {args.dataset_path}")
    
    server = viser.ViserServer(port=args.port)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.25, 0.0, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)
    
    if args.show_grid:
        server.scene.add_grid("/grid", 1.0, 1.0)
    
    # UI controls
    traj_slider = server.gui.add_slider(
        "episode",
        min=0,
        max=max(0, len(traj_keys) - 1),
        step=1,
        initial_value=0,
        disabled=(len(traj_keys) == 1),
    )
    
    playing = server.gui.add_button("▶ Play/Pause")
    
    frame_slider = server.gui.add_slider(
        "frame",
        min=0,
        max=1,
        step=1,
        initial_value=0,
    )
    
    sample_slider = server.gui.add_slider(
        "sample", min=0, max=1, step=1, initial_value=0, visible=False
    )
    
    # Add text panel for conditioning info
    conditioning_info = server.gui.add_text("Conditioning Info", initial_value="No conditioning")
    
    # Add markdown for info display
    info_markdown = server.gui.add_markdown(
        "info",
        "",
    )
    
    # Cache for current trajectory data
    cache: dict[str, Any] = {
        "traj_idx": None,
        "sample_idx": None,
        "recon_joint3d": None,
        "keypoint_traj": None,
        "gt_keypoint": None,
        "T": None,
        "is_playing": False,
        "num_samples": 1,
    }
    
    def load_trajectory(traj_idx: int, sample_idx: int = 0) -> None:
        """Load and process a trajectory, storing results in cache."""
        traj_id = traj_keys[traj_idx]
        
        if 'pose_name' in data[traj_id].keys():
            pose_name = data[traj_id]['pose_name'][()].decode('utf-8')
            print(f"Processing trajectory {traj_id} with pose {pose_name}")
        else:
            pose_name = "N/A"
        
        keypoint_key = "world_hand_joints"
        if 'world_hand_joints_biased' in data[traj_id].keys():
            print("Using biased keypoints input")
            keypoint_key = 'world_hand_joints_biased'
        
        raw_keypoint_traj = data[traj_id][keypoint_key][:]
        if raw_keypoint_traj.ndim == 4:
            cache["num_samples"] = raw_keypoint_traj.shape[0]
            sample_slider.max = max(1, cache["num_samples"] - 1)
            sample_slider.visible = True
            keypoint_traj = raw_keypoint_traj[sample_idx]
        else:
            cache["num_samples"] = 1
            sample_slider.visible = False
            keypoint_traj = raw_keypoint_traj

        side = data[traj_id]['mano_side'][0].decode('utf-8')
        betas = torch.tensor(data[traj_id]['betas'][:]).broadcast_to((keypoint_traj.shape[0], 10))
        
        # Get norm_stats from our earlier load_model call (needs changing line 196 as well)
        has_preference_extractor = hasattr(model, 'preference_extractor') and model.preference_extractor is not None
        paired_poses = None
        if has_preference_extractor:
            biased_key = 'world_hand_joints_biased'
            true_key_candidates = ['world_hand_joints', 'world_hand_joints_true']
            true_key = next((k for k in true_key_candidates if k in data[traj_id].keys()), None)
            
            if biased_key in data[traj_id].keys() and true_key is not None:
                biased_seq_raw = data[traj_id][biased_key][:]
                if biased_seq_raw.ndim == 4:
                    biased_seq_raw = biased_seq_raw[sample_idx]
                paired_poses = {
                    biased_key: biased_seq_raw,
                    true_key: data[traj_id][true_key][:]
                }

        # Use RefineModelInferencer to make predictions
        if cache.get("inferencer") is None:
            from calibration_net.policy.refine_inferencer import RefineModelInferencer
            cache["inferencer"] = RefineModelInferencer(
                model_dir=args.model_path,
                side=side,
                device=device,
                paired_poses=paired_poses,
                fixed_noise=True,
                loaded_model=(model, norm_stats, cfg)
            )
            status = cache["inferencer"].canonicalizer_status()
            if status is not None:
                print(status)
        else:
            cache["inferencer"].set_side(side)
            cache["inferencer"].update_condition(paired_poses=paired_poses)
            
        # Update conditioning info display
        if hasattr(cache["inferencer"].model, "hyperlora"):
            conditioning_info.value = "Hyperlora condition active"
        else:
            conditioning_info.value = "No conditioning"
            
        pred_theta = cache["inferencer"].inference(keypoint_traj, batched=(keypoint_traj.ndim == 3))
        pred_theta = torch.tensor(pred_theta)
        gt_theta = data[traj_id]['mano_angles'][:]
        # biased_thetas = data[traj_id]['mano_angles_biased'][:]
        theta_mse = torch.mean((pred_theta - gt_theta)**2).cpu().numpy()
        print(f"Theta MSE Loss: {theta_mse:.6f}")
        # pred_theta = torch.tensor(biased_thetas)[0].reshape(-1,45)
        # pred_theta = torch.tensor(gt_theta)
        _, recon_joint3d = mano_recon(pred_theta, side=data[traj_id]['mano_side'][0].decode('utf-8'), betas=betas)
        
        if hasattr(model.network, 'finger_scaling'):
            with torch.no_grad():
                recon_joint3d = scale_hand_fingers(recon_joint3d, model.network.finger_scaling)
        
        recon_joint3d = transform_joint3d(recon_joint3d)
        recon_joint3d = recon_joint3d.cpu().numpy()

        keypoint_traj = transform_joint3d(keypoint_traj)

        
        # if 'world_hand_joints' in data[traj_id].keys():
        # gt_keypoint = transform_joint3d(data[traj_id]['world_hand_joints'][:])
        _, recon_gt_kepoint = mano_recon(torch.tensor(gt_theta), side=data[traj_id]['mano_side'][0].decode('utf-8'), betas=betas)
        gt_keypoint = transform_joint3d(recon_gt_kepoint).cpu().numpy()
        # else:
        #     gt_keypoint = transform_joint3d(data[traj_id][keypoint_key][:])
        recon_loss = np.mean((recon_joint3d - gt_keypoint) ** 2)
        print(f"Reconstruction MSE Loss: {recon_loss:.6f}")
        recon_mae = np.mean(np.abs(recon_joint3d - gt_keypoint))
        print(f"Reconstruction MAE Loss: {recon_mae:.6f}")
        return recon_mae, recon_loss
    
    load_trajectory(0, 0)
        
#         T = gt_keypoint.shape[0]
#         frame_slider.max = max(1, T - 1)
#         frame_slider.value = 0
        
#         # Update cache
#         cache.update({
#             "traj_idx": traj_idx,
#             "sample_idx": sample_idx,
#             "recon_joint3d": recon_joint3d,
#             "keypoint_traj": keypoint_traj,
#             "gt_keypoint": gt_keypoint,
#             "T": T,
#         })
        
#         # Update info markdown
#         info_markdown.content = (
#             f"**episode**: `{traj_id}`  \n"
#             f"**frames**: `{T}`  \n"
#             f"**side**: `{side}`  \n"
#             f"**pose**: `{pose_name}`  \n"
#             f"**recon_mse**: `{recon_loss:.6f}`"
#         )
    
#     def render_frame(traj_idx: int, frame_idx: int, sample_idx: int = 0) -> None:
#         """Render a single frame."""
#         if cache["traj_idx"] != traj_idx or cache["sample_idx"] != sample_idx:
#             load_trajectory(traj_idx, sample_idx)
        
#         recon_joint3d = cache["recon_joint3d"]
#         keypoint_traj = cache["keypoint_traj"]
#         gt_keypoint = cache["gt_keypoint"]
        
#         with server.atomic():
#             # Predicted joints (blue)
#             server.scene.add_point_cloud(
#                 name="/joints",
#                 points=recon_joint3d[frame_idx],
#                 colors=np.array([0.0, 0.0, 1.0]),
#                 point_size=float(args.point_size),
#             )
#             server.scene.add_line_segments(
#                 name="/bones",
#                 points=_make_bone_segments(recon_joint3d[frame_idx]),
#                 colors=np.array([0.0, 0.0, 1.0]),
#                 line_width=float(args.line_width),
#             )
            
#             # Input joints (red)
#             server.scene.add_point_cloud(
#                 name="/joints_input",
#                 points=keypoint_traj[frame_idx],
#                 colors=np.array([1.0, 0.0, 0.0]),
#                 point_size=float(args.point_size),
#             )
#             server.scene.add_line_segments(
#                 name="/bones_input",
#                 points=_make_bone_segments(keypoint_traj[frame_idx]),
#                 colors=np.array([1.0, 0.0, 0.0]),
#                 line_width=float(args.line_width),
#             )
            
            
#             # gt joints (yellow)
#             server.scene.add_point_cloud(
#                 name="/joints_gt",
#                 points=gt_keypoint[frame_idx],
#                 colors=np.array([0.0, 1.0, 0.0]),
#                 point_size=float(args.point_size),
#             )
#             server.scene.add_line_segments(
#                 name="/bones_gt",
#                 points=_make_bone_segments(gt_keypoint[frame_idx]),
#                 colors=np.array([0.0, 1.0, 0.0]),
#                 line_width=float(args.line_width),
#             )
    
#     @traj_slider.on_update
#     def _(_evt) -> None:
#         if getattr(_evt, "client_id", None) is None:
#             return
#         cache["is_playing"] = False
#         render_frame(int(traj_slider.value), int(frame_slider.value), int(sample_slider.value))
    
#     @frame_slider.on_update
#     def _(_evt) -> None:
#         if getattr(_evt, "client_id", None) is None:
#             return
#         render_frame(int(traj_slider.value), int(frame_slider.value), int(sample_slider.value))
        
#     @sample_slider.on_update
#     def _(_evt) -> None:
#         if getattr(_evt, "client_id", None) is None:
#             return
#         cache["is_playing"] = False
#         render_frame(int(traj_slider.value), int(frame_slider.value), int(sample_slider.value))
    
#     @playing.on_click
#     def _(_evt) -> None:
#         cache["is_playing"] = not cache["is_playing"]
    
#     # Initial render
#     render_frame(int(traj_slider.value), int(frame_slider.value), int(sample_slider.value))
    
#     server.gui.add_markdown(
#         "help",
#         """
# Use **episode** and **frame** sliders, or click **Play/Pause** to toggle autoplay.

# Legend:
# - **blue**: predicted joints (model output)
# - **red**: input keypoints (ground truth)
# """.strip(),
#     )
    
#     # Main loop
#     dt = 1.0 / max(1e-6, float(args.fps))
#     while True:
#         if cache["is_playing"]:
#             next_frame = int(frame_slider.value) + 1
#             if next_frame > int(frame_slider.max):
#                 next_frame = 0
#             frame_slider.value = next_frame
#             render_frame(int(traj_slider.value), int(frame_slider.value), int(sample_slider.value))
#         time.sleep(dt)


if __name__ == "__main__":
    main(tyro.cli(Args))