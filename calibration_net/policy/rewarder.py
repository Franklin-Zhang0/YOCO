import os
import numpy as np
from calibration_net.utils import reorder_qpos, ROBOT_JOINT_ORDER, mano_recon
import quaternion
import torch
from pathlib import Path
import pyroki as pk
import yourdfpy
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

# Link mapping from URDF to 21-keypoint hand convention (same as generate_fk_ds.py)
LINK_MAPPING = [
    0,
    3, 4, 5, 6,
    8, 9, 10, 11,
    13, 14, 15, 16,
    17, 18, 19, 20,
    21, 22, 23, 24
]

# Fingertip indices in the 21-keypoint layout
TIP_INDICES = [4, 8, 12, 16, 20]

def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param keypoint_3d_array: 21 keypoints from detector.
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points_centered = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points_centered)
    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

def process_keypoints(keypoints, hand_type="right"):
    """
    Process 21 hand keypoints into MANO convention joint positions.
    
    Args:
        keypoints: (21, 3) numpy array of hand keypoints
        hand_type: "left" or "right"
    Returns:
        joint_pos: (21, 3) numpy array in MANO convention
    """
    # 1. Make all joint coordinates relative to wrist
    keypoint_3d_array = keypoints - keypoints[0:1, :]
    
    # 2. Estimate wrist rotation frame
    mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
    
    # 3. Apply coordinate system transformation
    operator2mano = OPERATOR2MANO_RIGHT if hand_type.lower() == "right" else OPERATOR2MANO_LEFT
    joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano
    
    return joint_pos

class RobotFK:
    """
    Analytic forward kinematics for robot hand using pyroki and yourdfpy.
    Computes 21 keypoints and fingertip positions from joint angles.
    """
    def __init__(self, urdf_file=None, robot_name='omni', hand_type='right'):
        """
        Initialize robot FK model.
        
        Args:
            urdf_file: Path to URDF file. If None, uses $DEX_URDF_PATH/robots/hands/{robot_name}_hand/{robot_name}_hand_{hand_type}.urdf
            robot_name: Robot name (e.g., 'omni', 'inspire', 'ro')
            hand_type: 'left' or 'right'
        """
        if urdf_file is None:
            dex_urdf_path = os.environ.get('DEX_URDF_PATH', '')
        urdf_file = Path(dex_urdf_path) / "robots" / "hands" / f"{robot_name}_hand" / f"{robot_name}_hand_{hand_type}.urdf"
        
        self.urdf_file = Path(urdf_file)
        self.robot_name = robot_name
        self.hand_type = hand_type
        
        # Load URDF
        def filename_handler(fname: str) -> str:
            base_path = self.urdf_file.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)
        
        try:
            urdf = yourdfpy.URDF.load(self.urdf_file, filename_handler=filename_handler)
            self.robot = pk.Robot.from_urdf(urdf)
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF from {self.urdf_file}: {e}")
        
        # Get actuated joint names from URDF for reordering
        self.urdf_joint_names = [urdf.robot.joints[i].name for i in urdf.actuated_joint_indices]
        
        # Precompute link mapping as numpy array
        self._link_mapping = np.array(LINK_MAPPING, dtype=np.int64)
        self._tip_indices = np.array(TIP_INDICES, dtype=np.int64)
        
    def compute_keypoints(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute 21 hand keypoints from joint angles using forward kinematics.
        
        Args:
            qpos: Joint angles in URDF actuated joint order (denormalized, in radians)
                  Shape: (n_joints,) or (batch, n_joints)
        
        Returns:
            keypoints: 21 keypoints in world frame, shape (21, 3) or (batch, 21, 3)
        """
        is_batched = qpos.ndim == 2
        if not is_batched:
            qpos = qpos[None, :]
        
        batch_size = qpos.shape[0]
        
        # Preallocate output
        all_keypoints = np.empty((batch_size, 21, 3), dtype=np.float64)
        
        for i in range(batch_size):
            # Compute FK for all links
            fk_result = self.robot.forward_kinematics(cfg=qpos[i])
            # Extract position (last 3 columns) and select links according to LINK_MAPPING
            all_keypoints[i] = fk_result[:, -3:][self._link_mapping]
        
        if not is_batched:
            return all_keypoints[0]
        
        return all_keypoints
    
    def compute_fingertips(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute fingertip positions from joint angles.
        
        Args:
            qpos: Joint angles in URDF actuated joint order (denormalized, in radians)
                  Shape: (n_joints,) or (batch, n_joints)
        
        Returns:
            fingertips: 5 fingertip positions, shape (5, 3) or (batch, 5, 3)
        """
        keypoints = self.compute_keypoints(qpos)
        
        # Extract fingertips using TIP_INDICES
        if keypoints.ndim == 2:
            return keypoints[self._tip_indices, :]
        else:
            return keypoints[:, self._tip_indices, :]
    
    def reorder_qpos_to_urdf(self, qpos: np.ndarray, source_joint_order: list) -> np.ndarray:
        """
        Reorder joint positions from source order to URDF actuated joint order.
        
        Args:
            qpos: Joint angles in source order
            source_joint_order: List of joint names in source order
        
        Returns:
            qpos_urdf: Joint angles in URDF actuated joint order
        """
        return reorder_qpos(qpos, self.urdf_joint_names, source_joint_order)

class DexRetargeter:
    def __init__(self, robot_name, hand_type, retargeting_type="dexpilot", urdf_path=None):
        from dex_retargeting.retargeting_config import RetargetingConfig
        from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
        self.robot_name = robot_name
        self.hand_type = hand_type.lower()
        
        # Load configuration
        try:
            robot_name_enum = RobotName[robot_name]
        except KeyError:
            # Fallback for names that might not be in the enum but exist in assets
            print(f"Robot name {robot_name} not found in RobotName enum. Available: {[n.name for n in RobotName]}")
            raise
            
        hand_type_enum = HandType[hand_type]
        retargeting_type_enum = RetargetingType[retargeting_type]
        
        if urdf_path is None:
            # Default path relative to the dex_retargeting package
            import dex_retargeting
            dex_path = os.path.dirname(dex_retargeting.__file__)
            urdf_path = os.path.join(dex_path, "../../assets/robots/hands")
        
        RetargetingConfig.set_default_urdf_dir(urdf_path)
        config_path = get_default_config_path(robot_name_enum, retargeting_type_enum, hand_type_enum)
        
        print(f"Loading retargeting config from: {config_path}")
        self.config = RetargetingConfig.load_from_file(config_path)
        self.config.project_dist=0.02
        self.config.escape_dist=0.03
        
        # Build retargeter
        self.retargeting = self.config.build()
        
        # Precompute reorder indices to avoid repeated list.index() calls
        self._idx_retarget_to_robot = np.array(
            [self.retargeting.joint_names.index(n) for n in ROBOT_JOINT_ORDER[self.robot_name]],
            dtype=np.int64
        )


    def retarget(self, keypoints):
        """
        Retarget 21 keypoints to robot joint angles.
        
        Args:
            keypoints: (21, 3) numpy array
        Returns:
            target_qpos: robot joint angles
        """
        self.retargeting.reset()
        joint_pos = process_keypoints(keypoints, self.hand_type)
        
        # Use retargeting logic
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices
        
        if retargeting_type == "POSITION":
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        qpos = self.retargeting.retarget(ref_value)
        
        # Use precomputed indices instead of reorder_qpos
        target_qpos = np.asarray(qpos, dtype=np.float32)[self._idx_retarget_to_robot]
        
        return target_qpos

# Worker function for parallel retargeting (must be at module level for pickling)
def _retarget_worker_init(robot_name, hand_type, retargeting_type, urdf_path):
    """Initialize a retargeter in a worker process."""
    global _worker_retargeter
    _worker_retargeter = DexRetargeter(robot_name, hand_type, retargeting_type, urdf_path)

def _retarget_worker_func(keypoints_chunk):
    """Retarget a chunk of keypoints using the worker's retargeter."""
    global _worker_retargeter
    results = []
    for keypoints in keypoints_chunk:
        results.append(_worker_retargeter.retarget(keypoints))
    return np.array(results, dtype=np.float32)

class DexRewarder:
    def __init__(self, config):
        self.config = config
        self.retargeter = DexRetargeter(config.robot_name, config.hand_type, config.retargeting_type, config.urdf_path)
        
        # Initialize FK model for computing robot fingertip positions
        self.robot_fk = RobotFK(
            urdf_file=config.urdf_file,
            robot_name=config.robot_name,
            hand_type=config.hand_type
        )
        
        # Precompute reorder indices for robot->URDF conversion
        self._idx_robot_to_urdf = np.array(
            [ROBOT_JOINT_ORDER[config.robot_name].index(n) for n in self.robot_fk.urdf_joint_names],
            dtype=np.int64
        )
        
        # Optional: LRU cache for target FK
        self._target_fk_cache = OrderedDict()
        self._target_fk_cache_size = getattr(config, 'target_fk_cache_size', 128)
        
        # Optional: multiprocessing pool for retargeting
        self._retarget_workers = getattr(config, 'retarget_workers', 0)
        self._retarget_min_chunk = getattr(config, 'retarget_min_chunk', 32)
        self._executor = None
        
        if self._retarget_workers > 0:
            # Create persistent process pool
            self._executor = ProcessPoolExecutor(
                max_workers=self._retarget_workers,
                initializer=_retarget_worker_init,
                initargs=(config.robot_name, config.hand_type, config.retargeting_type, config.urdf_path)
            )
    
    def __del__(self):
        """Clean up process pool on deletion."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
    
    def _compute_target_fk_cached(self, target_qpos_urdf: np.ndarray) -> np.ndarray:
        """
        Compute FK for target qpos with optional caching.
        
        Args:
            target_qpos_urdf: (M, n_joints) in URDF order
        Returns:
            fingertips: (M, 5, 3)
        """
        M = target_qpos_urdf.shape[0]
        result = np.empty((M, 5, 3), dtype=np.float64)
        
        if self._target_fk_cache_size <= 0:
            # No caching
            return self.robot_fk.compute_fingertips(target_qpos_urdf)
        
        # Check cache for each row
        for i in range(M):
            key = target_qpos_urdf[i].tobytes()
            if key in self._target_fk_cache:
                result[i] = self._target_fk_cache[key]
                # Move to end (LRU)
                self._target_fk_cache.move_to_end(key)
            else:
                # Compute and cache
                fk_result = self.robot_fk.compute_fingertips(target_qpos_urdf[i])
                result[i] = fk_result
                self._target_fk_cache[key] = fk_result
                
                # Evict oldest if cache is full
                if len(self._target_fk_cache) > self._target_fk_cache_size:
                    self._target_fk_cache.popitem(last=False)
        
        return result
    
    def compute_fingertip_distance_matrix(self, fingertips: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between fingertips within a hand.
        
        Args:
            fingertips: Fingertip positions, shape (B, 5, 3)
        
        Returns:
            distances: Pairwise distance matrix, shape (B, 5, 5)
        """
        # fingertips: (B, 5, 3)
        # Compute pairwise distances using broadcasting
        diff = fingertips[:, :, None, :] - fingertips[:, None, :, :]  # (B, 5, 5, 3)
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # (B, 5, 5)
        return distances
    
    def compute_reward(self, hand_keypoints: torch.Tensor, target_robot_qpos: torch.Tensor) -> torch.Tensor:
        """
        Compute reward based on multiple loss components:
        1. Fingertip position L2 distance
        2. Normed vector similarity (cosine similarity between fingertip vectors)
        3. Fingertip distance loss (5x5 pairwise distances within hand)
        
        Args:
            hand_keypoints: Hand keypoints from MANO reconstruction, shape (B, N_act, 21, 3)
            target_robot_qpos: Target robot joint positions (denormalized URDF joint angles in radians),
                              shape (B, N_act, n_joints). Must be in ROBOT_JOINT_ORDER[robot_name] order.
        
        Returns:
            reward: Combined negative loss, shape (B, N_act)
        """
        B, N_act = hand_keypoints.shape[:2]
        M = B * N_act
        hand_keypoints_np = hand_keypoints.reshape(M, 21, 3).cpu().numpy()
        target_robot_qpos_np = target_robot_qpos.reshape(M, -1).cpu().numpy()

        # Retarget hand keypoints to robot joint positions
        if self._executor is not None and M >= self._retarget_min_chunk:
            # Parallel retargeting
            chunk_size = max(1, M // self._retarget_workers)
            chunks = [hand_keypoints_np[i:i+chunk_size] for i in range(0, M, chunk_size)]
            futures = [self._executor.submit(_retarget_worker_func, chunk) for chunk in chunks]
            results = [f.result() for f in futures]
            retargeted_qpos = np.concatenate(results, axis=0)
        else:
            # Single-process retargeting with preallocated output
            retargeted_qpos = np.empty((M, len(ROBOT_JOINT_ORDER[self.config.robot_name])), dtype=np.float32)
            for i in range(M):
                retargeted_qpos[i] = self.retargeter.retarget(hand_keypoints_np[i])
        
        # Vectorized reorder from ROBOT_JOINT_ORDER to URDF order for FK
        target_qpos_urdf = target_robot_qpos_np[:, self._idx_robot_to_urdf]
        retargeted_qpos_urdf = retargeted_qpos[:, self._idx_robot_to_urdf]
        
        # Compute fingertip positions using FK with caching for targets
        target_fingertips = self._compute_target_fk_cached(target_qpos_urdf)  # (M, 5, 3)
        retargeted_fingertips = self.robot_fk.compute_fingertips(retargeted_qpos_urdf)  # (M, 5, 3)
        
        target_fingertips = np.concatenate([target_fingertips, np.zeros_like(target_fingertips[..., :])], axis=-1)       # add wrist point
        retargeted_fingertips = np.concatenate([retargeted_fingertips, np.zeros_like(retargeted_fingertips[..., :])], axis=-1)
        
        
        # 1. Compute L2 loss between fingertip positions
        fingertip_diff = retargeted_fingertips - target_fingertips  # (M, 5, 3)
        l2_loss = np.mean(fingertip_diff ** 2, axis=(1, 2))  # (M,)
        l2_reward = np.clip((0.003 - l2_loss) / 0.003, 0.0, 1.0)
        
        # 2. Compute pairwise normed vector similarity loss (5x5 cosine similarity matrix)
        # Normalize fingertip vectors
        target_fingertips_norm = target_fingertips / (np.linalg.norm(target_fingertips, axis=-1, keepdims=True) + 1e-8)  # (M, 5, 3)
        retargeted_fingertips_norm = retargeted_fingertips / (np.linalg.norm(retargeted_fingertips, axis=-1, keepdims=True) + 1e-8)  # (M, 5, 3)
        
        # Compute pairwise cosine similarity: target[i] · retargeted[j] for all i,j pairs
        # Using einsum for efficient computation: (B, i, d) x (B, j, d) -> (B, i, j)
        cosine_sim_matrix = np.einsum('bid,bjd->bij', target_fingertips_norm, retargeted_fingertips_norm)  # (M, 5, 5)
        
        # Convert to loss: 1 - cosine_similarity (ranges from 0 to 2, where 0 is perfect alignment)
        vector_similarity_matrix_reward = cosine_sim_matrix  # (M, 5, 5)
        vector_similarity_reward = np.mean(vector_similarity_matrix_reward, axis=(1, 2))  # (M,)
        
        # 3. Compute weighted fingertip distance loss (5x5 pairwise distances)
        target_distances = self.compute_fingertip_distance_matrix(target_fingertips)  # (M, 5, 5)
        retargeted_distances = self.compute_fingertip_distance_matrix(retargeted_fingertips)  # (M, 5, 5)
        
        # Compute distance-based weights: closer distances get higher weights
        # Weight = 1 / (distance + epsilon), normalized
        distance_weights = 0.05 / (target_distances + 1e-4)  # (M, 5, 5)
        
        # Set diagonal to zero (distance from fingertip to itself) - vectorized
        idx = np.arange(5)
        distance_weights[:, idx, idx] = 0.0
        
        # Normalize weights so they sum to 1 for each sample
        # weight_sum = np.sum(distance_weights, axis=(1, 2), keepdims=True)  # (M, 1, 1)
        # distance_weights = distance_weights / (weight_sum + 1e-8)  # (M, 5, 5)
        
        # Weighted L2 loss on distance matrices
        distance_diff = target_distances - retargeted_distances  # (M, 5, 5)
        weighted_distance_diff_sq = distance_weights * np.abs(distance_diff)  # (M, 5, 5)
        fingertip_distance_loss = np.sum(weighted_distance_diff_sq, axis=(1, 2))  # (M,)
        fingertip_distance_reward = np.clip((3.0 - fingertip_distance_loss) / 3.0, 0.0, 1.0)
        
        # Combine losses with weights (can be made configurable)
        l2_weight = getattr(self.config, 'l2_weight', 1.0)
        vector_sim_weight = getattr(self.config, 'vector_similarity_weight', 1.0)
        distance_weight = getattr(self.config, 'fingertip_distance_weight', 10.0)
        
        total_reward = (l2_weight * l2_reward +
                     vector_sim_weight * vector_similarity_reward +
                     distance_weight * fingertip_distance_reward)
        
        # Convert back to torch tensor and reshape
        total_reward_tensor = torch.tensor(total_reward, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype)
        total_reward_tensor = total_reward_tensor.reshape(B, N_act)
        
        # Prepare reward dictionary with individual components (reshaped to B, N_act)
        reward_dict = {
            'total_reward': total_reward_tensor,
            'l2_reward': torch.tensor(l2_reward, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype).reshape(B, N_act),
            'vector_similarity_reward': torch.tensor(vector_similarity_reward, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype).reshape(B, N_act),
            'fingertip_distance_reward': torch.tensor(fingertip_distance_reward, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype).reshape(B, N_act),
        }
        
        # Prepare loss dictionary (losses are negative rewards)
        loss_dict = {
            'l2_loss': torch.tensor(l2_loss, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype).reshape(B, N_act),
            'vector_similarity_loss': torch.tensor(1.0 - vector_similarity_reward, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype).reshape(B, N_act),
            'fingertip_distance_loss': torch.tensor(fingertip_distance_loss, device=target_robot_qpos.device, dtype=target_robot_qpos.dtype).reshape(B, N_act),
        }

        return total_reward_tensor, reward_dict, loss_dict
