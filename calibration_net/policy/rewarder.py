import os
import numpy as np
from calibration_net.utils import reorder_qpos, ROBOT_JOINT_ORDER, mano_recon
import quaternion
import torch

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

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
        target_qpos = qpos

        target_qpos = reorder_qpos(np.array(qpos), ROBOT_JOINT_ORDER[self.robot_name], self.retargeting.joint_names)
        
        return target_qpos

class DexRewarder:
    def __init__(self, config):
        self.config = config
        self.retargeter = DexRetargeter(config.robot_name, config.hand_type, config.retargeting_type, config.urdf_path)
    
    def compute_reward(self, hand_keypoints: torch.Tensor, target_robot_qpos: torch.Tensor) -> torch.Tensor:
        """
        Compute reward based on the difference between the hand pose reconstructed from MANO parameters
        and the target robot joint positions.
        """
        B, N_act = hand_keypoints.shape[:2]
        hand_keypoints = hand_keypoints.reshape(B * N_act, 21, 3)

        # Retarget hand keypoints to robot joint positions
        retargeted_qpos = []
        for i in range(hand_keypoints.shape[0]):
            retargeted_qpos.append(self.retargeter.retarget(hand_keypoints[i]))
        retargeted_qpos = np.stack(retargeted_qpos, axis=0)
        
        retargeted_qpos = torch.tensor(retargeted_qpos.reshape(B, N_act, -1), device=target_robot_qpos.device, dtype=target_robot_qpos.dtype)

        # Compute reward as negative MSE between retargeted qpos and target robot qpos
        mse_loss = ((retargeted_qpos - target_robot_qpos) ** 2)
        reward = -mse_loss

        return reward