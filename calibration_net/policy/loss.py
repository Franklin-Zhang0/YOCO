import torch
import torch.nn.functional as F
import numpy as np


TIP_INDICIES = [4, 8, 12, 16, 20]
FINGER_KEYPOINT_INDICIES = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
DEFAULT_PINCH_PAIR_INDICIES = [(4, 8), (4, 12), (4, 16), (4, 20)]
DEFAULT_PINCH_THRESHOLD_M = 0.005

def get_pose_weight_scale():
    weight_scale = torch.zeros(21, 21)
    weight_scale[TIP_INDICIES, :] = 1.0
    weight_scale[:, TIP_INDICIES] = 1.0
    weight_scale[TIP_INDICIES, TIP_INDICIES] = 5.0
    
    # no pos loss between joints in the same finger
    for finger in FINGER_KEYPOINT_INDICIES:
        for joint in finger:
            weight_scale[joint, finger] = 0
            weight_scale[finger, joint] = 0
    return weight_scale

def get_alignment_loss(points:torch.Tensor, side:torch.Tensor, mask:torch.Tensor, ik_model:torch.nn.Module, fk_model:torch.nn.Module) -> torch.Tensor:
    def ik_forward(x):
        return ik_model(x, side=side)
    points = points.float().cuda()
    embedded_point = fk_model(ik_forward(points))
    
    finger_keypoint_mask = torch.ones(points.shape[:2]).float().cuda()
    for i, finger_inds in enumerate(FINGER_KEYPOINT_INDICIES):
        finger_keypoint_mask[:, finger_inds] *= mask[:, i].unsqueeze(-1)

    embedded_finger_tip = embedded_point[:, TIP_INDICIES, :]
    mano_finger_tip = points[:, TIP_INDICIES, :]

    mano_pos_scale = ((torch.norm(mano_finger_tip, dim=-1) * mask).sum(-1) / (mask.sum(-1) + 1e-6)).mean()
    robot_pos_scale = ((torch.norm(embedded_finger_tip, dim=-1) * mask).sum(-1) / (mask.sum(-1) + 1e-6)).mean()

    # append wrist point at origin
    embedded_finger_tip = torch.cat(
        [embedded_finger_tip, torch.zeros_like(embedded_finger_tip[:, :1, :])], dim=1
    )
    mano_finger_tip = torch.cat(
        [mano_finger_tip, torch.zeros_like(mano_finger_tip[:, :1, :])], dim=1
    )

    # NxN grid of relative positions.
    # delta_mano = mano_finger_tip[:, :, None] - mano_finger_tip[:, None, :]
    # delta_robot = embedded_finger_tip[:, :, None] - embedded_finger_tip[:, None, :]
    
    delta_mano = points[:, :, None] - points[:, None, :]
    delta_robot = embedded_point[:, :, None] - embedded_point[:, None, :]

    # Vector regularization.

    residual_position_delta = (
        (delta_mano / mano_pos_scale - delta_robot / robot_pos_scale)
        * (1 - torch.eye(delta_mano.shape[1])[None, ..., None]).to(device=points.device)
    )
    residual_position_delta = (residual_position_delta**2 * 1/(mano_pos_scale + 0.01)).sum(-1)
    # take the upper triangle only to avoid duplicate pairs
    residual_position_delta = torch.triu(residual_position_delta, diagonal=1)
    # residual_position_delta[:, :5] *= mask.unsqueeze(-1).unsqueeze(-1)
    residual_position_delta[:] *= finger_keypoint_mask.unsqueeze(-1)
    weight_scale = get_pose_weight_scale().to(points.device)
    residual_position_delta = torch.clip(residual_position_delta, max=10.0)
    residual_position_delta *= weight_scale
    


    # Vector angle regularization.
    delta_mano_normalized = delta_mano / torch.norm(
        delta_mano + 1e-6, dim=-1, keepdim=True
    )
    delta_robot_normalized = delta_robot / torch.norm(
        delta_robot + 1e-6, dim=-1, keepdim=True
    )
    residual_angle_delta = 1 - (delta_mano_normalized * delta_robot_normalized).sum(
        dim=-1
    )
    residual_angle_delta = (
        residual_angle_delta
        * (1 - torch.eye(residual_angle_delta.shape[1])).to(device=points.device)
    )
    residual_angle_delta = torch.triu(residual_angle_delta, diagonal=1)
    # residual_angle_delta[:, :5] *= mask.unsqueeze(-1)
    residual_angle_delta[:] *= finger_keypoint_mask.unsqueeze(-1)
    
    loss = residual_position_delta.mean() + residual_angle_delta.mean()
    return loss


def get_geort_loss(points:torch.Tensor, side:torch.Tensor, ik_model:torch.nn.Module, fk_model:torch.nn.Module, robot_point_cloud:torch.Tensor) -> dict:
    loss_weights = {'curvature':0.0, 'pinch':0.0, 'direction':0.0, 'chamfer':100.0}
    points = points.float().cuda()
    side = side.float().cuda()
    def ik_forward(x):
        return ik_model(x, side=side)
    
    def fk_forward(x):
        pos = fk_model(x)[:, TIP_INDICIES, :]
        return pos
    
    embedded_point = fk_forward(ik_forward(points))

    loss_dict = {}
    loss_dict['curvature'] = curvature_loss(points, embedded_point, ik_forward, fk_forward)
    loss_dict['direction'] = direction_loss(points, embedded_point, ik_forward, fk_forward)
    loss_dict['pinch'] = pinch_loss(points, embedded_point)

    selected_idx = np.random.randint(0, robot_point_cloud.shape[1], len(points))
    target = torch.from_numpy(robot_point_cloud[:, selected_idx, :]).permute(1, 0, 2).float().cuda()
    loss_dict['chamfer'] = 0

    n_keypoints = embedded_point.shape[1]
    full_embedded_point = fk_model(ik_forward(points))
    for i in range(n_keypoints):
        loss_dict['chamfer'] += chamfer_distance(full_embedded_point[:, i, :].unsqueeze(0), target[:, i, :].unsqueeze(0))

    loss = 0.0
    for key in loss_dict.keys():
        loss += loss_weights[key] * loss_dict[key]
        
    loss_dict['loss'] = loss

    return loss_dict



def curvature_loss(points:torch.Tensor, embedded_point:torch.Tensor, ik_model:torch.nn.Module, fk_model:torch.nn.Module) -> torch.Tensor:
    direction = torch.zeros_like(points)
    rands = F.normalize(torch.randn_like(points), dim=-1, p=2)
    direction[:, TIP_INDICIES, :] = rands[:, TIP_INDICIES, :]

    scale = 0.002
    delta1 = direction * scale
    point_delta_1p = points + delta1 
    point_delta_1n = points - delta1 

    embedded_point_p = fk_model(ik_model(point_delta_1p))
    embedded_point_n = fk_model(ik_model(point_delta_1n))
    curvature_loss = ((embedded_point_p + embedded_point_n - 2 * embedded_point) ** 2).mean()
    return curvature_loss

def direction_loss(points:torch.Tensor, embedded_point:torch.Tensor, ik_model:torch.nn.Module, fk_model:torch.nn.Module):
    rands = F.normalize(torch.randn_like(points), dim=-1, p=2)
    direction = torch.zeros_like(points)
    direction[:, TIP_INDICIES, :] = rands[:, TIP_INDICIES, :]
    scale = 0.01 + torch.rand(points.size(0)).cuda().unsqueeze(-1).unsqueeze(-1) * 0.01
    point_delta = points + direction * scale

    joint_delta = ik_model(point_delta)
    embedded_point_delta = fk_model(joint_delta)

    B = point_delta.size(0)
    d1 = (point_delta.view(B, -1, 3)[:, TIP_INDICIES] - points.view(B, -1, 3)[:, TIP_INDICIES]).reshape(-1, 3)
    d2 = (embedded_point_delta - embedded_point).reshape(-1, 3)
    direction_loss = -(((F.normalize(d1, dim=-1, p=2, eps=1e-6) * F.normalize(d2, dim=-1, p=2, eps=1e-6)).sum(-1))).mean()
    
    return direction_loss

def pinch_loss(point:torch.Tensor, embedded_point:torch.Tensor):
    B = point.size(0)
    point = point.view(B, -1, 3)[:, TIP_INDICIES, :]
    n_finger = point.size(1)
    pinch_loss = 0

    for i in range(1, n_finger):
        distance = point[:, i, ...] - point[:, 0, ...]
        mask = (torch.norm(distance, dim=-1) < 0.007).float()
        e_distance = ((embedded_point[:, i, ...] - embedded_point[:, 0, ...]) ** 2).sum(dim=-1)
        pinch_loss += (mask * e_distance).mean() / (mask.sum() + 1e-7) * point.size(0)

    return pinch_loss

def chamfer_distance(input_points, target_points):
    """
    Args:
    - input_points (torch.Tensor): Input point cloud tensor of shape [B, N, 3].
    - target_points (torch.Tensor): Target point cloud tensor of shape [B, M, 3].
    
    Returns:
    - chamfer_dist (torch.Tensor): Chamfer distance.
    """
    B, N, _ = input_points.size()
    _, M, _ = target_points.size()
    
    input_points = input_points.clone()
    target_points = target_points.clone()
    input_points[..., 1] = input_points[..., 1] 
    target_points[..., 1] = target_points[..., 1]

    input_points = input_points.unsqueeze(2)    # [B, N, 1, 3]
    target_points = target_points.unsqueeze(1)  # [B, 1, M, 3]
    
    input_points_repeat = input_points.repeat(1, 1, M, 1)    # [B, N, M, 3]
    target_points_repeat = target_points.repeat(1, N, 1, 1)  # [B, N, M, 3]
    

    dist_matrix = torch.sum((input_points_repeat - target_points_repeat)**2, dim=-1)  # [B, N, M]
    
    min_dist_a, _ = torch.min(dist_matrix, dim=2)  # [B, N]
    min_dist_b, _ = torch.min(dist_matrix, dim=1)  # [B, M]
    
    chamfer_dist = torch.mean(min_dist_a, dim=1) + torch.mean(min_dist_b, dim=1)
    
    return chamfer_dist.mean()


def compute_relative_distances(joints, tip_indices=TIP_INDICIES):
    """Compute pairwise distances between fingertips within one hand.
    
    Args:
        joints: (B, 21, 3) joint positions
        tip_indices: list of fingertip joint indices
        
    Returns:
        distances: (B, C(5,2)) = (B, 10) pairwise distances
    """
    tips = joints[:, tip_indices, :]  # (B, 5, 3)
    
    # Compute all pairwise distances between 5 fingertips using cdist
    dist_matrix = torch.cdist(tips, tips)  # (B, 5, 5)
    # Extract upper triangle (excluding diagonal)
    idx = torch.triu_indices(5, 5, offset=1, device=joints.device)
    distances = dist_matrix[:, idx[0], idx[1]]  # (B, 10)
    return distances


def compute_relative_vectors(joints, tip_indices=TIP_INDICIES, base_index=0):
    """Compute normalized relative vectors:
    1. Base (wrist) to each fingertip: 5 vectors
    2. Each fingertip to each other fingertip: C(5,2) = 10 vectors
    Total: 15 normalized vectors
    
    Args:
        joints: (B, 21, 3) joint positions
        tip_indices: list of fingertip joint indices
        base_index: wrist joint index
        
    Returns:
        vectors: (B, 15, 3) normalized direction vectors
    """
    tips = joints[:, tip_indices, :]  # (B, 5, 3)
    base = joints[:, base_index:base_index+1, :]  # (B, 1, 3)
    
    vectors_list = []
    
    # 1. Base (wrist) to each fingertip: 5 vectors
    for i in range(len(tip_indices)):
        vec = tips[:, i, :] - base[:, 0, :]  # (B, 3)
        vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-6)  # Normalize
        vectors_list.append(vec)
    
    # 2. Each fingertip to each other fingertip: C(5,2) = 10 vectors
    n_tips = len(tip_indices)
    for i in range(n_tips):
        for j in range(i + 1, n_tips):
            vec = tips[:, i, :] - tips[:, j, :]  # (B, 3)
            vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-6)  # Normalize
            vectors_list.append(vec)
    
    vectors = torch.stack(vectors_list, dim=1)  # (B, 15, 3)
    return vectors


def relative_distance_loss(pred_joints, gt_joints, tip_indices=TIP_INDICIES):
    """MSE between predicted and GT relative fingertip distances.
    
    Args:
        pred_joints: (B, 21, 3) predicted joint positions from MANO FK
        gt_joints: (B, 21, 3) GT joint positions from MANO FK on GT theta
        
    Returns:
        loss: scalar MSE loss
    """
    TIP_SIGMA_M = 0.02
    pred_distances = compute_relative_distances(pred_joints, tip_indices)
    gt_distances = compute_relative_distances(gt_joints, tip_indices) - 0.005
    gt_distances = torch.clamp(gt_distances, min=0.0)
    
    square_error = (pred_distances - gt_distances)**2
    
    # scale = 1 / (torch.abs(gt_distances) + 1e-6)
    scale = torch.exp(-(gt_distances / TIP_SIGMA_M).abs())
    
    loss = (square_error * scale).mean(dim=0).sum()
    return loss


def relative_vector_loss(pred_joints, gt_joints, tip_indices=TIP_INDICIES, base_index=0):
    """MSE between predicted and GT relative vectors (normalized).
    
    Args:
        pred_joints: (B, 21, 3) predicted joint positions
        gt_joints: (B, 21, 3) GT joint positions
        
    Returns:
        loss: scalar MSE loss
    """
    pred_vectors = compute_relative_vectors(pred_joints, tip_indices, base_index)
    gt_vectors = compute_relative_vectors(gt_joints, tip_indices, base_index)
    
    # MSE loss on normalized direction vectors
    loss = F.mse_loss(pred_vectors, gt_vectors)
    return loss


def pinch_l1_loss(
    pred_joints,
    gt_joints,
    pinch_pairs=DEFAULT_PINCH_PAIR_INDICIES,
    threshold_m=DEFAULT_PINCH_THRESHOLD_M,
):
    """Mean predicted fingertip separation on GT pinch frames.

    Mirrors eval/evaluate_calibration.py's pinch_error_l1_m: a pair is counted
    when the GT fingertip distance is below threshold_m, then the loss is the
    L1 distance between that predicted fingertip pair.
    """
    pinch_values = []
    for left_idx, right_idx in pinch_pairs:
        gt_delta = gt_joints[:, left_idx] - gt_joints[:, right_idx]
        mask = torch.norm(gt_delta, dim=-1) < threshold_m
        if not mask.any():
            continue

        pred_delta = pred_joints[:, left_idx] - pred_joints[:, right_idx]
        pred_l1 = pred_delta[mask].abs().sum(dim=-1)
        pinch_values.append(pred_l1)

    if not pinch_values:
        return pred_joints.sum() * 0.0

    return torch.cat(pinch_values).mean()


def pinch_l1_loss(
    pred_joints,
    gt_joints,
    pinch_pairs=DEFAULT_PINCH_PAIR_INDICIES,
    threshold_m=DEFAULT_PINCH_THRESHOLD_M,
):
    """Mean predicted fingertip separation on GT pinch frames.

    Mirrors eval/evaluate_calibration.py's pinch_error_l1_m: a pair is counted
    when the GT fingertip distance is below threshold_m, then the loss is the
    L1 distance between that predicted fingertip pair.
    """
    pinch_values = []
    for left_idx, right_idx in pinch_pairs:
        gt_delta = gt_joints[:, left_idx] - gt_joints[:, right_idx]
        mask = torch.norm(gt_delta, dim=-1) < threshold_m
        if not mask.any():
            continue

        pred_delta = pred_joints[:, left_idx] - pred_joints[:, right_idx]
        pred_l1 = pred_delta[mask].abs().sum(dim=-1)
        pinch_values.append(pred_l1)

    if not pinch_values:
        return pred_joints.sum() * 0.0

    return torch.cat(pinch_values).mean()


def batch_similarity_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable similarity alignment from source joints to target joints."""
    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have the same shape, got {source.shape} "
            f"and {target.shape}."
        )

    source_mean = source.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.transpose(1, 2) @ target_centered
    u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.ones(
        source.shape[0],
        3,
        device=source.device,
        dtype=source.dtype,
    )
    correction[:, -1] = torch.sign(torch.det(u @ vh)).clamp(min=0) * 2 - 1
    correction_matrix = torch.diag_embed(correction)
    rotation = u @ correction_matrix @ vh

    source_var = (source_centered ** 2).sum(dim=(1, 2)).clamp_min(1e-8)
    scale = (singular_values * correction).sum(dim=1) / source_var

    return scale[:, None, None] * (source_centered @ rotation) + target_mean


def pa_keypoint_loss(pred_joints, gt_joints):
    """MSE after Procrustes-style similarity alignment."""
    aligned_pred = batch_similarity_align(pred_joints, gt_joints)
    return F.mse_loss(aligned_pred, gt_joints)


def pa_mpjpe_loss(pred_joints, gt_joints):
    """Mean joint L2 after Procrustes-style similarity alignment."""
    aligned_pred = batch_similarity_align(pred_joints, gt_joints)
    return torch.linalg.norm(aligned_pred - gt_joints, dim=-1).mean()


def get_refine_composite_loss(
    pred_theta,
    gt_theta,
    mse_loss,
    side,
    loss_weights,
    betas=None,
):
    """Compute refine_composite loss: MSE + geometric MANO-joint losses.
    
    Args:
        pred_theta: (B, 45) predicted MANO pose parameters (with gradients)
        gt_theta: (B, 45) GT MANO pose parameters
        mse_loss: scalar, pre-computed MSE loss on MANO parameters
        side: (B,) hand side tensor (1=left, -1=right)
        loss_weights: dict with keys like 'mse', 'rel_distance', 'rel_vector',
            'global_keypoint', and 'pinch' and their weights
        loss_weights: dict with keys like 'mse', 'rel_distance', 'rel_vector',
            'global_keypoint', and 'pinch' and their weights
        
    Returns:
        loss: dict with keys 'loss' and the unweighted component losses
    """
    from calibration_net.utils import mano_recon
    
    loss = {}
    loss['mse'] = mse_loss
    
    B = pred_theta.shape[0]
    if betas is not None:
        betas = betas.to(device=pred_theta.device, dtype=pred_theta.dtype)
    
    # Split batch by side and compute MANO joints
    left_mask = (side == 1)
    right_mask = (side == -1)
    
    pred_joints = torch.zeros(B, 21, 3, device=pred_theta.device, dtype=pred_theta.dtype)
    gt_joints = torch.zeros(B, 21, 3, device=gt_theta.device, dtype=gt_theta.dtype)
    
    # Process left hand (gradients flow through predicted joints)
    if left_mask.sum() > 0:
        left_idx = torch.where(left_mask)[0]
        left_pred_theta = pred_theta[left_idx]
        left_gt_theta = gt_theta[left_idx]
        
        left_betas = betas[left_idx] if betas is not None else None
        # Get predicted joints with gradients
        _, pred_left_joints = mano_recon(
            left_pred_theta,
            side='left',
            betas=left_betas,
            apply_left_shapedirs_fix=True,
        )
        # Center at wrist
        pred_left_joints = pred_left_joints - pred_left_joints[:, 0:1, :]
        pred_joints[left_idx] = pred_left_joints
        
        # Get GT joints without gradients
        with torch.no_grad():
            _, gt_left_joints = mano_recon(
                left_gt_theta,
                side='left',
                betas=left_betas,
                apply_left_shapedirs_fix=True,
            )
            gt_left_joints = gt_left_joints - gt_left_joints[:, 0:1, :]
        gt_joints[left_idx] = gt_left_joints
    
    # Process right hand (gradients flow through predicted joints)
    if right_mask.sum() > 0:
        right_idx = torch.where(right_mask)[0]
        right_pred_theta = pred_theta[right_idx]
        right_gt_theta = gt_theta[right_idx]
        
        right_betas = betas[right_idx] if betas is not None else None
        # Get predicted joints with gradients
        _, pred_right_joints = mano_recon(
            right_pred_theta,
            side='right',
            betas=right_betas,
            apply_left_shapedirs_fix=True,
        )
        # Center at wrist
        pred_right_joints = pred_right_joints - pred_right_joints[:, 0:1, :]
        pred_joints[right_idx] = pred_right_joints
        
        # Get GT joints without gradients
        with torch.no_grad():
            _, gt_right_joints = mano_recon(
                right_gt_theta,
                side='right',
                betas=right_betas,
                apply_left_shapedirs_fix=True,
            )
            gt_right_joints = gt_right_joints - gt_right_joints[:, 0:1, :]
        gt_joints[right_idx] = gt_right_joints
    
    # Compute relative losses
    loss['rel_distance'] = relative_distance_loss(pred_joints, gt_joints)
    loss['rel_vector'] = relative_vector_loss(pred_joints, gt_joints)
    loss['global_keypoint'] = F.mse_loss(pred_joints, gt_joints)
    loss['pa_keypoint'] = pa_keypoint_loss(pred_joints, gt_joints)
    loss['pa_mpjpe'] = pa_mpjpe_loss(pred_joints, gt_joints)
    loss['pinch'] = pinch_l1_loss(pred_joints, gt_joints)
    
    # Combine all losses using loss_weights
    total_loss = 0.0
    for key in loss_weights:
        if key in loss:
            total_loss += loss[key] * loss_weights[key]
    loss['loss'] = total_loss
    
    return loss
