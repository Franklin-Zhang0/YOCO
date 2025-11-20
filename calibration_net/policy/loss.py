import torch
import torch.nn.functional as F
import numpy as np


TIP_INDICIES = [4, 8, 12, 16, 20]
FINGER_KEYPOINT_INDICIES = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

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


