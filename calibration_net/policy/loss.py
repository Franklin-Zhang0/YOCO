import torch
import torch.nn.functional as F
import numpy as np


TIP_INDICIES = [4, 8, 12, 16, 20]

def get_geort_loss(points:torch.Tensor, side:torch.Tensor, ik_model:torch.nn.Module, fk_model:torch.nn.Module, loss_weights:dict, robot_point_cloud:torch.Tensor) -> dict:
    points = points.float().cuda()
    side = side.float().cuda()
    def ik_forward(x):
        return ik_model(x, side=side)
    
    def fk_forward(x):
        pos = fk_model(x)
        # pos[..., 0] *= -side[:, None]
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
    for i in range(n_keypoints):
        loss_dict['chamfer'] += chamfer_distance(embedded_point[:, i, :].unsqueeze(0), target[:, i, :].unsqueeze(0))

    loss = 0.0
    for key in loss_dict.keys():
        loss += loss_weights[key] * loss_dict[key]
        
    loss_dict['loss'] = loss

    return loss_dict



def curvature_loss(points:torch.Tensor, embedded_point:torch.Tensor, ik_model:torch.nn.Module, fk_model:torch.nn.Module) -> torch.Tensor:
    embedded_point = fk_model(ik_model(points))
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

    for i in range(1):
        for j in range(i + 1, n_finger):
            distance = point[:, i, ...] - point[:, j, ...]
            mask = (torch.norm(distance, dim=-1) < 0.007).float()
            e_distance = ((embedded_point[:, i, ...] - embedded_point[:, j, ...]) ** 2).sum(dim=-1)
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


