import numpy as np
import torch
from torch import nn


def normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """L2-normalize a quaternion on the last dimension."""
    return quat / quat.norm(dim=-1, keepdim=True)


def my_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """atan2 implementation that matches the historical behavior in this repo."""
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / x)
    ans = torch.where(((y > 0).float() * (x < 0).float()).bool(), ans + pi, ans)
    ans = torch.where(((y < 0).float() * (x < 0).float()).bool(), ans + pi, ans)
    return ans


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion(s) to angle-axis.

    Args:
        quaternion: (..., 4) in (w, x, y, z) format.

    Returns:
        angle_axis: (..., 3)
    """
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        my_atan2(-sin_theta, -cos_theta),
        my_atan2(sin_theta, cos_theta),
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = quaternion[..., 1:] * k.unsqueeze(-1)
    return angle_axis


class ManoHead(nn.Module):
    """MANO head: (15 quaternions) -> (45 angle-axis).

    This is a lightweight, callable module so backbone encoders can share the same
    MANO post-processing.

    Input shapes supported:
      - (B, 60) or (B, 15*4)
      - (B, 15, 4)

    Output:
      - so3: (B, 45)
      - optionally, normalized quaternions: (B, 15, 4)
    """

    def __init__(self):
        super().__init__()

    def forward(self, quat: torch.Tensor, return_quat: bool = False):
        if quat.dim() == 2:
            if quat.shape[-1] != 60:
                raise ValueError(f"Expected quat shape (B, 60); got {tuple(quat.shape)}")
            quat = quat.view(-1, 15, 4)
        elif quat.dim() == 3:
            if quat.shape[-2:] != (15, 4):
                raise ValueError(f"Expected quat shape (B, 15, 4); got {tuple(quat.shape)}")
        else:
            raise ValueError(f"quat must have shape (B, 60) or (B, 15, 4); got {tuple(quat.shape)}")

        quat = normalize_quaternion(quat)
        so3 = quaternion_to_angle_axis(quat).contiguous()
        so3 = so3.view(quat.shape[0], 15 * 3)

        if return_quat:
            return so3, quat
        return so3
