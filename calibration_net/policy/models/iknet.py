from .resmlp import ResMLPBlock
from torch import nn
import torch
import numpy as np
import os
from calibration_net.utils import Side
from calibration_net.policy.models.resmlp import ResMLPBlock

MANO_PATH = os.environ.get('MANO_PATH', '')

def normalize_quaternion(quat):
    return quat / quat.norm(dim=-1, keepdim=True)
    
def my_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / x)
    ans = torch.where(((y > 0).float() * (x < 0).float()).bool(), ans + pi, ans)
    ans = torch.where(((y < 0).float() * (x < 0).float()).bool(), ans + pi, ans)
    return ans
    
def quaternion_to_angle_axis(quaternion):
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(cos_theta < 0.0, my_atan2(-sin_theta, -cos_theta), my_atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = quaternion[..., 1:] * k.unsqueeze(2)

    return angle_axis

class IKNet(nn.Module):
    def __init__(
        self,
        njoints=21,
        hidden_layers=[256, 256, 256],
        output_pca=False,
        n_comps=21,
    ):
        super(IKNet, self).__init__()
        self.njoints = njoints
        self.output_pca = output_pca
        self.pca_comps = n_comps
        if self.output_pca:
            from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
            self.coeff = {}
            for side in ['left', 'right']:
                mano_param = ready_arguments(os.path.join(MANO_PATH, f'MANO_{side.upper()}.pkl'))
                component = mano_param['hands_components']
                self.coeff[side] = np.linalg.inv(component)[:self.pca_comps]
                self.coeff[side] = torch.from_numpy(self.coeff[side]).float()
            self.stack_coeff = torch.stack([self.coeff['left'], self.coeff['right']], dim=0)
        in_neurons = 3 * njoints
        out_neurons = 15 * 4  # 15 quats -- no global orientation
        neurons = [in_neurons] + hidden_layers

        invk_layers = []
        for layer_idx, (inps, outs) in enumerate(zip(neurons[:-1], neurons[1:])):
            invk_layers.append(ResMLPBlock(inps, outs))

        invk_layers.append(ResMLPBlock(neurons[-1], out_neurons))

        self.invk_layers = nn.ModuleList(invk_layers)

    def forward(self, joint, side, ik_y=None):
        B, N, C = joint.shape
        x = joint = joint.contiguous().view(-1, self.njoints * 3)
        
        for layers in self.invk_layers:
            x = layers(x)
        quat = x
        # quat = self.invk_layers(joint)
        quat = quat.view(-1, 15, 4)
        quat = normalize_quaternion(quat)
        so3 = quaternion_to_angle_axis(quat).contiguous()
        so3 = so3.view(B, 15 * 3)

        if self.output_pca:
            assert side is not None
            coeff_idx = side == Side.LEFT.value
            coeff = self.stack_coeff.to(so3.device, so3.dtype)[coeff_idx.long(), :, :] # B x n_comps x 45
            pca = coeff @ so3.unsqueeze(-1)  # B x n_comps x 1
            pca = pca.squeeze(-1).contiguous()  # B  x n_comps

        if ik_y is not None:
            ik_loss = nn.MSELoss()(pca, ik_y)
            return pca, quat, ik_loss
        
        else:
            return pca, quat