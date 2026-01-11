import torch
from torch import nn
import torch.nn.functional as F
from calibration_net.policy.models.resmlp import ResMLPBlock
import numpy as np

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

class CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, input_channel, hidden_layers=[]):
        super(CNN_MLP, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        
        self.conv1 = nn.Conv1d(input_channel, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        layers = []
        in_features = 128 * input_size
        for hidden_size in hidden_layers:
            layers.append(ResMLPBlock(in_features, hidden_size))
            in_features = hidden_size
        self.fc = nn.Linear(in_features, 60)
        # self.fc = nn.Linear(in_features, output_size)
        self.fc_norm = nn.Tanh()
        layers.append(self.fc)
        layers.append(self.fc_norm)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.input_size, self.input_channel)
        
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.layers(x)
        
        
        quat = x
        quat = quat.view(-1, 15, 4)
        quat = normalize_quaternion(quat)
        so3 = quaternion_to_angle_axis(quat).contiguous()
        so3 = so3.view(x.shape[0], 15 * 3)
        return so3

    def train_forward(self, x, y):
        pred_y = self.forward(x)
        loss = F.mse_loss(pred_y, y, reduction='none')
        result = {
            'loss': loss,
            'pred_y': pred_y
        }
        return result

    def predict(self, x, noisy_act=None):
        return self.forward(x)

