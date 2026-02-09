import torch
from torch import nn
import torch.nn.functional as F

from calibration_net.policy.models.mano_head import ManoHead
from calibration_net.policy.models.resmlp import ResMLPBlock

class CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, input_channel, hidden_layers=[]):
        super(CNN_MLP, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel

        self.mano_head = ManoHead()
        
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
        
        
        return self.mano_head(x)

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

