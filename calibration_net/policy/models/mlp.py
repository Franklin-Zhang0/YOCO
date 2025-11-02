import torch
from torch import nn
import torch.nn.functional as F
from calibration_net.policy.models.resmlp import ResMLPBlock

class MLP(nn.Module):
    def __init__(self, input_size, output_size, input_channel, hidden_layers=[]):
        super(MLP, self).__init__()
        layers = []
        self.input_features = input_size * input_channel
        self.input_size = input_size
        self.input_channel = input_channel
        
        in_features = self.input_features
        for hidden_size in hidden_layers:
            layers.append(ResMLPBlock(in_features, hidden_size))
            in_features = hidden_size
        self.fc = nn.Linear(in_features, output_size)
        self.fc_norm = nn.Tanh()
        layers.append(self.fc)
        layers.append(self.fc_norm)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_features)
        x = self.layers(x)
        return x

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

