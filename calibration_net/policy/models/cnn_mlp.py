import torch
from torch import nn
import torch.nn.functional as F
from calibration_net.policy.models.resmlp import ResMLPBlock

class CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, input_channel, hidden_layers=[]):
        super(CNN_MLP, self).__init__()
        self.calib_block = nn.Sequential(nn.Conv1d(
                in_channels=input_channel,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
                nn.GELU(),
                nn.Conv1d(
                    in_channels=128,
                    out_channels=1,
                    kernel_size=5,
                    stride=1,
                    padding=2
                )
            )
        layers = []
        in_features = input_size
        for hidden_size in hidden_layers:
            layers.append(ResMLPBlock(in_features, hidden_size))
            in_features = hidden_size
        self.fc = nn.Linear(in_features, output_size)
        layers.append(self.fc)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-2)  # add channel dim
        x = self.calib_block(x).squeeze(-2)
        x = self.layers(x)
        return x

    def train_forward(self, x, y):
        pred_y = self.forward(x)
        loss = F.mse_loss(pred_y, y)
        result = {
            'loss': loss,
            'pred_y': pred_y
        }
        return result

    def predict(self, x, noisy_act=None):
        return self.forward(x)

