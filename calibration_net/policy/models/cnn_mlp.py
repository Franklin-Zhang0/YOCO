import torch
from torch import nn

class CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, input_channel, hidden_layers=[]):
        super(MLP, self).__init__()
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
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.GELU())
            in_features = hidden_size
        self.fc = nn.Linear(in_features, output_size)
        layers.append(self.fc)
    
    def forward(self, x):
        x = self.calib_block(x).squeeze(-2)
        x = self.fc(x)
        return x

