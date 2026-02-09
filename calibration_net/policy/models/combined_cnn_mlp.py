import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from calibration_net.policy.models.mano_head import ManoHead
from calibration_net.policy.models.resmlp import ResMLPBlock

Mediapipe2Mano = [
    [5, 6, 7, 8],    # Index
    [9, 10, 11, 12], # Middle
    [17, 18, 19, 20],# Pinky
    [13, 14, 15, 16],# Ring
    [1, 2, 3, 4]     # Thumb
]

class Combined_CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, input_channel, hidden_layers=[]):
        super(Combined_CNN_MLP, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel

        self.mano_head = ManoHead()
        
        self.finger_encoder = nn.ModuleList()
        self.mano_regressor = nn.ModuleList()
        
        conv_channel = 64
        encoder_out_dim = conv_channel * 5
        
        
        for i in range(5):
        
            conv1 = nn.Conv1d(input_channel, conv_channel//2, kernel_size=3, padding=1)
            bn1 = nn.BatchNorm1d(conv_channel//2)
            conv2 = nn.Conv1d(conv_channel//2, conv_channel, kernel_size=3, padding=1)
            bn2 = nn.BatchNorm1d(conv_channel)
            conv3 = nn.Conv1d(conv_channel, conv_channel, kernel_size=3, padding=1)
            bn3 = nn.BatchNorm1d(conv_channel)
            
            encoder = nn.Sequential(
                conv1,
                bn1,
                nn.LeakyReLU(),
                conv2,
                bn2,
                nn.LeakyReLU(),
                conv3,
                bn3,
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            self.finger_encoder.append(encoder)
            
            in_features = encoder_out_dim
            layers = []
            for hidden_size in hidden_layers:
                layers.append(ResMLPBlock(in_features, hidden_size, activation='leaky_relu'))
                in_features = hidden_size
            fc = nn.Linear(in_features, 12)
            layers.append(fc)
            layers = nn.Sequential(*layers)
            self.mano_regressor.append(layers)
            
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.input_size, self.input_channel)
        
        x = x.permute(0, 2, 1)
        
        finger_outputs = []
        finger_feats = []
        for i in range(5):
            finger_input = torch.cat([x[:, :, :1], x[:, :, np.array(Mediapipe2Mano[i])]], dim=2) # discard wrist, 4 points per finger
            finger_feat = self.finger_encoder[i](finger_input)
            finger_feats.append(finger_feat)
            finger_mano = self.mano_regressor[i](finger_feat)
            finger_outputs.append(finger_mano)
        x = torch.cat(finger_outputs, dim=1)
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

