import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from calibration_net.policy.models.resmlp import ResMLPBlock

Mediapipe2Mano = [
    [5, 6, 7, 8],    # Index
    [9, 10, 11, 12], # Middle
    [17, 18, 19, 20],# Pinky
    [13, 14, 15, 16],# Ring
    [1, 2, 3, 4]     # Thumb
]

# Standard MediaPipe Tip Indices corresponding to the order in Mediapipe2Mano
# 8 (Index), 12 (Middle), 20 (Pinky), 16 (Ring), 4 (Thumb)
TIP_INDICES = [8, 12, 20, 16, 4]

class FingertipFeatureEncoder(nn.Module):
    """Encodes the 5x5 pairwise vectors between fingertips into a feature vector."""
    def __init__(self, input_channel, output_dim):
        super().__init__()
        self.num_tips = 5
        # Input is 5x5 vectors, each has 'input_channel' dimensions
        flattened_input_dim = self.num_tips * self.num_tips * input_channel
        
        self.encoder = nn.Sequential(
            nn.Linear(flattened_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
            nn.LeakyReLU()
        )

    def forward(self, x_tips):
        # x_tips shape: (Batch, Input_Channel, 5)
        B, C, N = x_tips.shape
        
        # Calculate pairwise vectors: tip_i - tip_j
        # Expand to (B, C, 5, 5)
        # (B, C, N, 1) - (B, C, 1, N) broadcasted
        diff_vectors = x_tips.unsqueeze(3) - x_tips.unsqueeze(2)
        
        # Flatten to (B, C * 5 * 5) = (B, 75) if C=3
        diff_vectors = diff_vectors.view(B, -1)
        
        return self.encoder(diff_vectors)

class Robot_Joint_Regressor(nn.Module):
    def __init__(self, input_size, output_size, input_channel, joints_layout, hidden_layers=[]):
        """
        Args:
            input_size: Number of input keypoints (typically 21 for hand)
            output_size: Total number of output joints (sum of joints_layout)
            input_channel: Number of channels per keypoint (typically 3 for xyz)
            joints_layout: List indicating number of joints for each finger [thumb, index, middle, ring, pinky]
            hidden_layers: List of hidden layer sizes for the MLP blocks
        """
        super(Robot_Joint_Regressor, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.joints_per_finger = joints_layout
        
        self.finger_encoder = nn.ModuleList()
        self.joint_regressor = nn.ModuleList()
        
        conv_channel = 64
        encoder_out_dim = conv_channel * 5 # 320
        self.finger_scaling = nn.Parameter(torch.ones(5))

        # --- NEW: Fingertip Encoder ---
        self.tip_encoder = FingertipFeatureEncoder(input_channel, encoder_out_dim)
        
        # --- NEW: Self-Attention Fusion ---
        # We use a Transformer Encoder Layer.
        # d_model matches the encoder_out_dim (320)
        self.fusion_norm = nn.LayerNorm(encoder_out_dim)
        self.fusion_attention = nn.TransformerEncoderLayer(
            d_model=encoder_out_dim, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        
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
            
            # Output dimension depends on joints for this finger
            joints_for_finger = len(joints_layout[i])
            in_features = encoder_out_dim
            layers = []
            for hidden_size in hidden_layers:
                layers.append(ResMLPBlock(in_features, hidden_size, activation='leaky_relu'))
                in_features = hidden_size
            fc = nn.Linear(in_features, joints_for_finger)
            layers.append(fc)
            layers = nn.Sequential(*layers)
            self.joint_regressor.append(layers)
            

    def extract_features(self, x):
        """Extract intermediate token features for use in preference extractor.
        
        Args:
            x: (B, 21, 3) or (B, 63) keypoints
            
        Returns:
            features: (B, 6, 320) token features after fusion
                     6 tokens = [5 finger tokens, 1 fingertip context token]
        """
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.input_size, self.input_channel)
        
        # Permute to (Batch, Channel, Points) for Conv1d
        x = x.permute(0, 2, 1)
        
        # 1. Individual Finger Encoding
        finger_feats_list = []
        for i in range(5):
            finger_input = torch.cat([x[:, :, :1], x[:, :, np.array(Mediapipe2Mano[i])]], dim=2)
            finger_feat = self.finger_encoder[i](finger_input) # (B, 320)
            finger_feats_list.append(finger_feat)
        
        # Stack finger features: (B, 5, 320)
        finger_feats_stack = torch.stack(finger_feats_list, dim=1)
        
        # 2. Fingertip Feature Extraction
        tips = x[:, :, TIP_INDICES] # (B, C, 5)
        tip_feat = self.tip_encoder(tips) # (B, 320)
        tip_feat = tip_feat.unsqueeze(1)  # (B, 1, 320)
        
        # 3. Fusion (Self-Attention)
        fusion_input = torch.cat([finger_feats_stack, tip_feat], dim=1)  # (B, 6, 320)
        fusion_out = self.fusion_attention(fusion_input) # (B, 6, 320)
        
        # Return full token features (not pooled)
        return fusion_out  # (B, 6, 320)
    
    def forward(self, x, return_feat=False):
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.input_size, self.input_channel)
        
        # Permute to (Batch, Channel, Points) for Conv1d
        x = x.permute(0, 2, 1)
        
        # 1. Individual Finger Encoding
        finger_feats_list = []
        for i in range(5):
            finger_input = torch.cat([x[:, :, :1], x[:, :, np.array(Mediapipe2Mano[i])]], dim=2)
            finger_feat = self.finger_encoder[i](finger_input) # (B, 320)
            finger_feats_list.append(finger_feat)
        
        # Stack finger features: (B, 5, 320)
        finger_feats_stack = torch.stack(finger_feats_list, dim=1)
        
        # 2. Fingertip Feature Extraction
        # Extract the 5 fingertips based on TIP_INDICES
        # x is (B, C, N), we slice N
        tips = x[:, :, TIP_INDICES] # (B, C, 5)
        tip_feat = self.tip_encoder(tips) # (B, 320)
        tip_feat = tip_feat.unsqueeze(1)  # (B, 1, 320)
        
        # 3. Fusion (Self-Attention)
        # Concatenate: [Finger1, Finger2, Finger3, Finger4, Finger5, TipsContext]
        # Shape: (B, 6, 320)
        fusion_input = torch.cat([finger_feats_stack, tip_feat], dim=1)
        
        # Apply Transformer Encoder
        # We usually apply residual connection or norm, but TransformerEncoderLayer handles that internally.
        # We might want to normalize input before attention or let the layer handle it.
        fusion_out = self.fusion_attention(fusion_input) # (B, 6, 320)
        
        # 4. Decoding
        # Extract the updated finger features (first 5 elements)
        updated_finger_feats = fusion_out[:, :5, :]
        
        finger_outputs = []
        for i in range(5):
            # Pass the attended feature to the regressor
            f_feat = updated_finger_feats[:, i, :]
            finger_joints = self.joint_regressor[i](f_feat)
            finger_outputs.append(finger_joints)
            
        # Concatenate all finger joint outputs
        joint_positions = torch.cat(finger_outputs, dim=1)

        if return_feat:
            return joint_positions, updated_finger_feats
        else:
            return joint_positions

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