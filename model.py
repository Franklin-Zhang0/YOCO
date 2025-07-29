import torch
from torch import nn
import os
import yaml
import pickle
from utils import walk_dict, create_dict_from_flattened
import numpy as np

class Model(nn.Module):
    def __init__(self, input_size, output_size, type, hidden_layers=None):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model_type = type
        self.hidden_layers = hidden_layers
        self.network = self.build_network()
        
    def build_network(self):
        if self.model_type == 'MLP':
            self.calib_block = nn.Sequential(nn.Conv1d(
                in_channels=1,
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
            in_features = self.input_size
            for hidden_size in self.hidden_layers:
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(nn.GELU())
                in_features = hidden_size
            self.fc = nn.Linear(in_features, self.output_size)
            layers.append(self.fc)
            return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.calib_block(x.unsqueeze(-2)).squeeze(-2)  # Apply calibration block
        
        return self.network(x)
    
    @torch.no_grad()
    def predict(self, x, normalization_stats=None, norm_type='minmax'):
        self.eval()
        if normalization_stats:
            if norm_type == 'minmax':
                x = (x - normalization_stats['x']['min']) / (normalization_stats['x']['max'] - normalization_stats['x']['min'] + 1e-5)
            elif norm_type == 'meanstd':
                x = (x - normalization_stats['x']['mean']) / normalization_stats['x']['std']
        x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x)
        if normalization_stats:
            y = y * normalization_stats['y']['std'] + normalization_stats['y']['mean']
        return y
    

def load_model(model_dir: str, device: str = 'cpu') -> Model:
    """
    Load a model from the specified path.
    
    Args:
        model_dir (str): Path to the model dir.
        device (str): Device to load the model on ('cpu' or 'cuda').
        
    Returns:
        Model: The loaded model.
    """
    model_dir = os.path.expanduser(model_dir)
    cfg_path = os.path.join(model_dir, 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        model_cfg = cfg['model']
    
    with open(os.path.join(model_dir, 'norm_stats.pkl'), 'rb') as f:
        normalization_stats = pickle.load(f)

    normalization_stats = {k: np.array(v) for k,v in walk_dict(normalization_stats)}
    normalization_stats = create_dict_from_flattened(normalization_stats)

    model = Model(**model_cfg)
    model_weight_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    model.to(device)
    
    return model, normalization_stats, cfg
    
    