import torch
from torch import nn
import os
import yaml
import pickle
from calibration_net.utils import walk_dict, create_dict_from_flattened
from calibration_net.policy.models import *
import numpy as np
from calibration_net.policy.models.hyper_lora import HyperLoRA
from calibration_net.policy.models.iknet import IKNet
from calibration_net.policy.models.preference_extractor import TransformerExtractor
from contextlib import nullcontext
from omegaconf import OmegaConf


class Model(nn.Module):
    def __init__(self, input_size, output_size, input_channel, type, hypernet_cfg, preference_extractor_cfg, iknet_cfg={}, hidden_layers=None, joints_layout=[], full_input=False, num_fuse_layer=0, train=True):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model_type = type
        self.hidden_layers = hidden_layers
        self.hypernet_cfg = hypernet_cfg
        self.iknet_cfg = iknet_cfg
        self.input_channel = input_channel
        self.is_train = train
        self.joints_layout = joints_layout
        self.full_input = full_input
        self.num_fuse_layer = num_fuse_layer
        self.preference_extractor_cfg = preference_extractor_cfg
        self.build_network()
        
    def build_network(self):
        self.calib_block = nn.ModuleList()
        if self.model_type == 'MLP':
            self.network = MLP(
                input_size=self.input_size,
                output_size=self.output_size,
                input_channel=self.input_channel,
                hidden_layers=self.hidden_layers
            )
            
        elif self.model_type == 'CNN_MLP':
            self.network = CNN_MLP(
                input_size=self.input_size,
                output_size=self.output_size,
                input_channel=self.input_channel,
                hidden_layers=self.hidden_layers
            )
            
        elif self.model_type == 'Combined_CNN_MLP':
            self.network = Combined_CNN_MLP(
                input_size=self.input_size,
                output_size=self.output_size,
                input_channel=self.input_channel,
                hidden_layers=self.hidden_layers
            )
        elif self.model_type == 'Combined_CNN_MLP_MIXER':
            self.network = Combined_CNN_MLP_MIXER(
                input_size=self.input_size,
                output_size=self.output_size,
                input_channel=self.input_channel,
                hidden_layers=self.hidden_layers
            )

        elif self.model_type == 'CombinedMLP':
            self.network = CombinedModel(
                joints_layout=self.joints_layout,
                model_cls=MLP,
                full_input=self.full_input,
                input_size=self.input_size,
                input_channel=self.input_channel,
                hidden_layers=self.hidden_layers,
                num_fuse_layer=self.num_fuse_layer,
            )

        elif self.model_type == 'Flow':
            self.network = FlowMatching(
                input_dim=self.input_size * self.input_channel,
                output_dim=self.output_size,
                hidden_layers=self.hidden_layers,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")    
        
        if self.iknet_cfg.get('enabled', False):
            iknet_cfg = dict(self.iknet_cfg)
            del iknet_cfg['enabled']
            self.iknet = IKNet(**iknet_cfg)

        if self.hypernet_cfg['input_size'] > 0:
            self.hyperlora = HyperLoRA(**self.hypernet_cfg, base_model=self.network, train_init=self.is_train)
            self.calib_block.append(self.hyperlora.lora_modules)
            
        if self.preference_extractor_cfg.get('enabled', False):
            preference_extractor_cfg = dict(self.preference_extractor_cfg)
            del preference_extractor_cfg['enabled']
            self.preference_extractor = TransformerExtractor(
                **preference_extractor_cfg
            )
            self.calib_block.append(self.preference_extractor)
        return self.network
    
    def forward(self, x, y=None, hypernet_x=None, noisy_act=None, ik_y=None, side=None):
        context = self.hyperlora.use(hypernet_x) if hypernet_x is not None else nullcontext()
        
        ik_loss = torch.tensor(0.0, device=x.device)

        with context:
            if self.iknet_cfg.get('enabled', False):
                if ik_y is None:
                    axis_angle, quat = self.iknet(x, side=side)
                else:
                    axis_angle, quat, ik_loss = self.iknet(x, side=side, ik_y=ik_y)
                x = axis_angle


            if y is not None:
                result = self.network.train_forward(x, y)
                result['loss_qpos'] = result['loss'].clone()
                result['loss'] += ik_loss * 0.1
                result['loss_ik'] = ik_loss
            else:
                result = self.network.predict(x, noisy_act=noisy_act)
                
        return result

    @torch.no_grad()
    def predict(self, x, hypernet_x=None, noisy_act=None, side=None):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x, hypernet_x=hypernet_x, noisy_act=noisy_act, side=side)
        return y
    

def load_model(model_dir: str, device: str = 'cpu') -> Model:
    """
    Load a model from the specified path.
    
    Args:
        model_dir (str): Path to the model dir.
        device (str): Device to load the model on ('cpu' or 'cuda').
        
    Returns:
        Model: The loaded model.
        Normalization stats: dict
        Config: dict
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

    model = Model(**model_cfg, train=False)
    model_weight_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    model.to(device)
    
    return model, normalization_stats, cfg
    
    