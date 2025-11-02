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
from contextlib import nullcontext
from omegaconf import OmegaConf

class CombinedModel(nn.Module):
    def __init__(self, joints_layout, model_cls, full_input, **kwargs):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList()
        self.joints_layout = joints_layout
        self.full_input = full_input
        kwargs['input_size'] = len(self.get_pos_range(0, kwargs['input_size']))
        for joints in joints_layout:
            output_size = len(joints)
            model = model_cls(output_size=output_size ,**kwargs)
            self.models.append(model)

    def get_pos_range(self, i, n_pos):
        if self.full_input:
            return list(range(n_pos))
        if n_pos == 21:
            pos_range = list(range(1+4*i, 1+4*(i+1)))
        elif n_pos == 5:
            pos_range = list(range(i, i+1))
        else:
            raise NotImplementedError(f"n_pos {n_pos} not supported")
        return pos_range

            
    def forward(self, x, **kwargs):
        outs = []
        for i, model in enumerate(self.models):
            pos_range = self.get_pos_range(i, x.shape[-2])
            out = model(x[:, pos_range], **kwargs)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        return x
    
    def train_forward(self, x, y):
        outs = []
        losses = []
        for i, model in enumerate(self.models):
            pos_range = self.get_pos_range(i, x.shape[-2])
            out = model.train_forward(x[:, pos_range], y[:, sum(len(j) for j in self.joints_layout[:i]):sum(len(j) for j in self.joints_layout[:i+1])])
            outs.append(out['pred_y'])
            losses.append(out['loss'])
        pred_y = torch.cat(outs, dim=-1)
        loss = sum(losses)
        result = {
            'loss': loss,
            'pred_y': pred_y,
        }
        return result
    
    def predict(self, x, **kwargs):
        outs = []
        for i, model in enumerate(self.models):
            pos_range = self.get_pos_range(i, x.shape[-2])
            out = model.predict(x[:, pos_range], **kwargs)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        return x

class Model(nn.Module):
    def __init__(self, input_size, output_size, input_channel, type, hypernet_cfg, iknet_cfg={}, hidden_layers=None, joints_layout=[], full_input=False, train=True):
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
        self.build_network()
        
    def build_network(self):
        if self.model_type == 'MLP':
            self.network = MLP(
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
                hidden_layers=self.hidden_layers
            )

        elif self.model_type == 'Flow':
            self.network = FlowMatching(
                input_dim=self.input_size * self.input_channel,
                output_dim=self.output_size,
                hidden_layers=self.hidden_layers,
            )
            self.calib_block = self.network.calib_block # dummy calib block
        if self.iknet_cfg.get('enabled', False):
            iknet_cfg = dict(self.iknet_cfg)
            del iknet_cfg['enabled']
            self.iknet = IKNet(**iknet_cfg)

        if self.hypernet_cfg['input_size'] > 0:
            self.hyperlora = HyperLoRA(**self.hypernet_cfg, base_model=self.network, train_init=self.is_train)
            self.calib_block = self.hyperlora.lora_modules
        else:
            self.calib_block = None
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
    
    