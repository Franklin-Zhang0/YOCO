import torch
import torch.nn as nn
from .resmlp import ResMLPBlock

class FusionModule(nn.Module):
    def __init__(self, input_size, output_size, n_layers=2):
        super(FusionModule, self).__init__()
        self.fuse_module = nn.ModuleList()
        for i in range(n_layers):
            self.fuse_module.append(ResMLPBlock(input_size + output_size, input_size + output_size))

    def forward(self, x, cond):
        cond = cond.reshape(cond.shape[0], -1)
        inputs = torch.cat((x, cond), dim=-1)
        for layer in self.fuse_module:
            inputs = layer(inputs)
        return inputs[..., :x.shape[-1]] * 0.01 + x

class CombinedModel(nn.Module):
    def __init__(self, joints_layout, model_cls, full_input, num_fuse_layer, **kwargs):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList()
        self.joints_layout = joints_layout
        self.full_input = full_input
        self.full_input_size = kwargs['input_size']
        kwargs['input_size'] = len(self.get_pos_range(0, kwargs['input_size']))
        for joints in joints_layout:
            output_size = len(joints)
            model = model_cls(output_size=output_size ,**kwargs)
            self.models.append(model)

        if num_fuse_layer > 0:
            self.fuse_module = FusionModule(
                input_size=self.full_input_size * 3,
                output_size=sum(len(j) for j in joints_layout),
                n_layers=num_fuse_layer
            )
        else:
            self.fuse_module = lambda x, cond: x
        

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
        res = torch.cat(outs, dim=-1)
        res = self.fuse_module(res, x)
        return res

    def train_forward(self, x, y):
        outs = []
        losses = []
        for i, model in enumerate(self.models):
            pos_range = self.get_pos_range(i, x.shape[-2])
            out = model.train_forward(x[:, pos_range], y[:, sum(len(j) for j in self.joints_layout[:i]):sum(len(j) for j in self.joints_layout[:i+1])])
            outs.append(out['pred_y'])
            losses.append(out['loss'])
        pred_y = torch.cat(outs, dim=-1)
        pred_y = self.fuse_module(pred_y, x)
        loss = torch.cat(losses, dim=-1)
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
        res = torch.cat(outs, dim=-1)
        res = self.fuse_module(res, x)
        return res