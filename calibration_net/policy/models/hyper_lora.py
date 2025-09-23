import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from functools import partial
from contextlib import AbstractContextManager
import numpy as np
from tqdm import tqdm


def lora_hook(module, input, output, lora_weight, rank, alpha=1.0, drop_out=0.0):
    rand = torch.rand(1).item()
    if rand < drop_out:
        return output
    
    in_dim, out_dim = module.in_features, module.out_features
    batch_size = lora_weight.shape[0]
    lora_A = lora_weight[:, :in_dim*rank].reshape(batch_size, in_dim, rank)
    lora_B = lora_weight[:, in_dim*rank:].reshape(batch_size, rank, out_dim)
    lora_output = input[0].unsqueeze(1) @ lora_A @ lora_B / float(rank) * alpha
    output = output + lora_output.squeeze(1)
    return output

class HyperLoRAHookContext(AbstractContextManager):
    def __init__(self, outer, hyper_lora_input: Tensor):
        self.outer = outer
        self.hyper = hyper_lora_input
    def __enter__(self):
        self.outer.register_hooks(self.hyper)
        return self.outer
    def __exit__(self, exc_type, exc, tb):
        self.outer.unregister_hooks()
        return False  # don’t suppress exceptions

def w1_to_uniform(x: torch.Tensor, a: float = 0.0, b: float = 1.0):
    """
    1D Wasserstein-1 distance between empirical samples x and Uniform(a,b).
    """
    x = x.flatten().to(torch.float64)
    n = x.numel()
    if n == 0:
        return torch.tensor(float("nan"), dtype=torch.float64, device=x.device)
    xs = torch.sort(x).values
    # mid-quantiles of the target uniform
    grid = (torch.arange(1, n+1, device=x.device, dtype=torch.float64) - 0.5) / n
    target = a + grid * (b - a)
    return torch.mean(torch.abs(xs - target))

def init_hypernet(net, sample_num=128, n_steps=20):
    optim = torch.optim.Adam(net.parameters(), lr=0.01)
    for i in range(n_steps):
        optim.zero_grad()
        rand_in = torch.normal(mean=0, std=1, size=(sample_num, net.input_dim))
        weight_samples = net(rand_in)
        lora_a_samples = weight_samples[:, :net.in_feat * net.rank]
        lora_a_fan_in = net.in_feat
        lora_a_bound = 1 / np.sqrt(lora_a_fan_in) 
        loss_a = w1_to_uniform(lora_a_samples, -lora_a_bound, lora_a_bound)
        loss = loss_a
        loss.backward()
        optim.step()
    return net

class HyperBlock(nn.Module):
    def __init__(self, input_dim, in_feat, out_feat, rank, hidden_layers=[]):
        super(HyperBlock, self).__init__()
        self.input_dim = input_dim
        if hidden_layers == [] or hidden_layers is None:
            hidden_layers = [256, 256]
        self.linear1 = nn.Linear(input_dim, hidden_layers[0])
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.gelu = nn.GELU()
        self.layers = nn.ModuleList()
        hidden_dim = hidden_layers[0]
        for dim in hidden_layers:
            self.layers.append(nn.Linear(hidden_dim, dim))
            self.layers.append(nn.GELU())
            self.layers.append(nn.LayerNorm(dim))
            hidden_dim = dim

        self.fc = nn.Linear(hidden_dim, in_feat * rank + rank * out_feat)
        init_hypernet(self, sample_num=256, n_steps=30)

        # nn.init.zeros_(self.fc.weight[:in_feat * rank])
        # nn.init.zeros_(self.fc.bias[:in_feat * rank])
        nn.init.zeros_(self.fc.weight[in_feat * rank:])
        nn.init.zeros_(self.fc.bias[in_feat * rank:])

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x

class HyperLoRA(nn.Module):
    def __init__(self, input_size, base_model: nn.Module, max_modules:int = -1, rank: int = 2, alpha: float = 2.0, drop_out: float=0.0, hidden_layers=[], name_filter=[]):
        super(HyperLoRA, self).__init__()
        self.input_dim = input_size
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.max_modules = max_modules
        self.hidden_layers = hidden_layers
        self.name_filter = name_filter
        self.drop_out = drop_out
        self.build_lora()

    def build_lora(self):
        print("building hyper lora")
        self.modules = []
        for name, layers in self.base_model.named_modules():
            if self.name_filter is not None:
                for filter in self.name_filter:
                    if filter in name: break
                else: continue
            print(f"building {name} hyper lora module")
            if self.max_modules > 0 and len(self.modules) >= self.max_modules:
                break
            if isinstance(layers, nn.Linear):
                module = HyperBlock(input_dim=self.input_dim, in_feat=layers.in_features, out_feat=layers.out_features, rank=self.rank, hidden_layers=self.hidden_layers)
                self.modules.append((layers, module))
        self.lora_modules = nn.ModuleList([m for _, m in self.modules])

    def register_hooks(self, hyper_lora_input):
        for layers, module in self.modules:
            lora_weight = module(hyper_lora_input)
            layers.register_forward_hook(partial(lora_hook, lora_weight=lora_weight, rank=self.rank, alpha=self.alpha, drop_out=self.drop_out))

    def unregister_hooks(self):
        for layers, _ in self.modules:
            layers._forward_hooks.clear()

    def forward(self, x: Tensor, hyper_lora_input: Tensor) -> Tensor:
        self.register_hooks(hyper_lora_input)
        x = self.base_model(x)
        self.unregister_hooks()
        return x
    
    def use(self, hyper_lora_input: Tensor) -> HyperLoRAHookContext:
        """Factory that returns a context manager carrying your argument."""
        return HyperLoRAHookContext(self, hyper_lora_input)

if __name__ == "__main__":
    simple_mlp = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 512)
    )
    hyper_lora = HyperLoRA(input_size=5, base_model=simple_mlp, rank=2, max_modules=1, hidden_layers=[256])
    print(hyper_lora(torch.randn(1, 256), torch.randn(1, 5)))
    print(hyper_lora)
    breakpoint()
