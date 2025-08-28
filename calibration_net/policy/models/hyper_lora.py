import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from functools import partial
from contextlib import AbstractContextManager


def lora_hook(module, input, output, lora_weight, rank, alpha=1.0):
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


class HyperBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, in_feat, out_feat, rank):
        super(HyperBlock, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, in_feat * rank + rank * out_feat)

        nn.init.kaiming_uniform_(self.linear2.weight[:in_feat * rank], a=math.sqrt(5))
        nn.init.zeros_(self.linear2.weight[in_feat * rank:])
        nn.init.zeros_(self.linear2.bias[in_feat * rank:])
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class HyperLoRA(nn.Module):
    def __init__(self, input_size, base_model: nn.Module, max_modules:int = -1, rank: int = 2, alpha: float = 2.0):
        super(HyperLoRA, self).__init__()
        self.input_dim = input_size
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.max_modules = max_modules
        self.build_lora()

    def build_lora(self):
        self.modules = []
        for name, layers in self.base_model.named_modules():
            if self.max_modules > 0 and len(self.modules) >= self.max_modules:
                break
            if isinstance(layers, nn.Linear):
                module = HyperBlock(input_dim=self.input_dim, hidden_dim=256, in_feat=layers.in_features, out_feat=layers.out_features, rank=self.rank)
                self.modules.append((layers, module))
        self.lora_modules = nn.ModuleList([m for _, m in self.modules])

    def register_hooks(self, hyper_lora_input):
        for layers, module in self.modules:
            lora_weight = module(hyper_lora_input)
            layers.register_forward_hook(partial(lora_hook, lora_weight=lora_weight, rank=self.rank, alpha=self.alpha))

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
    hyper_lora = HyperLoRA(input_dim=5, base_model=simple_mlp, rank=2, max_modules=1)
    print(hyper_lora(torch.randn(1, 256), torch.randn(1, 5)))
    print(hyper_lora)
    breakpoint()
