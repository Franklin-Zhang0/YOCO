from torch import nn, Tensor
import torch

class FiLM(nn.Module):
    def __init__(self, hdim, edim):
        super().__init__()
        self.to_scale = nn.Linear(edim, hdim)
        self.to_shift = nn.Linear(edim, hdim)
    def forward(self, h, e):
        return h * (1 + self.to_scale(e)) + self.to_shift(e)

class ResMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, film_dim=None):
        super(ResMLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

        if film_dim is not None:
            self.film = FiLM(output_dim, film_dim)
        else:
            self.film = None

    def forward(self, x: Tensor, cond: Tensor=None) -> Tensor:
        residual = x
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        if x.shape[-1] != residual.shape[-1]:
            return x
        else:
            return x + residual