import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResMLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.linear(x)
        x = self.activation(x)
        if x.shape[-1] != residual.shape[-1]:
            return x
        else:
            return x + residual

class FlowMatching(nn.Module):
    def __init__(self, input_dim, output_dim, n_steps=5, hidden_layers=[]):
        """
        Initialize the Flow-Matching 1D Diffusion Model.
        
        Parameters:
        - input_dim: Dimension of the input sequence.
        - hidden_dim: Number of hidden units in the MLP.
        - n_steps: Number of diffusion steps.
        - diffusion_type: The type of diffusion ('forward' or 'reverse').
        """
        super(FlowMatching, self).__init__()
        
        self.condition_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.time_dim = 1
        self.calib_block = nn.Linear(1,1)

        # MLP network for flow-matching process
        input_dim = self.condition_dim + self.output_dim + self.time_dim
        self.layers = []
        for layers in hidden_layers:
            self.layers.append(ResMLPBlock(input_dim, layers))
            input_dim = layers
        self.layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.ModuleList(self.layers)

        # Noise schedule
        self.noise_schedule = torch.linspace(0.0, 1.0, n_steps + 1)  # Diffusion noise schedule

    def forward(self, t: Tensor, x_t: Tensor, cond: Tensor):
        latent = torch.cat((t, x_t, cond), -1)
        for layers in self.layers:
            latent = layers(latent)
        return latent

    
    def train_forward(self, cond: Tensor, x_1: Tensor):
        cond = cond.reshape(cond.shape[0], -1)
        x_1 = x_1.reshape(x_1.shape[0], -1)
        x_0 = torch.randn_like(x_1, device=cond.device)
        t =  torch.rand(len(x_1), 1).to(cond.device)
        dx_t = x_1 - x_0
        x_t = (1-t) * x_0 + t * x_1
        pred_dx_t = self.forward(t=t, x_t=x_t, cond=cond)
        loss = F.mse_loss(pred_dx_t, dx_t)
        result = {
            'loss': loss,
            'pred_dx_t': pred_dx_t
        }
        return result


    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, cond: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

        mid_point_x_t = x_t + self(x_t=x_t, t=t_start, cond=cond) * (t_end - t_start) / 2

        mid_t = t_start + (t_end - t_start) / 2

        return x_t + (t_end - t_start) * self(t=mid_t, x_t=mid_point_x_t, cond=cond)

    def predict(self, cond: Tensor):
        noise = torch.randn(cond.shape[0], self.output_dim).to(cond.device)
        cond = cond.reshape(cond.shape[0], -1)
        x_t = noise
        for i in range(self.n_steps):
            t_start = self.noise_schedule[i].to(cond.device)
            t_end = self.noise_schedule[i + 1].to(cond.device)

            x_t = self.step(x_t, t_start, t_end, cond=cond)

        return x_t

# Example usage:
if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 64
    n_steps = 5
    model = FlowMatching(input_dim, hidden_dim, n_steps)
    
    # Example input sequence (batch_size, seq_length, input_dim)
    x = torch.randn(16, 50, input_dim)  # 16 samples, 50 timesteps, 10 feature dimensions
    
    output = model(x)
    print(output.shape)  # Should output shape: (16, 50, 10)
