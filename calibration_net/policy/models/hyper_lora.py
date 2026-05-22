import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from functools import partial
from contextlib import AbstractContextManager
import numpy as np
from tqdm import tqdm


def lora_hook(module, input, output, lora_weight, rank, alpha=1.0, drop_out=0.0, gate=None, max_norm=None):
    """LoRA forward hook with optional gating and max-norm constraint.
    
    Args:
        module: Target nn.Linear layer
        input: Tuple of input tensors
        output: Base layer output
        lora_weight: Generated LoRA weights (B, in_dim*rank + rank*out_dim)
        rank: LoRA rank
        alpha: LoRA scaling factor
        drop_out: Dropout probability
        gate: Optional learnable gate parameter (scalar)
        max_norm: Optional max Frobenius norm for delta weights
    """
    rand = torch.rand(1).item()
    if rand < drop_out:
        return output
    
    in_dim, out_dim = module.in_features, module.out_features
    batch_size = lora_weight.shape[0]
    lora_A = lora_weight[:, :in_dim*rank].reshape(batch_size, in_dim, rank)
    lora_B = lora_weight[:, in_dim*rank:].reshape(batch_size, rank, out_dim)
    
    # Compute delta weights
    delta_w = lora_A @ lora_B / float(rank) * alpha  # (B, in_dim, out_dim)
    
    # Optional: max-norm constraint
    if max_norm is not None:
        norm = torch.norm(delta_w, p='fro', dim=(-2, -1), keepdim=True)  # (B, 1, 1)
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        delta_w = delta_w * scale
    # Apply delta
    if input[0].dim() == 2:
        lora_output = input[0].unsqueeze(1) @ delta_w  # (B, 1, out_dim)
        lora_output = lora_output.squeeze(1)  # (B, out_dim)
    else:
        lora_output = input[0] @ delta_w 
    
    # Optional: learnable gate (starts near 0)
    if gate is not None:
        gate_value = torch.sigmoid(gate)
        lora_output = gate_value * lora_output
    
    output = output + lora_output
    return output

class HyperLoRAHookContext(AbstractContextManager):
    """Context manager for HyperLoRA forward hooks."""
    def __init__(self, outer, hyper_lora_input: Tensor):
        self.outer = outer
        self.hyper = hyper_lora_input
        self.lora_weights = self.outer.prepare_lora_weights(hyper_lora_input)
    def __enter__(self):
        self.outer.register_hooks(self.hyper, self.lora_weights)
        return self.outer
    def __exit__(self, exc_type, exc, tb):
        self.outer.unregister_hooks()
        return False  # don’t suppress exceptions

def w1_to_uniform(x: torch.Tensor, a: float = 0.0, b: float = 1.0):
    """1D Wasserstein-1 distance between empirical samples x and Uniform(a,b)."""
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
    """Initialize hypernet to match uniform distribution (Wasserstein method)."""
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
    """Hypernetwork block that generates LoRA weights.
    
    Args:
        input_dim: Conditioning vector dimension
        in_feat: Input features of target layer
        out_feat: Output features of target layer
        rank: LoRA rank
        hidden_layers: List of hidden layer sizes
        train_init: Whether to use Wasserstein initialization
        init_method: 'xavier' | 'wasserstein'
        use_tanh: Whether to bound outputs with tanh
        scale_A: Scale for A matrix (after tanh)
        scale_B: Scale for B matrix (after tanh)
    """
    
    def __init__(
        self,
        input_dim,
        in_feat,
        out_feat,
        rank,
        hidden_layers=[],
        train_init=True,
        init_method='xavier',
        use_tanh=False,
        scale_A=1.0,
        scale_B=1.0,
    ):
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
        
        # Bounded parameterization
        self.use_tanh = use_tanh
        self.scale_A = scale_A
        self.scale_B = scale_B
        
        # Initialize based on method
        if init_method == 'wasserstein' and train_init:
            init_hypernet(self, sample_num=256, n_steps=30)
        elif init_method == 'xavier':
            # Xavier initialization for A part
            fan_in_A = in_feat
            bound_A = 1.0 / np.sqrt(fan_in_A)
            with torch.no_grad():
                # Initialize A part
                self.fc.weight[:, :in_feat * rank].uniform_(-bound_A, bound_A)
                self.fc.bias[:in_feat * rank].uniform_(-bound_A, bound_A)

        # B always zero-initialized (standard LoRA)
        nn.init.zeros_(self.fc.weight[:, in_feat * rank:])
        nn.init.zeros_(self.fc.bias[in_feat * rank:])

    def forward(self, x):
        """Generate LoRA weights from conditioning vector.
        
        Args:
            x: Conditioning vector (B, input_dim)
        
        Returns:
            LoRA weights (B, in_feat*rank + rank*out_feat)
        """
        x = self.linear1(x)
        x = self.gelu(x)
        for layer in self.layers:
            x = layer(x)
        raw_weights = self.fc(x)
        
        if self.use_tanh:
            # Split and bound separately
            lora_A_raw = raw_weights[:, :self.in_feat * self.rank]
            lora_B_raw = raw_weights[:, self.in_feat * self.rank:]
            
            lora_A = torch.tanh(lora_A_raw) * self.scale_A
            lora_B = torch.tanh(lora_B_raw) * self.scale_B
            
            return torch.cat([lora_A, lora_B], dim=-1)
        else:
            return raw_weights


class FlowTimeEmbed(nn.Module):
    def __init__(self, hidden_dim: int, n_fourier: int = 32):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(10000), n_fourier)))
        self.proj = nn.Sequential(
            nn.Linear(n_fourier * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t[None]
        phases = t[:, None] / self.freq[None].to(t.device)
        return self.proj(torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1))


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTFlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(6, dim=-1)
        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_msa[:, None, :] * attn_out
        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(mlp_in)
        return x


class DiTFlowHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_tokens: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        steps: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.steps = max(int(steps), 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.start_tokens = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.cond_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.init_proj = nn.Linear(hidden_dim, num_tokens * hidden_dim)
        self.time_embed = FlowTimeEmbed(hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTFlowBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.velocity_norm = nn.LayerNorm(hidden_dim)
        self.velocity_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(num_tokens * hidden_dim, output_dim)

    def _velocity(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        block_cond = cond + self.time_embed(t)
        for block in self.blocks:
            x = block(x, block_cond)
        return self.velocity_proj(self.velocity_norm(x))

    def forward(self, cond: Tensor) -> Tensor:
        cond = self.cond_proj(cond)
        x = self.start_tokens.expand(cond.shape[0], -1, -1) + self.pos_embed
        x = x + self.init_proj(cond).view(cond.shape[0], self.num_tokens, self.hidden_dim)
        dt = 1.0 / float(self.steps)
        for step in range(self.steps):
            t = torch.full((cond.shape[0],), step * dt, dtype=cond.dtype, device=cond.device)
            x = x + self._velocity(x, cond, t) * dt
        x = self.output_norm(x).reshape(cond.shape[0], self.num_tokens * self.hidden_dim)
        return self.output_proj(x)


class DiTFlowHyperBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        in_feat,
        out_feat,
        rank,
        hidden_dim=128,
        num_tokens=4,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        steps=2,
        use_tanh=False,
        scale_A=1.0,
        scale_B=1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.use_tanh = use_tanh
        self.scale_A = scale_A
        self.scale_B = scale_B
        self.flow_head = DiTFlowHead(
            input_dim=input_dim,
            output_dim=in_feat * rank + rank * out_feat,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            steps=steps,
        )

    def forward(self, x):
        raw_weights = self.flow_head(x)
        if not self.use_tanh:
            return raw_weights
        lora_A_raw = raw_weights[:, :self.in_feat * self.rank]
        lora_B_raw = raw_weights[:, self.in_feat * self.rank:]
        lora_A = torch.tanh(lora_A_raw) * self.scale_A
        lora_B = torch.tanh(lora_B_raw) * self.scale_B
        return torch.cat([lora_A, lora_B], dim=-1)


class HyperLoRA(nn.Module):
    """HyperLoRA: Hypernetwork-generated LoRA for dynamic adaptation.
    
    Args:
        input_size: Conditioning vector dimension
        base_model: Base model to adapt
        max_modules: Maximum number of modules to adapt (-1 for all)
        rank: LoRA rank
        alpha: LoRA scaling factor
        drop_out: Dropout probability
        hidden_layers: Hidden layer sizes for hypernetwork
        name_filter: List of name patterns to filter modules
        train_init: Whether to use Wasserstein initialization
        init_method: 'xavier' | 'wasserstein'
        use_tanh: Whether to bound LoRA outputs with tanh
        scale_A: Scale for A matrix (after tanh)
        scale_B: Scale for B matrix (after tanh)
        use_gates: Whether to use per-module learnable gates
        gate_init: Initial gate value (before sigmoid)
        max_norm: Optional max Frobenius norm for delta weights
        reg_weight: Weight for regularization loss
        reg_type: 'frobenius' | 'b_only' | 'nuclear'
    """
    
    def __init__(
        self,
        input_size,
        base_model: nn.Module,
        max_modules: int = -1,
        rank: int = 2,
        alpha: float = 2.0,
        drop_out: float = 0.0,
        hidden_layers=[],
        name_filter=[],
        train_init=True,
        init_method='xavier',
        use_tanh=False,
        scale_A=1.0,
        scale_B=1.0,
        use_gates=False,
        gate_init=0.0,
        max_norm=None,
        reg_weight=0.0,
        reg_type='frobenius',
        head_type='mlp',
        dit_hidden_dim=128,
        dit_num_tokens=4,
        dit_depth=2,
        dit_num_heads=4,
        dit_mlp_ratio=2.0,
        dit_steps=2,
    ):
        super(HyperLoRA, self).__init__()
        self.input_dim = input_size
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.max_modules = max_modules
        self.hidden_layers = hidden_layers
        self.name_filter = name_filter
        self.drop_out = drop_out
        self.train_init = train_init
        self.init_method = init_method
        self.use_tanh = use_tanh
        self.scale_A = scale_A
        self.scale_B = scale_B
        self.use_gates = use_gates
        self.gate_init = gate_init
        self.max_norm = max_norm
        self.reg_weight = reg_weight
        self.reg_type = reg_type
        self.head_type = head_type
        self.dit_hidden_dim = dit_hidden_dim
        self.dit_num_tokens = dit_num_tokens
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_steps = dit_steps
        
        # Hook management
        self._hook_handles = []
        self._cached_lora_weights = None
        # Saved original weights for merge/unmerge
        self._original_weights = None
        
        self.build_lora()

    def build_lora(self):
        """Build HyperLoRA modules for target layers."""
        print("building hyper lora")
        self.modules_list = []
        self.gates = nn.ParameterList() if self.use_gates else None
        
        for name, layers in self.base_model.named_modules():
            if self.name_filter is not None and len(self.name_filter)>0:
                for filter in self.name_filter:
                    if filter in name: break
                else: continue
            print(f"building {name} hyper lora module")
            if self.max_modules > 0 and len(self.modules_list) >= self.max_modules:
                break
            if isinstance(layers, nn.Linear):
                if self.head_type == 'dit_flow':
                    module = DiTFlowHyperBlock(
                        input_dim=self.input_dim,
                        in_feat=layers.in_features,
                        out_feat=layers.out_features,
                        rank=self.rank,
                        hidden_dim=self.dit_hidden_dim,
                        num_tokens=self.dit_num_tokens,
                        depth=self.dit_depth,
                        num_heads=self.dit_num_heads,
                        mlp_ratio=self.dit_mlp_ratio,
                        steps=self.dit_steps,
                        use_tanh=self.use_tanh,
                        scale_A=self.scale_A,
                        scale_B=self.scale_B,
                    )
                else:
                    module = HyperBlock(
                        input_dim=self.input_dim,
                        in_feat=layers.in_features,
                        out_feat=layers.out_features,
                        rank=self.rank,
                        hidden_layers=self.hidden_layers,
                        train_init=self.train_init,
                        init_method=self.init_method,
                        use_tanh=self.use_tanh,
                        scale_A=self.scale_A,
                        scale_B=self.scale_B,
                    )
                self.modules_list.append((layers, module))
                
                # Add gate if enabled
                if self.use_gates:
                    gate = nn.Parameter(torch.tensor(self.gate_init))
                    self.gates.append(gate)
        
        self.lora_modules = nn.ModuleList([m for _, m in self.modules_list])

    def prepare_lora_weights(self, hyper_lora_input: Tensor):
        """Generate LoRA weights from conditioning vector.

        Supports either:
          - shared conditioning: (B, input_dim)
          - per-module conditioning: (B, M, input_dim), where module i consumes hyper[:, i, :]

        Args:
            hyper_lora_input: Conditioning vector(s)

        Returns:
            List of (layer, lora_weight) tuples
        """
        lora_weights = []

        # Determine conditioning mode
        if hyper_lora_input.dim() == 2:
            per_module = False
        elif hyper_lora_input.dim() == 3:
            per_module = True
            if hyper_lora_input.shape[1] != len(self.modules_list):
                raise ValueError(
                    f"Per-module hyper input has M={hyper_lora_input.shape[1]} but HyperLoRA has "
                    f"{len(self.modules_list)} modules"
                )
        else:
            raise ValueError(
                f"hyper_lora_input must have shape (B, D) or (B, M, D); got {tuple(hyper_lora_input.shape)}"
            )

        for idx, (layers, module) in enumerate(self.modules_list):
            if per_module:
                cond = hyper_lora_input[:, idx, :]
            else:
                cond = hyper_lora_input
            lora_weight = module(cond)
            lora_weights.append((layers, lora_weight))
        return lora_weights

    def register_hooks(self, hyper_lora_input, lora_weights=None):
        """Register forward hooks with safe handle management.
        
        Args:
            hyper_lora_input: Conditioning vector
            lora_weights: Pre-computed LoRA weights (optional)
        """
        # Clear old hooks first
        self.unregister_hooks()
        
        if lora_weights is None:
            lora_weights = self.prepare_lora_weights(hyper_lora_input)
        
        # Cache for regularization
        self._cached_lora_weights = lora_weights
        
        assert len(lora_weights) == len(self.modules_list)
        for idx, (layers, lora_weight) in enumerate(lora_weights):
            gate = self.gates[idx] if self.use_gates else None
            handle = layers.register_forward_hook(
                partial(
                    lora_hook,
                    lora_weight=lora_weight,
                    rank=self.rank,
                    alpha=self.alpha,
                    drop_out=self.drop_out,
                    gate=gate,
                    max_norm=self.max_norm,
                )
            )
            self._hook_handles.append(handle)

    def unregister_hooks(self):
        """Safely remove only our hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._cached_lora_weights = None

    def forward(self, x: Tensor, hyper_lora_input: Tensor) -> Tensor:
        """Forward pass with HyperLoRA.
        
        Args:
            x: Input to base model
            hyper_lora_input: Conditioning vector
        
        Returns:
            Base model output with LoRA adaptation
        """
        self.register_hooks(hyper_lora_input)
        x = self.base_model(x)
        self.unregister_hooks()
        return x
    
    def use(self, hyper_lora_input: Tensor) -> HyperLoRAHookContext:
        """Context manager for HyperLoRA forward pass.
        
        Args:
            hyper_lora_input: Conditioning vector
        
        Returns:
            Context manager
        """
        return HyperLoRAHookContext(self, hyper_lora_input)
    
    def reg_loss(self, lora_weights=None, reg_type=None):
        """Compute regularization loss on LoRA deltas.
        
        Args:
            lora_weights: Pre-computed LoRA weights (optional, uses cached)
            reg_type: 'frobenius' | 'b_only' | 'nuclear' (optional, uses self.reg_type)
        
        Returns:
            Scalar regularization loss
        """
        if lora_weights is None:
            lora_weights = self._cached_lora_weights
        
        if lora_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if reg_type is None:
            reg_type = self.reg_type
        
        total_reg = 0.0
        count = 0
        
        for layers, lora_weight in lora_weights:
            B, _ = lora_weight.shape
            in_dim, out_dim = layers.in_features, layers.out_features
            
            # Reshape to matrices
            lora_A = lora_weight[:, :in_dim*self.rank].reshape(B, in_dim, self.rank)
            lora_B = lora_weight[:, in_dim*self.rank:].reshape(B, self.rank, out_dim)
            
            if reg_type == 'frobenius':
                # Penalize ||A @ B||_F^2
                delta_w = lora_A @ lora_B  # (B, in_dim, out_dim)
                reg = torch.mean(torch.norm(delta_w.reshape(B, -1), p=2, dim=-1) ** 2)
            elif reg_type == 'b_only':
                # Penalize ||B||_F^2 (common in LoRA literature)
                reg = torch.mean(torch.norm(lora_B.reshape(B, -1), p=2, dim=-1) ** 2)
            elif reg_type == 'nuclear':
                # Penalize nuclear norm (sum of singular values)
                delta_w = lora_A @ lora_B
                # Nuclear norm via SVD
                reg = 0.0
                for b in range(B):
                    _, s, _ = torch.svd(delta_w[b])
                    reg += s.sum()
                reg = reg / B
            else:
                reg = torch.tensor(0.0, device=lora_weight.device)
            
            total_reg += reg
            count += 1
        
        return total_reg / max(count, 1)
    
    @torch.no_grad()
    def unmerge_weight(self):
        """Restore base model weights to their pre-merge state."""
        if self._original_weights is None:
            return
        for (layers, _), orig in zip(self.modules_list, self._original_weights):
            layers.weight.data.copy_(orig)

    @torch.no_grad()
    def merge_weight(self, hyper_lora_input: Tensor):
        """Merge LoRA weights into base model (for inference).

        Saves original weights on first call and restores them before each
        subsequent call, so repeated calls do not accumulate LoRA deltas.

        Supports either:
          - shared conditioning: (1, input_dim)
          - per-module conditioning: (1, M, input_dim)

        Args:
            hyper_lora_input: Conditioning vector(s) (batch size must be 1)

        Returns:
            Base model with merged weights
        """
        assert hyper_lora_input.shape[0] == 1, "only support batch size 1 for merging weight"

        # Save original weights on the first call
        if self._original_weights is None:
            self._original_weights = [
                layers.weight.data.clone() for layers, _ in self.modules_list
            ]
        else:
            # Restore original weights before applying new LoRA deltas
            self.unmerge_weight()

        lora_weights = self.prepare_lora_weights(hyper_lora_input)
        for idx, (layers, lora_weight) in enumerate(lora_weights):
            in_dim, out_dim = layers.in_features, layers.out_features
            lora_A = lora_weight[:, :in_dim*self.rank].reshape(1, in_dim, self.rank)
            lora_B = lora_weight[:, in_dim*self.rank:].reshape(1, self.rank, out_dim)
            delta_w = (lora_A @ lora_B) / float(self.rank) * self.alpha
            # Apply per-module gate (must match training hook behavior)
            if self.use_gates:
                gate_value = torch.sigmoid(self.gates[idx])
                delta_w = gate_value * delta_w
            delta_w = delta_w.squeeze(0)
            layers.weight.data += delta_w.T
        return self.base_model


if __name__ == "__main__":
    simple_mlp = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 5)
    )
    hyper_lora = HyperLoRA(
        input_size=5,
        base_model=simple_mlp,
        rank=2,
        max_modules=-1,
        hidden_layers=[256],
        use_tanh=True,
        scale_A=0.1,
        scale_B=0.1,
        use_gates=True,
    )
    # print(hyper_lora(torch.randn(1, 256), torch.randn(1, 5)))
    inputs = torch.ones(1,256)
    print(simple_mlp(inputs))
    nn.init.xavier_normal_(hyper_lora.modules_list[0][1].fc.weight[:, :])
    with hyper_lora.use(torch.randn((1,5))):
        print(simple_mlp(inputs))
