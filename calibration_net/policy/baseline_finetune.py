from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import trange


@dataclass
class BaselineConfig:
    method: str = "lora"
    num_calib_actions: int = 4
    steps: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    target_modules: list[str] = field(default_factory=list)
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.1
    lora_use_gates: bool = True
    lora_gate_init: float = 0.0


class StaticLoRAAdapter(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Iterable[str],
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.1,
        use_gates: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.target_modules = list(target_modules)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.use_gates = bool(use_gates)
        self.gate_init = float(gate_init)
        self.adapters = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        self.target_names: list[str] = []
        self._hooks = []
        self._build()

    def _matches(self, name: str) -> bool:
        if not self.target_modules:
            return True
        return any(pattern in name for pattern in self.target_modules)

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace(".", "__")

    def _build(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

        for name, module in self.base_model.named_modules():
            if not isinstance(module, nn.Linear) or not self._matches(name):
                continue

            adapter = nn.ModuleDict(
                {
                    "A": nn.Linear(module.in_features, self.rank, bias=False),
                    "B": nn.Linear(self.rank, module.out_features, bias=False),
                }
            )
            nn.init.kaiming_uniform_(adapter["A"].weight, a=5**0.5)
            nn.init.zeros_(adapter["B"].weight)
            safe_name = self._safe_name(name)
            self.adapters[safe_name] = adapter
            gate = None
            if self.use_gates:
                gate = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
                self.gates[safe_name] = gate
            self.target_names.append(name)

            def hook(_module, inputs, output, adapter=adapter, gate=gate):
                x = self.dropout(inputs[0])
                lora_output = adapter["B"](adapter["A"](x)) * self.scaling
                if gate is not None:
                    lora_output = torch.sigmoid(gate) * lora_output
                return output + lora_output

            self._hooks.append(module.register_forward_hook(hook))

        if len(self.adapters) == 0:
            raise ValueError(
                f"No nn.Linear modules matched target_modules={self.target_modules}"
            )

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class ResidualLinearBaseline(nn.Module):
    def __init__(self, base_model: nn.Module, input_shape: tuple[int, ...], output_size: int):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        input_dim = 1
        for dim in input_shape:
            input_dim *= int(dim)
        self.residual = nn.Linear(input_dim, int(output_size))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def train_forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            base_pred = self.base_model(x)
        pred_y = base_pred + self.residual(x.reshape(x.shape[0], -1))
        loss = F.mse_loss(pred_y, y, reduction="none")
        return {"pred_y": pred_y, "loss": loss}

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            base_pred = self.base_model(x)
        return base_pred + self.residual(x.reshape(x.shape[0], -1))


def configure_baseline_model(
    model: nn.Module,
    method: str,
    input_shape: tuple[int, ...],
    output_size: int,
    target_modules: Iterable[str],
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.1,
    lora_use_gates: bool = True,
    lora_gate_init: float = 0.0,
) -> nn.Module:
    method = method.lower()
    if method == "lora":
        return StaticLoRAAdapter(
            model,
            target_modules=target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_gates=lora_use_gates,
            gate_init=lora_gate_init,
        )

    if method == "full_ft":
        for param in model.parameters():
            param.requires_grad = True
        return model

    if method == "residual_linear":
        return ResidualLinearBaseline(model, input_shape=input_shape, output_size=output_size)

    if method == "last_layer":
        for param in model.parameters():
            param.requires_grad = False
        trainable = 0
        for name, module in model.named_modules():
            if "mano_regressor" not in name or not isinstance(module, nn.Linear):
                continue
            # In Combined_CNN_MLP_MIXER, the last regressor layer outputs 12 dims.
            if module.out_features == 12:
                for param in module.parameters():
                    param.requires_grad = True
                    trainable += param.numel()
        if trainable == 0:
            raise ValueError("No last-layer mano_regressor Linear modules were found")
        return model

    raise ValueError(
        f"Unknown baseline method '{method}'. Expected lora, full_ft, "
        "residual_linear, or last_layer."
    )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _keep_frozen_modules_eval(module: nn.Module) -> None:
    for child in module.modules():
        params = list(child.parameters(recurse=True))
        if params and not any(param.requires_grad for param in params):
            child.eval()


def adaptation_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = 1.0,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "train_forward"):
        result = model.train_forward(x, y)
    else:
        result = model(x, y)
    loss = result["loss"].mean()
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip),
        )
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    if hasattr(model, "predict"):
        return model.predict(x)
    result = model(x)
    if isinstance(result, dict):
        return result["pred_y"]
    return result


def adapt_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float | None = 1.0,
    show_progress: bool = False,
    progress_desc: str = "adapt",
) -> tuple[list[float], float]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for baseline adaptation")
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    losses: list[float] = []
    start = perf_counter()
    model.train()
    _keep_frozen_modules_eval(model)
    iterator = trange(
        int(steps),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    )
    for _ in iterator:
        loss = adaptation_step(model, x, y, optimizer, grad_clip=grad_clip)
        losses.append(loss)
        if show_progress:
            iterator.set_postfix(loss=f"{loss:.4e}")
    return losses, perf_counter() - start
