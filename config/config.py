from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    path: str = field(default_factory=str)
    input_keys: str = field(default_factory=str)
    hypernet_input_keys: str = field(default_factory=str)
    output_keys: str = field(default_factory=str)
    side: str = 'left'
    augmentations: Dict[str, float] = field(default_factory=dict)
    obs_horizon: int = 1
    action_horizon: int = 1
    preprocessing_x: list = field(default_factory=list)
    preprocessing_y: list = field(default_factory=list)

@dataclass
class ValDatasetConfig:
    path: str = field(default_factory=str)
    
@dataclass
class HypernetCfg:
    input_size: int = field(default_factory=int)
    max_modules: int = -1
    rank: int = 2
    alpha: float = 2.0

@dataclass
class ModelConfig:
    type : str = 'MLP'
    input_size: int = field(default_factory=int)
    input_channel: int = field(default_factory=int)
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    output_size: int = field(default_factory=int)
    hypernet_cfg: HypernetCfg = field(default_factory=HypernetCfg)

@dataclass
class TrainerConfig:
    expname: str = field(default_factory=str)
    batch_size: int = field(default_factory=int)
    learning_rate: float = field(default_factory=float)
    grad_clipping: float = field(default_factory=float)
    calibnet_warmup_epochs: int = 0
    num_epochs: int = field(default_factory=int)
    val_split: float = 0.2
    device: str = 'cuda'  # 'cpu', 'cuda', or 'xpu'
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    save_interval: int = 5  # Save model every n epochs
    calib_finetune: bool = False
    
@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: ValDatasetConfig = field(default_factory=ValDatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)