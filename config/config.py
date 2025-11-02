from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    path: str = field(default_factory=str)
    input_keys: str = field(default_factory=str)
    hypernet_input_keys: str = field(default_factory=str)
    output_keys: str = field(default_factory=str)
    augmentations: Dict[str, float] = field(default_factory=dict)
    preprocessing_x: list = field(default_factory=list)
    preprocessing_y: list = field(default_factory=list)
    cat_side: bool = False

@dataclass
class ValDatasetConfig:
    path: str = field(default_factory=str)
    
@dataclass
class HypernetCfg:
    input_size: int = field(default_factory=int)
    max_modules: int = -1
    rank: int = 2
    alpha: float = 2.0
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    name_filter: list[str] = field(default_factory=list)
    drop_out: float = 0.0

@dataclass
class IKNetCfg:
    enabled: bool = False
    hidden_layers: list = field(default_factory=lambda: [256, 256, 256])
    output_pca: bool = False
    n_comps: int = 21

@dataclass
class ModelConfig:
    type : str = 'MLP'
    input_size: int = field(default_factory=int)
    input_channel: int = field(default_factory=int)
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    output_size: int = field(default_factory=int)
    hypernet_cfg: HypernetCfg = field(default_factory=HypernetCfg)
    iknet_cfg: IKNetCfg = field(default_factory=IKNetCfg)
    joints_layout: Optional[list] = field(default_factory=lambda: [[0,1,2,3], [4,5,6], [7,8,9], [10], [11]])
    full_input: bool = False # Only for CombinedModel, whether to use full input for each sub-model
    num_fuse_layer: int = 0  # Number of fusion layers, only for CombinedModel

@dataclass
class LoggerConfig:
    type: list[str] = field(default_factory=lambda: ['terminal'])  # 'wandb', 'tensorboard', 'csv', etc.
    project: str = 'Calibration_net'

@dataclass
class TrainerConfig:
    expname: str = field(default_factory=str)
    batch_size: int = field(default_factory=int)
    learning_rate: float = field(default_factory=float)
    weight_decay: float = field(default_factory=float)
    grad_clipping: float = field(default_factory=float)
    calibnet_warmup_epochs: int = 0
    num_epochs: int = field(default_factory=int)
    val_split: float = 0.2
    device: str = 'cuda'  # 'cpu', 'cuda', or 'xpu'
    loss_type: str = 'composite'  # 'qpos', 'geort', 'composite'
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'qpos':1.0, 'geort': 0.0, 'alignment':0.1})
    loss_weights_decay: Dict[str, Any] = field(default_factory=lambda: {'qpos': -0.01, 'geort': 0.00, 'alignment': 0.01})
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    fk_model_path: Optional[str] = "expdata/fk/model/omni.pth"
    fk_ds_path: Optional[str] = "expdata/fk/dataset/omni.npz"
    save_interval: int = 5  # Save model every n epochs
    calib_finetune: bool = False
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    
@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: ValDatasetConfig = field(default_factory=ValDatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)