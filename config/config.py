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
    preference_cfg: Dict[str, Any] = field(default_factory=dict)
    data_caching: bool = True
    cat_side: bool = False
    # New: MANO-angle-space bias configuration
    bias_cfg: Optional[Dict[str, Any]] = None
    num_hypernet_pairs: int = 4
    pair_subsample: bool = False

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
    # Stability improvements
    train_init: bool = False  # Use Wasserstein initialization (expensive)
    init_method: str = 'xavier'  # 'xavier' | 'wasserstein'
    use_tanh: bool = False  # Bound LoRA outputs with tanh
    scale_A: float = 1.0  # Scale for A matrix (after tanh)
    scale_B: float = 1.0  # Scale for B matrix (after tanh)
    use_gates: bool = False  # Per-module learnable gates
    gate_init: float = 0.0  # Initial gate value (before sigmoid)
    max_norm: Optional[float] = None  # Max Frobenius norm for delta weights
    reg_weight: float = 0.0  # Weight for regularization loss
    reg_type: str = 'frobenius'  # 'frobenius' | 'b_only' | 'nuclear'

@dataclass
class IKNetCfg:
    enabled: bool = False
    hidden_layers: list = field(default_factory=lambda: [256, 256, 256])
    output_pca: bool = False
    n_comps: int = 21
    
@dataclass
class PreferenceExtractorConfig:
    enabled: bool = False
    # Legacy TransformerExtractor settings
    input_dim: int = field(default_factory=int)
    hidden_dim: int = 32
    num_heads: int = 4
    n_layers: int = 4
    max_tokens: int = 10
    min_tokens: int = 5
    # New PairedPoseExtractor settings
    use_paired_pose_extractor: bool = False
    pair_fusion_dim: int = 256
    output_dim: int = 256
    dropout: float = 0.1

    # NEW: Shared-transformer, per-module attention pooling for HyperLoRA
    per_module_pooling: bool = False

@dataclass
class ModelConfig:
    type : str = 'MLP'
    input_size: int = field(default_factory=int)
    input_channel: int = field(default_factory=int)
    hidden_layers: list = field(default_factory=lambda: [256, 256])
    output_size: int = field(default_factory=int)
    hypernet_cfg: HypernetCfg = field(default_factory=HypernetCfg)
    preference_extractor_cfg: PreferenceExtractorConfig = field(default_factory=PreferenceExtractorConfig)
    iknet_cfg: IKNetCfg = field(default_factory=IKNetCfg)
    joints_layout: Optional[list] = field(default_factory=lambda: [[0,1,2,3], [4,5,6], [7,8,9], [10], [11]])
    full_input: bool = False # Only for CombinedModel, whether to use full input for each sub-model
    num_fuse_layer: int = 0  # Number of fusion layers, only for CombinedModel

@dataclass
class LoggerConfig:
    type: list[str] = field(default_factory=lambda: ['terminal','file'])  # 'wandb', 'file', 'terminal'.
    project: str = 'Calibration_net'


@dataclass
class LRSchedulerConfig:
    enabled: bool = False
    total_steps: Optional[int] = None
    warmup_steps: int = 0
    peak_lr: Optional[float] = None  # None => fallback to trainer.learning_rate
    final_lr: float = 0.0
    start_lr: float = 0.0


@dataclass
class TrainerConfig:
    expname: str = field(default_factory=str)
    batch_size: int = field(default_factory=int)
    learning_rate: float = field(default_factory=float)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    weight_decay: float = field(default_factory=float)
    grad_clipping: float = field(default_factory=float)
    calibnet_warmup_epochs: int = 0
    num_epochs: int = field(default_factory=int)
    val_split: float = 0.2
    device: str = 'cuda'  # 'cpu', 'cuda', or 'xpu'
    loss_type: str = 'composite'  # 'qpos', 'geort', 'composite'
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'qpos':1.0, 'geort': 0.01, 'alignment':0.1})
    loss_weights_decay: Dict[str, Any] = field(default_factory=lambda: {'qpos': 0.00, 'geort': 0.02, 'alignment': 0.02})
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