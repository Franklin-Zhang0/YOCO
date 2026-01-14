from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .config import Config, TrainerConfig

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning"""
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # e.g., ["mano_regressor.*", "fusion_attention.*"]
    train_finger_scaling: bool = True  # Whether to also train finger_scaling parameter

@dataclass
class GRPOConfig(TrainerConfig):
    group_size: int = 8
    eps_clip: float = 0.2
    std: float = 0.1
    std_decay: float = 1.0
    mano_comp: int = 10
    scale_kl_coeff: float = 1e-6
    mano_kl_coeff: float = 1e-6
    lora: LoRAConfig = field(default_factory=LoRAConfig)

@dataclass
class RewarderConfig:
    robot_name: str = 'omni'
    hand_type: str = 'right'
    retargeting_type: str = 'dexpilot'
    urdf_path: Optional[str] = None  # Path to dex_retargeting URDF directory
    urdf_file: Optional[str] = None  # Path to specific robot URDF file for FK (if None, uses $DEX_URDF_PATH/robots/hands/{robot_name}_hand/{robot_name}_hand_{hand_type}.urdf)
    
    # Performance options
    retarget_workers: int = 0  # Number of parallel workers for retargeting (0 = single-process)
    retarget_min_chunk: int = 32  # Minimum batch size to enable multiprocessing
    target_fk_cache_size: int = 128  # LRU cache size for target FK results (0 = disabled)

@dataclass
class GRPOFullConfig(Config):
    trainer: GRPOConfig = field(default_factory=GRPOConfig)
    rewarder_config: RewarderConfig = field(default_factory=RewarderConfig)

