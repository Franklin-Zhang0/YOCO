from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .config import Config, TrainerConfig

@dataclass
class GRPOConfig(TrainerConfig):
    group_size: int = 8
    eps_clip: float = 0.2
    std: float = 0.1
    std_decay: float = 1.0
    mano_comp: int = 10
    scale_kl_coeff: float = 1e-6
    mano_kl_coeff: float = 1e-6

@dataclass
class RewarderConfig:
    robot_name: str = 'omni'
    hand_type: str = 'right'
    retargeting_type: str = 'dexpilot'
    urdf_path: Optional[str] = None  # Path to dex_retargeting URDF directory
    urdf_file: Optional[str] = None  # Path to specific robot URDF file for FK (if None, uses $DEX_URDF_PATH/robots/hands/{robot_name}_hand/{robot_name}_hand_{hand_type}.urdf)

@dataclass
class GRPOFullConfig(Config):
    trainer: GRPOConfig = field(default_factory=GRPOConfig)
    rewarder_config: RewarderConfig = field(default_factory=RewarderConfig)

