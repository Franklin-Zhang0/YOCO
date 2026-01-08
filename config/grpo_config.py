from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .config import Config, TrainerConfig

@dataclass
class GRPOConfig(TrainerConfig):
    group_size: int = 8
    eps_clip: float = 0.2
    beta: float = 0.01
    std: float = 0.1
    std_decay: float = 1.0

@dataclass
class RewarderConfig:
    robot_name: str = 'omni'
    hand_type: str = 'right'
    retargeting_type: str = 'dexpilot'
    urdf_path: Optional[str] = None

@dataclass
class GRPOFullConfig(Config):
    trainer: GRPOConfig = field(default_factory=GRPOConfig)
    rewarder_config: RewarderConfig = field(default_factory=RewarderConfig)

