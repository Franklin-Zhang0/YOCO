import wandb
import datetime
import uuid
from pprint import pprint
from omegaconf import OmegaConf


def get_logger_cls(logger_type):
    if logger_type == 'wandb':
        return WandbLogger
    elif logger_type == 'terminal':
        return TerminalLogger
    else:
        raise ValueError("Invalid logger type")

class BaseLogger:
    def __init__(self):
        raise NotImplementedError
    def log(self, metrics, step):
        raise NotImplementedError
    
class WandbLogger(BaseLogger):
    def __init__(self, config):
        logger_cfg = config.trainer.logger
        wandb_id = str(uuid.uuid4())
        run_name = config.trainer.expname
        self.logger = wandb.init(id=wandb_id, project=logger_cfg.project, name=run_name, config=OmegaConf.to_container(config))
    
    def log(self, metrics, step):
        self.logger.log(metrics, step=step)
        
class TerminalLogger(BaseLogger):
    def __init__(self, config):
        pprint(OmegaConf.to_container(config))
        pass
    
    def log(self, metrics, step):
        print(f"Step: {step}, Metrics: {metrics}")