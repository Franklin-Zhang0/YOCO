import wandb
import datetime
import uuid
import os
from pprint import pprint
from omegaconf import OmegaConf


def get_logger_cls(logger_type):
    if logger_type == 'wandb':
        return WandbLogger
    elif logger_type == 'terminal':
        return TerminalLogger
    elif logger_type == 'file':
        return FileLogger
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

class FileLogger(BaseLogger):
    def __init__(self, config):
        trainer_cfg = config.trainer
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        expname = trainer_cfg.expname
        
        # Construct the save directory path (same as in trainer.py)
        if trainer_cfg.save_path:
            save_dir = os.path.join(trainer_cfg.save_path, expname + "_" + timestamp)
            os.makedirs(save_dir, exist_ok=True)
            self.log_file_path = os.path.join(save_dir, 'log.txt')
        else:
            # Fallback to current directory if no save_path is specified
            self.log_file_path = 'log.txt'
        
        # Create/open the log file and write header
        with open(self.log_file_path, 'w') as f:
            f.write(f"Training Log - {expname}\n")
            f.write(f"Started at: {timestamp}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, metrics, step):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, 'a') as f:
            f.write(f"[{timestamp}] Step {step}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")