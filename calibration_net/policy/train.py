from .trainer import Trainer
from omegaconf import OmegaConf
from config.config import Config

def main():
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)
    del cli_args.config  # Remove config from CLI args to avoid conflicts
    default_config = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_config, file_config, cli_args)
    trainer = Trainer(args=config)
    trainer.train()
    
if __name__ == "__main__":
    main()