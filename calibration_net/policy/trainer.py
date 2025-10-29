import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import pickle
from .dataset import TeleopDataset, SequenceDataset
from datetime import datetime
from .model import Model, load_model
from .models.fk_model import FKModel, FKTrainer
from calibration_net.utils import walk_dict, create_dict_from_flattened
from copy import deepcopy
from .logger import get_logger_cls 
from .loss import get_geort_loss
from contextlib import nullcontext

class Trainer():
    def __init__(self, args):
        self.args = args
        self.create_logger()

        self.trainer_args = args.trainer
        
        self.device = self.trainer_args.device
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.expname = self.trainer_args.expname

        self.load_dataset()
        example = self.trainset[0]
        print("Example input shape:", example['input'].shape)
        
        if self.args.model.input_size < 45:
            self.args.model.iknet_cfg.output_pca = True

        if self.args.model.iknet_cfg.enabled:
            self.args.model.input_size = self.args.model.iknet_cfg.n_comps if self.args.model.iknet_cfg.output_pca else 45
        else:
            self.args.model.input_size = example['input'].shape[-2]
        self.args.model.input_channel = example['input'].shape[-1]
        self.args.model.output_size = example['output'].shape[-1]
        
        if 'hypernet_input' in example:
            self.args.model.hypernet_cfg.input_size = example['hypernet_input'].shape[-1]
        else:
            self.args.model.hypernet_cfg.input_size = 0

        if self.trainer_args.load_path:
            print(f"Loading model from {self.trainer_args.load_path}")
            self.model, normalization_stats, loaded_cfg = load_model(self.trainer_args.load_path)
            self.dataset.normalization_stats = normalization_stats
            if loaded_cfg['model']['output_size'] != self.args.model.output_size:
                self.switch_last_layer()
                loaded_cfg['model']['output_size'] = self.args.model.output_size
            # assert all other model configurations match
            assert loaded_cfg['model'] == self.args.model
        else:
            self.model = Model(**self.args.model)
            
        self.model.to(self.device)

        if self.trainer_args.loss_type in ['geort', 'composite']:
            print("Using FK model for loss computation")
            self.fk_model = FKModel(self.trainer_args.joint_layout).to(self.device)
            self.fk_trainer = FKTrainer(self.fk_model, self.trainer_args.fk_ds_path, self.trainer_args.fk_model_path, self.device)
            if os.path.exists(self.trainer_args.fk_model_path):
                print(f"Loading FK model from {self.trainer_args.fk_model_path}")
                fk_state_dict = torch.load(self.trainer_args.fk_model_path, map_location=self.device)
                self.fk_model.load_state_dict(fk_state_dict)
            else:
                self.fk_trainer.train()
            self.fk_model.eval()
        
        if self.trainer_args.calib_finetune:
            self.set_trainable(main_network=False, calib=True)
        elif self.trainer_args.calibnet_warmup_epochs == 0:
            self.set_trainable(main_network=True, calib=True)
        elif self.trainer_args.calibnet_warmup_epochs > 0:
            self.set_trainable(main_network=True, calib=False)
        self.create_optimizer()

    def set_trainable(self, main_network=True, calib=True):
        for param in self.model.parameters():
            param.requires_grad = main_network
        if self.model.calib_block is not None:
            for param in self.model.calib_block.parameters():
                param.requires_grad = calib

    def switch_last_layer(self):
        self.model.fc = nn.Linear(self.model.fc.in_features, self.args.model.output_size)
        self.model.network[-1] = self.model.fc

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_args.learning_rate)
        self.loss_function = nn.MSELoss()
    
    def load_dataset(self):
        self.dataset = SequenceDataset(**self.args.dataset)
        if self.args.val_dataset.path == "" or self.args.val_dataset.path is None:
            val_size = int(len(self.dataset) * self.trainer_args.val_split)
            # self.trainset, self.valset = torch.utils.data.random_split(self.dataset, [len(self.dataset) - val_size, val_size])
            self.trainset = torch.utils.data.Subset(self.dataset, range(len(self.dataset) - val_size))  # Ensure the dataset is not modified
            self.valset = torch.utils.data.Subset(self.dataset, range(len(self.dataset) - val_size, len(self.dataset)))
        else:
            self.trainset = self.dataset
            val_dataset_args = deepcopy(self.args.dataset)
            val_dataset_args.path = self.args.val_dataset.path
            self.valset = SequenceDataset(**val_dataset_args, is_train=False)
            self.valset.normalization_stats = self.trainset.normalization_stats  # use the same stats for validation set
        # self.trainset = torch.utils.data.Subset(self.dataset, range(256))  
        # self.valset = torch.utils.data.Subset(self.dataset, range(128))
        
        self.trainloader = DataLoader(self.trainset, batch_size=self.trainer_args.batch_size, shuffle=True, num_workers=4)
        self.valloader = DataLoader(self.valset, batch_size=self.trainer_args.batch_size, shuffle=False, num_workers=4)
        
    def get_loss(self, data, result):
        loss_type = self.trainer_args.loss_type
        if loss_type == 'qpos':
            return {'loss': result['loss']}
        elif loss_type == 'geort':
            loss = get_geort_loss(points=data['input'], side=data['side'], ik_model=self.model, fk_model=self.fk_model, robot_point_cloud=self.fk_trainer.robot_point_cloud, loss_weights=self.trainer_args.loss_weights)
            return loss
        elif loss_type == 'composite':
            geo_rt_loss = get_geort_loss(points=data['input'], side=data['side'], ik_model=self.model, fk_model=self.fk_model, robot_point_cloud=self.fk_trainer.robot_point_cloud, loss_weights=self.trainer_args.loss_weights)
            geo_rt_loss['qpos_loss'] = result['loss']
            geo_rt_loss['loss'] = result['loss'] * self.trainer_args.loss_weights['qpos'] + geo_rt_loss['loss']
            return geo_rt_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def train_step(self, data, do_backward=True):
        x, y, side = data['input'], data['output'], data['side']
        x[..., 0] *= -side.unsqueeze(-1)  # flip x axis based on side
        x, y, side = x.float().to(self.device), y.float().to(self.device), side.to(self.device)
        
        if 'ik_y' in data:
            ik_y = data['ik_y'].float().to(self.device)
        else:
            ik_y = None

        if 'hypernet_input' in data:
            hypernet_input = data['hypernet_input'].float().to(self.device)
        else:
            hypernet_input = None

        step_info = {}

        context = self.model.hyperlora.use(hypernet_input) if hypernet_input is not None else nullcontext()
        with context:
            if do_backward: #training
                self.global_steps += 1
                # result = self.model(x, y, hypernet_input, side=side, ik_y=ik_y)
                result = self.model(x, y, side=side, ik_y=ik_y)
                loss = self.get_loss(data, result)
                self.optimizer.zero_grad()
                loss['loss'].backward()
                nn.utils.clip_grad_value_(self.model.parameters(), self.trainer_args.grad_clipping)
                self.optimizer.step()
                for key, value in loss.items():
                    step_info[f'train/{key}'] = value.item()
                self.log_step(step_info, self.global_steps)
            else: # evaluating
                # pred_y = self.model(x, hypernet_x=hypernet_input, side=side)
                pred_y = self.model(x, side=side)
                loss = self.loss_function(pred_y, y)
                mae = torch.mean(torch.abs(pred_y - y))
                step_info['val/mae'] = mae.item()
                step_info['val/loss'] = loss.item()

        return step_info
    
    def train_epoch(self):
        self.model.train()
        data_tqdm = tqdm(self.trainloader, desc='Training')
        train_loss = []
        info = {}
        for data in data_tqdm:
            step_info = self.train_step(data)
            loss = step_info['train/loss']
            train_loss.append(loss)
            data_tqdm.set_postfix(loss=loss)

        info['train/loss'] = sum(train_loss) / len(train_loss)
        print(f"Training Loss: {info['train/loss']}")
        info['val/loss'] = self.validate()
        return info

    def validate(self):
        print("Start Validation")
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        with torch.no_grad():
            for data in tqdm(self.valloader):
                step_info = self.train_step(data, do_backward=False)
                total_loss += step_info['val/loss']
                total_mae += step_info['val/mae']
        avg_loss = total_loss / len(self.valloader)
        avg_mae = total_mae / len(self.valloader)
        self.log_step({'val/loss': avg_loss, 'val/mae': avg_mae}, self.global_steps)
        print(f"Validation MAE: {avg_mae}")
        print(f"Validation Loss: {avg_loss}")
        return avg_loss

    def train(self):
        self.global_steps = 0
        num_epoch = self.trainer_args.num_epochs
        best_val_loss = float('inf')
        for i in range(num_epoch):
            print(f"Epoch {i+1}/{num_epoch}")
            if self.trainer_args.calibnet_warmup_epochs > 0 and i == self.trainer_args.calibnet_warmup_epochs:
                self.set_trainable(main_network=True, calib=True)

            train_info = self.train_epoch()
            if self.trainer_args.save_path:
                path = os.path.join(self.trainer_args.save_path, self.expname + "_" + self.timestamp, f'latest')
                self.save_model(path)
                if train_info['val/loss'] < best_val_loss:
                    best_val_loss = train_info['val/loss']
                    best_path = os.path.join(self.trainer_args.save_path, self.expname + "_" + self.timestamp, f'best')
                    self.save_model(best_path)
                if (i+1) % self.trainer_args.save_interval == 0:
                    save_path = os.path.join(self.trainer_args.save_path, self.expname + "_" + self.timestamp, f'epoch_{i+1}')
                    self.save_model(save_path)
                
        print("Training complete.")

    def create_logger(self):
        self.loggers = []
        logger_cfg = self.args.trainer.logger
        for log_type in logger_cfg.type:
            cls = get_logger_cls(log_type)
            self.loggers.append(cls(self.args))

    def log_step(self, log_dict, step):
        for logger in self.loggers:
            logger.log(log_dict, step=step)

    def save_model(self, save_dir):
        save_dir = os.path.expanduser(save_dir)
        
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'model.pth')
        stats_path = os.path.join(save_dir, 'norm_stats.pkl')
        torch.save(self.model.state_dict(), model_path)
        
        with open(stats_path, 'wb') as f:
            norm_stats = {k: v.tolist() for k,v in walk_dict(self.dataset.normalization_stats)}  # convert to list for better backward compatibility for numpy versions
            norm_stats = create_dict_from_flattened(norm_stats)

            pickle.dump(norm_stats, f)
            
        # save config
        config_path = os.path.join(save_dir, 'config.yaml')
        from  omegaconf import OmegaConf
        with open(config_path, 'w') as f:
            cfg = OmegaConf.to_container(self.args)
            yaml.safe_dump(cfg, f)
