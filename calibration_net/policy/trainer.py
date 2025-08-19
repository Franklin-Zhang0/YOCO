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
from calibration_net.utils import walk_dict, create_dict_from_flattened

class Trainer():
    def __init__(self, args):
        self.args = args
        self.trainer_args = args.trainer
        
        self.device = self.trainer_args.device
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.expname = self.trainer_args.expname

        self.load_dataset()
        example = self.trainset[0]
        print("Example input shape:", example['input'].shape)
        self.args.model.input_size = example['input'].shape[-1]
        self.args.model.input_channel = example['input'].shape[-2]

        self.model = Model(**self.args.model)

        if self.trainer_args.load_path:
            print(f"Loading model from {self.trainer_args.load_path}")
            self.model, normalization_stats, loaded_cfg = load_model(self.trainer_args.load_path)
            if loaded_cfg['model']['output_size'] != self.args.model.output_size:
                self.switch_last_layer()
                loaded_cfg['model']['output_size'] = self.args.model.output_size
            # assert all other model configurations match
            assert loaded_cfg['model'] == self.args.model
            
        self.model.to(self.device)
        
        if self.trainer_args.calib_finetune:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.calib_block.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.model.calib_block.parameters():
                param.requires_grad = True
        self.create_optimizer()

    def switch_last_layer(self):
        self.model.fc = nn.Linear(self.model.fc.in_features, self.args.model.output_size)
        self.model.network[-1] = self.model.fc

    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.trainer_args.learning_rate)
        self.loss_function = nn.MSELoss()
    
    def load_dataset(self):
        self.dataset = SequenceDataset(**self.args.dataset)
        val_size = int(len(self.dataset) * self.trainer_args.val_split)
        # self.trainset, self.valset = torch.utils.data.random_split(self.dataset, [len(self.dataset) - val_size, val_size])
        self.trainset = torch.utils.data.Subset(self.dataset, range(len(self.dataset) - val_size))  # Ensure the dataset is not modified
        self.valset = torch.utils.data.Subset(self.dataset, range(len(self.dataset) - val_size, len(self.dataset)))

    def train_step(self, data, do_backward=True):
        x, y = data['input'], data['output']
        x, y = x.float().to(self.device), y.float().to(self.device)
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        
        if do_backward:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        step_info = {
            'loss': loss.item(),
            'mae': mae.item()
        }   
        return step_info
    
    def train_epoch(self):
        self.model.train()
        self.dataloader = DataLoader(self.trainset, batch_size=self.trainer_args.batch_size, shuffle=True)
        data_tqdm = tqdm(self.dataloader, desc='Training')
        train_loss = []
        info = {}
        for data in data_tqdm:
            step_info = self.train_step(data)
            loss = step_info['loss']
            train_loss.append(loss)
            data_tqdm.set_postfix(loss=loss)
        info['val_loss'] = self.validate()
        info['train_loss'] = sum(train_loss) / len(train_loss)
        return info

    def validate(self):
        self.model.eval()
        self.dataloader = DataLoader(self.valset, batch_size=4, shuffle=False)
        total_loss = 0.0
        total_mae = 0.0
        with torch.no_grad():
            for data in self.dataloader:
                step_info = self.train_step(data, do_backward=False)
                total_loss += step_info['loss']
                total_mae += step_info['mae']
        avg_loss = total_loss / len(self.dataloader)
        avg_mae = total_mae / len(self.dataloader)
        print(f"Validation MAE: {avg_mae}")
        print(f"Validation Loss: {avg_loss}")
        return avg_loss

    def train(self):
        num_epoch = self.trainer_args.num_epochs
        best_val_loss = float('inf')
        for i in range(num_epoch):
            print(f"Epoch {i+1}/{num_epoch}")
            train_info = self.train_epoch()
            if self.trainer_args.save_path:
                path = os.path.join(self.trainer_args.save_path, self.expname + "_" + self.timestamp, f'latest')
                self.save_model(path)
                if train_info['val_loss'] < best_val_loss:
                    best_val_loss = train_info['val_loss']
                    best_path = os.path.join(self.trainer_args.save_path, self.expname + "_" + self.timestamp, f'best')
                    self.save_model(best_path)
                if (i+1) % self.trainer_args.save_interval == 0:
                    save_path = os.path.join(self.trainer_args.save_path, self.expname + "_" + self.timestamp, f'epoch_{i+1}')
                    self.save_model(save_path)
                
        print("Training complete.")
        
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
