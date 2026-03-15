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
from .loss import get_geort_loss, get_alignment_loss, get_refine_composite_loss
from .lr_scheduler import build_warmup_cosine_scheduler
from contextlib import nullcontext
from omegaconf import OmegaConf

class Trainer():
    def __init__(self, args):
        self.args = args
        self.create_logger()

        self.trainer_args = args.trainer
        
        self.device = self.trainer_args.device
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.expname = self.trainer_args.expname
        
        self.args.dataset.preference_cfg = {
            'min_tokens': self.args.model.preference_extractor_cfg.min_tokens,
            'max_tokens': self.args.model.preference_extractor_cfg.max_tokens,
        }

        self.load_dataset()
        example = self.trainset[0]
        print("Example input shape:", example['input'].shape)
        print("Example output shape:", example['output'].shape)
        
        if self.args.model.input_size < 45:
            self.args.model.iknet_cfg.output_pca = True

        if self.args.model.iknet_cfg.enabled:
            self.args.model.input_size = self.args.model.iknet_cfg.n_comps if self.args.model.iknet_cfg.output_pca else 45
            self.args.model.input_channel = 1
        else:
            self.args.model.input_size = torch.prod(torch.tensor(example['input'].shape[:-1])).item()
            self.args.model.input_channel = example['input'].shape[-1]
        self.args.model.output_size = example['output'].shape[-1]
        
        if 'hypernet_input' in example:
            self.args.model.hypernet_cfg.input_size = example['hypernet_input'].shape[-1]
            if self.args.model.preference_extractor_cfg.enabled:
                self.args.model.hypernet_cfg.input_size += self.args.model.preference_extractor_cfg.hidden_dim
                self.args.model.preference_extractor_cfg.input_dim = example['preference_input'].shape[-1]
        # else:
        #     self.args.model.hypernet_cfg.input_size = 0

        if self.trainer_args.load_path:
            print(f"Loading model from {self.trainer_args.load_path}")
            model, normalization_stats, loaded_cfg = load_model(self.trainer_args.load_path)
            self.dataset.normalization_stats = normalization_stats
            if loaded_cfg['model']['output_size'] != self.args.model.output_size:
                # self.switch_last_layer()
                # loaded_cfg['model']['output_size'] = self.args.model.output_size
                Warning("Model output shape doen't match dataset output shape")
                # for key in loaded_cfg['model']:
            hypernet_cfg = deepcopy(self.args.model.hypernet_cfg)
            preference_extractor_cfg = deepcopy(self.args.model.preference_extractor_cfg)
            
            self.args.model = OmegaConf.merge(self.args.model, loaded_cfg['model'])
            # assert all model configurations match
            # assert loaded_cfg['model'] == self.args.model
            self.args.model.hypernet_cfg = hypernet_cfg # switch the hypernet cfg in case of finetuning
            self.args.model.preference_extractor_cfg = preference_extractor_cfg
            self.model = Model(**self.args.model)
            self.model.load_state_dict(model.state_dict(), strict=False)
        else:
            self.model = Model(**self.args.model)
            
        self.model.to(self.device)

        if self.trainer_args.loss_type in ['geort', 'composite', 'alignment']:
            print("Using FK model for loss computation")
            self.fk_model = FKModel(self.args.model.joints_layout).to(self.device)
            self.fk_trainer = FKTrainer(self.fk_model, self.trainer_args.fk_ds_path, self.trainer_args.fk_model_path, self.device)
            if os.path.exists(self.trainer_args.fk_model_path):
                print(f"Loading FK model from {self.trainer_args.fk_model_path}")
                fk_state_dict = torch.load(self.trainer_args.fk_model_path, map_location=self.device)
                self.fk_model.load_state_dict(fk_state_dict)
            else:
                self.fk_trainer.train()
            self.fk_model.eval()
        
        # if self.trainer_args.calib_finetune:
        #     self.set_trainable(main_network=False, calib=True)
        # elif self.trainer_args.calibnet_warmup_epochs == 0:
        #     self.set_trainable(main_network=True, calib=True)
        # elif self.trainer_args.calibnet_warmup_epochs > 0:
        #     self.set_trainable(main_network=True, calib=False)
        self.create_optimizer()

    def set_trainable(self, main_network=True, calib=True):
        """Set which parts of the model are trainable.

        Notes:
        - When calibrating (calib_finetune), we typically freeze the base network weights.
        - If the preference extractor shares the base network as its pose encoder, freezing
          the base network weights is still fine, but we should avoid updating BatchNorm
          running stats / Dropout behavior by keeping the base network in eval() mode.
        """
        for param in self.model.parameters():
            param.requires_grad = main_network

        # if hasattr(self.model, 'calib_block') and self.model.calib_block is not None:
        for param in self.model.calib_block.parameters():
            param.requires_grad = calib
        
        for param in self.model.network.parameters():
            param.requires_grad = main_network

        # Also explicitly enable calibration components that are not in calib_block
        # if calib:
        #     if hasattr(self.model, 'hyperlora') and self.model.hyperlora is not None:
        #         for param in self.model.hyperlora.parameters():
        #             param.requires_grad = True
        #     if hasattr(self.model, 'preference_extractor') and self.model.preference_extractor is not None:
        #         for param in self.model.preference_extractor.parameters():
        #             param.requires_grad = True

    def switch_last_layer(self):
        self.model.fc = nn.Linear(self.model.fc.in_features, self.args.model.output_size)
        self.model.network[-1] = self.model.fc

    def create_optimizer(self):
        # Freeze/unfreeze according to mode
        if self.trainer_args.calib_finetune:
            self.set_trainable(main_network=False, calib=True)
        else:
            self.set_trainable(main_network=True, calib=True)

        # Collect only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found. Check calib_finetune and model wiring."
            )

        # Parse LR scheduler config (backward compatible)
        lr_scheduler_cfg = getattr(self.trainer_args, 'lr_scheduler', None)
        if lr_scheduler_cfg is not None:
            total_steps = getattr(lr_scheduler_cfg, 'total_steps', None)
            warmup_steps = getattr(lr_scheduler_cfg, 'warmup_steps', 0)
            peak_lr = getattr(lr_scheduler_cfg, 'peak_lr', self.trainer_args.learning_rate)
            if peak_lr is None:
                peak_lr = self.trainer_args.learning_rate
            final_lr = getattr(lr_scheduler_cfg, 'final_lr', 0.0)
            start_lr = getattr(lr_scheduler_cfg, 'start_lr', 0.0)
        else:
            total_steps = None
            warmup_steps = 0
            peak_lr = self.trainer_args.learning_rate
            final_lr = 0.0
            start_lr = 0.0

        # Create optimizer with peak_lr as base LR
        self.optimizer = torch.optim.AdamW(trainable_params, lr=peak_lr)
        self.loss_function = nn.MSELoss()

        # Create LR scheduler if enabled
        self.lr_scheduler = None
        if total_steps is not None and total_steps > 0:
            self.lr_scheduler = build_warmup_cosine_scheduler(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak_lr=peak_lr,
                final_lr=final_lr,
                start_lr=start_lr,
            )
            
            print(f"LR Scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}, "
                  f"start_lr={start_lr}, peak_lr={peak_lr}, final_lr={final_lr}")
        else:
            print(f"LR Scheduler: disabled (using constant lr={peak_lr})")

        # Lightweight diagnostic
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in trainable_params)
        print(f"Trainable params: {trainable_num:,} / {total_params:,} ({100.0*trainable_num/max(total_params,1):.2f}%)")
    
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
        
        self.trainloader = DataLoader(self.trainset, batch_size=self.trainer_args.batch_size, shuffle=True, num_workers=0)
        self.valloader = DataLoader(self.valset, batch_size=self.trainer_args.batch_size, shuffle=False, num_workers=0)
        
    def get_loss(self, data, result):
        B = data['input'].shape[0]
        finger_mask = data.get('finger_mask', torch.ones(B, 5, dtype=torch.bool)).to(self.device)
        original_input = data.get('original_data', None)
        if original_input is not None:
            original_input = original_input.float().to(self.device)
        loss_type = self.trainer_args.loss_type

        def get_masked_loss(loss, mask):
            joints_mask = []
            for i, joints in enumerate(self.args.model.joints_layout):
                joint_mask = mask[:, i].unsqueeze(-1).repeat(1, len(joints))
                joints_mask.append(joint_mask)
            joints_mask = torch.cat(joints_mask, dim=-1)
            loss_dict = {'loss': (loss * joints_mask).mean(), 'qpos': loss.mean()}
            if original_input is not None:
                pred = self.model(original_input, side=data['side'])
                drift_error = (pred - result['pred_y']) ** 2
                loss_dict['drift_error'] = (drift_error * joints_mask).mean()
                loss_dict['loss'] += loss_dict['drift_error'] * 10.0
            return loss_dict

        if loss_type == 'qpos':
            loss = get_masked_loss(result['loss'], finger_mask)
        elif loss_type == 'geort':
            loss = get_geort_loss(points=data['input'], side=data['side'], ik_model=self.model, fk_model=self.fk_model, robot_point_cloud=self.fk_trainer.robot_point_cloud)
        elif loss_type == 'alignment':
            loss = get_alignment_loss(points=data['input'], side=data['side'], mask=finger_mask, ik_model=self.model, fk_model=self.fk_model)
            loss = {'loss': loss}
        elif loss_type == 'composite':
            total_loss = 0.0
            loss = {}
            loss['geort'] = get_geort_loss(points=data['input'], side=data['side'], ik_model=self.model, fk_model=self.fk_model, robot_point_cloud=self.fk_trainer.robot_point_cloud)['loss']
            loss['alignment'] = get_alignment_loss(points=data['input'], side=data['side'], mask=finger_mask, ik_model=self.model, fk_model=self.fk_model)
            loss.update(get_masked_loss(result['loss'], finger_mask))

            for key in self.loss_weights:
                total_loss += loss[key] * self.loss_weights[key]
            loss['loss'] = total_loss
        elif loss_type == 'mse':
            loss = {'loss': result['loss'].mean()}
            return loss
        elif loss_type == 'refine_composite':
            loss = get_refine_composite_loss(
                pred_theta=result['pred_y'],
                gt_theta=data['output'].to(self.device),
                mse_loss=result['loss'].mean(),
                side=data['side'].to(self.device),
                loss_weights=self.loss_weights
            )
            return loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if self.prefer_feat_loss is not None:
            loss['loss'] += self.prefer_feat_loss * 0.1  # weight for preference feature loss
            loss['prefer_feat_loss'] = self.prefer_feat_loss

        # Add HyperLoRA regularization loss (only during training)
        if hasattr(self.model, 'hyperlora') and self.model.hyperlora is not None:
            if self.model.training and self.args.model.hypernet_cfg.reg_weight > 0:
                hyperlora_reg = self.model.hyperlora.reg_loss()
                if hyperlora_reg.item() > 0:
                    loss['hyperlora_reg'] = hyperlora_reg
                    loss['loss'] += hyperlora_reg * self.args.model.hypernet_cfg.reg_weight

        return loss

    def _compute_param_norms(self) -> dict:
        """Compute L2 norms of trainable parameters.

        Returns:
            dict: Contains 'train/param_norm' (total) and 'train/param_norm/{module_name}' for key modules.
        """
        param_norms = {}

        # Total parameter norm across all trainable parameters
        param_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                param_norm += (float(p.detach().pow(2).sum().cpu()) / p.numel()) ** 0.5
        # param_norm = param_norm ** 0.5
        param_norm /= len(list(self.model.parameters()))
        param_norms['param_norm'] = param_norm

        # Per-module parameter norms
        module_names = ['network', 'hyperlora']
        for mod_name in module_names:
            if hasattr(self.model, mod_name):
                mod = getattr(self.model, mod_name)
                mod_norm = 0.0
                for p in mod.parameters():
                    if p.requires_grad:
                        mod_norm += (float(p.detach().pow(2).sum().cpu()) / p.numel()) ** 0.5
                # mod_norm = mod_norm ** 0.5
                mod_norm /= len(list(mod.parameters()))
                param_norms[f'param_norm/{mod_name}'] = mod_norm

        return param_norms

    def train_step(self, data, do_backward=True):
        x, y, side = data['input'], data['output'], data['side']
        x, y, side = x.float().to(self.device), y.float().to(self.device), side.to(self.device)

        # Determine hypernet_input based on available data
        hypernet_input = None
        self.prefer_feat_loss = None
        
        if self.args.model.preference_extractor_cfg.enabled:
            # New paired-pose mode: preference extractor produces the full conditioning vector
            if 'preference_pairs' in data:
                # New PairedPoseExtractor path
                preference_pairs = data['preference_pairs'].float().to(self.device)
                preference_mask = data['preference_mask'].to(self.device)
                
                if do_backward:
                    hypernet_input, self.prefer_feat_loss = self.model.preference_extractor.train_forward(
                        preference_pairs, preference_mask
                    )
                else:
                    hypernet_input = self.model.preference_extractor(preference_pairs, preference_mask)
                    
            elif 'preference_input' in data:
                # Legacy TransformerExtractor path
                if do_backward:
                    preference_feat, self.prefer_feat_loss = self.model.preference_extractor.train_forward(
                        data['preference_input'].float().to(self.device),
                        data['preference_mask'].to(self.device)
                    )
                else:
                    preference_feat = self.model.preference_extractor(
                        data['preference_input'].float().to(self.device),
                        data['preference_mask'].to(self.device)
                    )
                
                # Optionally concatenate with hypernet_input if provided
                if 'hypernet_input' in data:
                    base_hypernet = data['hypernet_input'].float().to(self.device)
                    hypernet_input = torch.cat([base_hypernet, preference_feat], dim=-1)
                else:
                    hypernet_input = preference_feat
        elif 'hypernet_input' in data:
            # No preference extractor, use raw hypernet_input
            hypernet_input = data['hypernet_input'].float().to(self.device)

        step_info = {}

        context = self.model.hyperlora.use(hypernet_input) if hypernet_input is not None else nullcontext()
        with context:
            if do_backward: #training
                self.global_steps += 1
                # result = self.model(x, y, hypernet_input, side=side, ik_y=ik_y)
                result = self.model(x, y, side=side)
                loss = self.get_loss(data, result)
                self.optimizer.zero_grad()
                loss['loss'].backward()

                # Gradient norm logging (every 100 steps)
                if self.global_steps % 100 == 0:
                    # Total gradient norm across all trainable parameters
                    total_grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            total_grad_norm += p.grad.detach().pow(2).sum().item()
                    total_grad_norm = total_grad_norm ** 0.5
                    step_info['train/grad_norm'] = total_grad_norm
                    # Parameter norm logging
                    step_info.update(self._compute_param_norms())

                # Gradient diagnostics (lightweight; helps catch zero-grad issues)
                if self.global_steps % 200 == 0:
                    grads = {}
                    for name, p in self.model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            grads[name] = float(p.grad.detach().abs().mean().cpu())
                    # Report a few key gradients if present
                    for key in [
                        'preference_extractor.pool_queries.weight',
                        'preference_extractor.pool_attn.in_proj_weight',
                        'hyperlora.gates.0',
                        'hyperlora.lora_modules.0.fc.weight',
                    ]:
                        if key in grads:
                            print(f"grad|{key}|mean={grads[key]:.3e}")

                nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_args.grad_clipping)
                self.optimizer.step()
                
                # Step LR scheduler if enabled
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Log loss and metrics
                for key, value in loss.items():
                    step_info[f'train/{key}'] = value.item()
                
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                step_info['train/lr'] = current_lr
                
                self.log_step(step_info, self.global_steps)
            else: # evaluating
                # pred_y = self.model(x, hypernet_x=hypernet_input, side=side)
                result = self.model(x, y, side=side)
                mae = torch.mean(torch.abs(y - result['pred_y']))
                loss = self.get_loss(data, result)
                for key, value in loss.items():
                    step_info[f'val/{key}'] = value.item()

        return step_info
    
    def train_epoch(self):
        # If we're in calib_finetune, keep the base network in eval() to prevent
        # BatchNorm running-stat updates and Dropout noise drift.
        self.model.train()
        if self.trainer_args.calib_finetune and hasattr(self.model, 'network'):
            self.model.network.eval()
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
        
        for key in self.loss_weights:
            self.loss_weights[key] += self.trainer_args.loss_weights_decay.get(key, 0.0)
            self.loss_weights[key] = max(self.loss_weights[key], 0.0)
    
        return info

    def validate(self):
        print("Start Validation")
        self.model.eval()
        total_metrics = {}
        with torch.no_grad():
            for data in tqdm(self.valloader):
                step_info = self.train_step(data, do_backward=False)
                for key, value in step_info.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
        for key, value in total_metrics.items():
            total_metrics[key] = value / len(self.valloader)
        self.log_step(total_metrics, self.global_steps)
        print(total_metrics)
        return total_metrics['val/loss']

    def set_epoch(self, epoch):
        
        # Inform dataset of current epoch (for multi-sample bias indexing)
        if hasattr(self.trainset, 'dataset') and hasattr(self.trainset.dataset, 'set_epoch'):
            self.trainset.dataset.set_epoch(epoch)
        elif hasattr(self.trainset, 'set_epoch'):
            self.trainset.set_epoch(epoch)
            
        if hasattr(self.valset, 'dataset') and hasattr(self.valset.dataset, 'set_epoch'):
            self.valset.dataset.set_epoch(epoch)
        elif hasattr(self.valset, 'set_epoch'):
            self.valset.set_epoch(epoch)

    def train(self):
        self.global_steps = 0
        num_epoch = self.trainer_args.num_epochs
        self.loss_weights = self.trainer_args.loss_weights
        best_val_loss = float('inf')
        for i in range(num_epoch):
            print(f"Epoch {i+1}/{num_epoch}")
            self.set_epoch(i)
            # if self.trainer_args.calibnet_warmup_epochs > 0 and i == self.trainer_args.calibnet_warmup_epochs:
            #     self.set_trainable(main_network=True, calib=True)

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
