import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from copy import deepcopy
from .trainer import Trainer
from .rewarder import DexRewarder
from calibration_net.utils import mano_recon, transform_joint3d
import quaternion

class GRPOTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # GRPO specific parameters
        # These can be added to the config.yaml under trainer
        self.args = args
        self.group_size = getattr(self.trainer_args, 'group_size', 8)
        self.eps_clip = getattr(self.trainer_args, 'eps_clip', 0.2)
        self.std_decay = getattr(self.trainer_args, 'std_decay', 1.0)
        self.beta = getattr(self.trainer_args, 'beta', 0.01)
        self.std = getattr(self.trainer_args, 'std', 0.1)
        self.rewarder = DexRewarder(self.args.rewarder_config)
        
        # Reference model for KL penalty
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    def get_log_prob(self, mean, action):
        # Gaussian log prob: -0.5 * ((x-mu)/sigma)^2 - log(sigma * sqrt(2*pi))
        # We sum over the action dimensions
        var = self.std ** 2
        log_prob = -0.5 * ((action - mean) ** 2) / var - torch.log(torch.tensor(self.std * (2 * torch.pi)**0.5, device=mean.device))
        return log_prob

    def train_step(self, data, do_backward=True):
        if not do_backward:
            # For validation, just use the base class MSE evaluation
            return super().train_step(data, False)

        x, y, side = data['input'], data['output'], data['side']
        x, y, side = x.float().to(self.device), y.float().to(self.device), side.to(self.device)
        
        # Handle hypernet input if present
        if 'hypernet_input' in data:
            hypernet_input = data['hypernet_input'].float().to(self.device)
            if self.args.model.preference_extractor_cfg.enabled:
                # We need to handle preference extractor if enabled
                preference_feat, _ = self.model.preference_extractor.train_forward(
                    data['preference_input'].float().to(self.device),
                    data['preference_mask'].to(self.device)
                )
                hypernet_input = torch.cat([hypernet_input, preference_feat], dim=-1)
        else:
            hypernet_input = None

        batch_size = x.shape[0]

        # 1. Sample G actions from current policy (old policy for the ratio)
        with torch.no_grad():
            self.model.eval()
            # Get mean from current model
            mu_old = self.model(x, hypernet_x=hypernet_input, side=side)
            
            # Sample G actions: [batch_size, group_size, action_dim]
            actions = mu_old.unsqueeze(1) + torch.randn(batch_size, self.group_size, mu_old.shape[-1], device=self.device) * self.std
            
            
            # Reconstruct hand keypoints from MANO parameters
            _, hand_keypoints = mano_recon(actions)
            
            orient_params=torch.zeros(3, device=self.device)
            
            orient_quat = quaternion.from_euler_angles(orient_params.cpu().numpy())
            rot_matrix = quaternion.as_rotation_matrix(orient_quat)
            rot_matrix = rot_matrix[None, None]  # [batch_size, 3, 3]
            hand_keypoints = hand_keypoints.detach().cpu().numpy() @ rot_matrix.transpose((0, 1, 3, 2))
        
            
            # Compute rewards: -MSE for each sampled action
            y_target = y.unsqueeze(1).expand(-1, self.group_size, -1)
            # MSE loss per sample: [batch_size, group_size]
            # mse_per_sample = F.mse_loss(actions, y_target, reduction='none').mean(dim=-1)
            # rewards = -mse_per_sample
            
        rewards = self.rewarder.compute_reward(hand_keypoints, y_target)
        rewards = rewards.detach()
        
        with torch.no_grad():
            # Compute advantages within each group
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True)
            advantages = (rewards - mean_r) / (std_r + 1e-8) # [batch_size, group_size]
            
            # Compute log_prob under old policy
            log_prob_old = self.get_log_prob(mu_old.unsqueeze(1), actions) # [batch_size, group_size]
            
            # Compute KL with reference model
            self.ref_model.eval()
            mu_ref = self.ref_model(x, hypernet_x=hypernet_input, side=side)
            # KL between two Gaussians with same std: 0.5 * ((mu1 - mu2)/std)^2
        
        # 2. Policy Update
        self.model.train()
        mu_now = self.model(x, hypernet_x=hypernet_input, side=side)
        _, recon_keypoints = mano_recon(mu_now)
        log_prob_now = self.get_log_prob(mu_now.unsqueeze(1), actions)
        
        ratio = torch.exp(log_prob_now - log_prob_old)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        kl = 0.5 * ((mu_now - mu_ref)**2 / (self.std**2)).sum(dim=-1) # [batch_size]
        recon_loss = F.mse_loss(transform_joint3d(recon_keypoints), x, reduction='mean')
        
        kl_loss = kl.mean()
        
        loss = policy_loss + self.beta * kl_loss #+ recon_loss
        # loss = recon_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_args.grad_clipping)
        self.optimizer.step()
        
        self.global_steps += 1
        self.std *= self.std_decay
        
        step_info = {
            'train/loss': loss.item(),
            'train/kl_loss': kl_loss.item(),
            'train/reward': rewards.mean().item(),
            'train/recon_loss': recon_loss.item()
        }
        self.log_step(step_info, self.global_steps)
        
        return step_info

def main():
    from omegaconf import OmegaConf
    from config.grpo_config import GRPOFullConfig
    cli_args = OmegaConf.from_cli()
    if 'config' not in cli_args:
        print("Please provide a config file using config=path/to/config.yaml")
        return
    file_config = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_config = OmegaConf.structured(GRPOFullConfig)
    config = OmegaConf.merge(default_config, file_config, cli_args)
    trainer = GRPOTrainer(args=config)
    trainer.train()

if __name__ == "__main__":
    main()
