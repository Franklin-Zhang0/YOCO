import torch
import torch.nn as nn
from copy import deepcopy
from .trainer import Trainer
from .rewarder import DexRewarder
from calibration_net.utils import mano_recon, transform_joint3d, Side, scale_hand_fingers
import quaternion
from calibration_net.policy.preprocessing import MANO_PCA
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not available. LoRA mode will be disabled.")

class GRPOTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # GRPO specific parameters
        # These can be added to the config.yaml under trainer
        self.args = args
        self.group_size = getattr(self.trainer_args, 'group_size', 8)
        self.eps_clip = getattr(self.trainer_args, 'eps_clip', 0.2)
        self.std_decay = getattr(self.trainer_args, 'std_decay', 1.0)
        self.std = getattr(self.trainer_args, 'std', 0.1)
        self.mano_kl_coeff = getattr(self.trainer_args, 'mano_kl_coeff', 1e-5)
        self.scale_kl_coeff = getattr(self.trainer_args, 'scale_kl_coeff', 1e-5)
        self.mano_comp = getattr(self.trainer_args, 'mano_comp', 10)
        self.rewarder = DexRewarder(self.args.rewarder_config)
        self.mano_pca = {
            'left': MANO_PCA(self.mano_comp, 'left'),
            'right': MANO_PCA(self.mano_comp, 'right'),
        }
        
        # LoRA configuration
        self.lora_config = getattr(self.trainer_args, 'lora', None)
        self.use_lora = self.lora_config and getattr(self.lora_config, 'enabled', False)
        
        # Apply LoRA if enabled
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft library is required for LoRA mode. Install with: pip install peft")
            self._apply_lora()
        
        # Reference model for KL penalty
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Set trainable parameters based on mode
        if self.use_lora:
            # LoRA mode: freeze all base weights, only LoRA adapters are trainable
            # (peft already handles this, but we need to handle finger_scaling separately)
            for param in self.model.parameters():
                param.requires_grad = False
            # Re-enable LoRA parameters (peft sets these as trainable)
            for name, param in self.model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            # Optionally train finger_scaling
            if getattr(self.lora_config, 'train_finger_scaling', True):
                self.model.network.finger_scaling.requires_grad = True
            
            # Recreate optimizer with only trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.trainer_args.learning_rate)
            
            print(f"LoRA mode enabled. Trainable parameters: {sum(p.numel() for p in trainable_params)}")
            print(f"Optimizer updated with {len(trainable_params)} parameter groups")
        else:
            # Standard mode: train mano_regressor, fusion_attention, and finger_scaling
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.network.mano_regressor.parameters():
                param.requires_grad = True
            for param in self.model.network.fusion_attention.parameters():
                param.requires_grad = True
            self.model.network.finger_scaling.requires_grad = True
            
            # Recreate optimizer with updated trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.trainer_args.learning_rate)
    
    def _apply_lora(self):
        """Apply LoRA adapters to the model using peft library"""
        print("Applying LoRA to model...")
        
        # Get target modules from config
        target_modules = getattr(self.lora_config, 'target_modules', None)
        if target_modules is None:
            # Default: target all Linear layers in mano_regressor and fusion_attention
            target_modules = []
        
        # If target_modules is a simple list like ["mano_regressor", "fusion_attention"],
        # we need to expand it to actual Linear layer paths that peft can understand
        expanded_modules = []
        
        # Collect all Linear layer names in the network
        for name, module in self.model.network.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module matches any of the target patterns
                if target_modules:
                    for pattern in target_modules:
                        if pattern in name:
                            expanded_modules.append(name)
                            break
                else:
                    # If no target_modules specified, include all Linear layers
                    expanded_modules.append(name)
        
        if not expanded_modules:
            raise ValueError(
                f"No Linear layers found matching target_modules: {target_modules}. "
                f"Available modules: {[n for n, m in self.model.network.named_modules() if isinstance(m, nn.Linear)]}"
            )
        
        print(f"Found {len(expanded_modules)} Linear layers to apply LoRA:")
        for mod in expanded_modules[:5]:  # Show first 5
            print(f"  - {mod}")
        if len(expanded_modules) > 5:
            print(f"  ... and {len(expanded_modules) - 5} more")
        
        # Create peft LoRA config with expanded module names
        peft_config = LoraConfig(
            r=getattr(self.lora_config, 'rank', 8),
            lora_alpha=getattr(self.lora_config, 'alpha', 16.0),
            lora_dropout=getattr(self.lora_config, 'dropout', 0.1),
            target_modules=expanded_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # Generic task type for custom models
        )
        
        # Apply LoRA to the network (not the whole model, just the base network)
        # We need to wrap only self.model.network
        self.model.network = get_peft_model(self.model.network, peft_config)
        
        print("LoRA applied successfully!")
        self.model.network.print_trainable_parameters()
            
    def get_log_prob(self, mean, action):
        # Gaussian log prob: -0.5 * ((x-mu)/sigma)^2 - log(sigma * sqrt(2*pi))
        # We sum over the action dimensions
        var = self.std ** 2
        # log_prob = -0.5 * ((action - mean) ** 2) / var - torch.log(torch.tensor(self.std * (2 * torch.pi)**0.5, device=mean.device))
        log_prob = -0.5 * ((action - mean) ** 2)/0.5**2  - torch.log(torch.tensor(0.5*(2 * torch.pi)**0.5, device=mean.device))
        return log_prob
    
    def validate(self):
        """Validate the model using reward as the metric."""
        print("Start Validation")
        self.model.eval()
        total_reward = 0.0
        num_batches = 0
        for data in tqdm(self.valloader, desc='Validation'):
            
            with torch.no_grad():
                x, y, side = data['input'], data['output'], data['side']
                x, y, side = x.float().to(self.device), y.float().to(self.device), side.to(self.device)
                
                # Handle hypernet input
                if 'hypernet_input' in data:
                    hypernet_input = data['hypernet_input'].float().to(self.device)
                    if self.args.model.preference_extractor_cfg.enabled:
                        preference_feat, _ = self.model.preference_extractor.train_forward(
                            data['preference_input'].float().to(self.device),
                            data['preference_mask'].to(self.device)
                        )
                        hypernet_input = torch.cat([hypernet_input, preference_feat], dim=-1)
                else:
                    hypernet_input = None
                
                # Get model predictions (mean actions)
                mu = self.model(x, hypernet_x=hypernet_input, side=side)
                mu_scale = self.model.network.finger_scaling
                
                # Reconstruct hand keypoints from MANO parameters
                _, hand_keypoints = mano_recon(mu)
                
                # Apply scaling factors
                hand_keypoints = scale_hand_fingers(hand_keypoints, mu_scale)
                
                # Rotation and Projection
                orient_params = torch.zeros(3, device=self.device)
                orient_quat = quaternion.from_euler_angles(orient_params.cpu().numpy())
                rot_matrix = quaternion.as_rotation_matrix(orient_quat)
                rot_matrix = rot_matrix[None, None]  # [1, 1, 3, 3]
                
                hand_keypoints_np = hand_keypoints.detach().cpu().numpy()
                hand_keypoints_rotated = hand_keypoints_np @ rot_matrix.transpose((0, 1, 3, 2))
                hand_keypoints_rotated = torch.tensor(hand_keypoints_rotated)
                
                # Compute rewards (no group sampling for validation, just evaluate mean)
                # Add batch dimension for compatibility with rewarder
                y_target = y.unsqueeze(1)  # [B, 1, output_dim]
                hand_keypoints_rotated = hand_keypoints_rotated  # [B, 1, 21, 3]
                
            rewards, _, _ = self.rewarder.compute_reward(hand_keypoints_rotated, y_target)
            
            # Average reward for this batch
            batch_reward = rewards.mean().item()
            total_reward += batch_reward
            num_batches += 1
        
        avg_reward = total_reward / num_batches
        self.log_step({'val/reward': avg_reward}, self.global_steps)
        print(f"Validation Reward: {avg_reward}")
        
        return avg_reward

    def train_step(self, data, do_backward=True):
        x, y, side = data['input'], data['output'], data['side']
        x, y, side = x.float().to(self.device), y.float().to(self.device), side.to(self.device)
        
        # Hypernet handling
        if 'hypernet_input' in data:
            hypernet_input = data['hypernet_input'].float().to(self.device)
            if self.args.model.preference_extractor_cfg.enabled:
                preference_feat, _ = self.model.preference_extractor.train_forward(
                    data['preference_input'].float().to(self.device),
                    data['preference_mask'].to(self.device)
                )
                hypernet_input = torch.cat([hypernet_input, preference_feat], dim=-1)
        else:
            hypernet_input = None

        batch_size = x.shape[0]
        
        # Define a separate std for scaling if needed, otherwise use self.std
        # Scaling factors are sensitive, so we often want smaller noise (e.g. 0.01)
        scale_std = getattr(self, 'scale_std', 0.5)

        # 1. Sample Actions (MANO + Scaling)
        with torch.no_grad():
            self.model.eval()
            
            # --- A. MANO Parameters ---
            mu_old = self.model(x, hypernet_x=hypernet_input, side=side)
            
            # Sample G actions: [batch_size, group_size, mano_comp]
            rand_main_comp_act = torch.randn(batch_size, self.group_size, self.mano_comp, device=self.device, dtype=mu_old.dtype) * self.std
            
            # Convert PCA to Full Params
            rand_act = torch.empty(batch_size, self.group_size, mu_old.shape[-1], device=self.device, dtype=mu_old.dtype)
            rand_act[side==Side.LEFT.value] = self.mano_pca['left'].inv_transform(rand_main_comp_act[side==Side.LEFT.value])
            rand_act[side==Side.RIGHT.value] = self.mano_pca['right'].inv_transform(rand_main_comp_act[side==Side.RIGHT.value])
            
            actions = mu_old.unsqueeze(1) + rand_act # [B, G, 48]

            # --- B. Scaling Parameters ---
            # Current mean scaling: [1, 5]
            mu_scale_old = self.model.network.finger_scaling.clone()
            
            # Sample G scaling factors: [batch_size, group_size, 5]
            # We use Gaussian noise here to support PPO log_prob calculation
            scale_noise = torch.randn(batch_size, self.group_size, 5, device=self.device) * scale_std
            # scale_noise = torch.randn(batch_size, self.group_size, 1, device=self.device) * scale_std
            # scale_noise = scale_noise.repeat(1, 1, 5)
            scale_actions = mu_scale_old.unsqueeze(0) + scale_noise
            
            # --- C. Reconstruction & Rewards ---
            # Reconstruct hand keypoints from MANO parameters
            _, hand_keypoints = mano_recon(actions) # [B, G, 21, 3]
            
            # Apply the sampled scaling factors
            # scale_actions is [B, G, 5], needs to broadcast to [B, G, 21, 3]
            # Assuming the scaler handles [B, G, 5] -> [B, G, 21, 3] logic internally 
            # or we broadcast manually:
            # (If finger_scaling is 5 dims, it usually maps to fingers. 
            # Ensure your multiplication logic matches your network's forward pass)
            hand_keypoints = scale_hand_fingers(hand_keypoints, scale_actions)
            # If shapes mismatch, use your specific broadcasting logic here. 
            # Assuming simple broadcast for now based on context.

            # Rotation and Projection
            orient_params = torch.zeros(3, device=self.device)
            orient_quat = quaternion.from_euler_angles(orient_params.cpu().numpy())
            rot_matrix = quaternion.as_rotation_matrix(orient_quat)
            rot_matrix = rot_matrix[None, None]  # [1, 1, 3, 3]
            
            hand_keypoints_np = hand_keypoints.detach().cpu().numpy() 
            hand_keypoints_rotated = hand_keypoints_np @ rot_matrix.transpose((0, 1, 3, 2))
            hand_keypoints_rotated = torch.tensor(hand_keypoints_rotated)
        
        # Compute rewards
        y_target = y.unsqueeze(1).expand(-1, self.group_size, -1)
        rewards, reward_dict, loss_dict = self.rewarder.compute_reward(hand_keypoints_rotated, y_target)
        rewards = rewards.detach()
        
        # 2. Optimization Step
        with torch.no_grad():
            # Advantages
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True)
            advantages = (rewards - mean_r) / (std_r + 1e-8)
            
            # Log Probs OLD
            # MANO log prob
            log_prob_mano_old = self.get_log_prob(mu_old.unsqueeze(1), actions).sum(-1)
            # Scale log prob (using custom std)
            # Custom log prob calculation for scale to allow different std
            var_scale = scale_std ** 2
            log_prob_scale_old = -0.5 * ((scale_actions - mu_scale_old.unsqueeze(0)) ** 2) / var_scale \
                                 - torch.log(torch.tensor(scale_std * (2 * torch.pi)**0.5, device=self.device))
            log_prob_scale_old = log_prob_scale_old.sum(dim=-1) # Sum over 5 scale dims
            
            # Total Old Log Prob
            total_log_prob_old = log_prob_mano_old + log_prob_scale_old

            # Reference Model for KL
            self.ref_model.eval()
            mu_ref = self.ref_model(x, hypernet_x=hypernet_input, side=side)
            mu_scale_ref = self.ref_model.network.finger_scaling.detach()
        
        # Train Forward
        # self.model.train()
        
        # New means
        mu_now = self.model(x, hypernet_x=hypernet_input, side=side)
        mu_scale_now = self.model.network.finger_scaling # This is now the parameter we want to update
        
        # Log Probs NEW
        log_prob_mano_now = self.get_log_prob(mu_now.unsqueeze(1), actions).sum(-1)
        
        log_prob_scale_now = -0.5 * ((scale_actions - mu_scale_now.unsqueeze(0)) ** 2) / var_scale \
                             - torch.log(torch.tensor(scale_std * (2 * torch.pi)**0.5, device=self.device))
        log_prob_scale_now = log_prob_scale_now.sum(dim=-1)
        
        total_log_prob_now = log_prob_mano_now + log_prob_scale_now
        
        # Ratio
        ratio = torch.exp(total_log_prob_now - total_log_prob_old)
        
        # PPO Loss
        surr1 = ratio * advantages # advantages shape check: [B, G] vs ratio [B, G]
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL Penalties
        # 1. MANO KL
        mano_bias_mse = (mu_now - mu_ref)**2
        kl_mano = self.mano_kl_coeff * (mano_bias_mse / (self.std**2 + 1e-5)).sum(dim=-1).mean()
        # 2. Scale KL
        kl_scale = self.scale_kl_coeff * ((mu_scale_now - mu_scale_ref)**2 / (scale_std**2 + 1e-5)).mean()
        
        kl_loss = kl_mano + kl_scale
        
        # Auxiliary Recon Loss (Optional, for regularization)
        _, recon_keypoints = mano_recon(mu_now)
        recon_keypoints = scale_hand_fingers(recon_keypoints, mu_scale_now)
        
        loss = policy_loss + kl_loss # + recon_loss * 0.1 
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_args.grad_clipping)
        self.optimizer.step()
        
        self.global_steps += 1
        self.std *= self.std_decay
        
        step_info = {
            'train/loss': loss.item(),
            'train/kl_mano': kl_mano.item(),
            'train/reward': rewards.mean().item(),
            'train/scale_mean': mu_scale_now.mean().item(),
            'train/scale_grad': mu_scale_now.grad.norm().item() if mu_scale_now.grad is not None else 0,
            'train/mano_bias_mse': mano_bias_mse.mean().item(),
            # Add detailed reward components
            'train/l2_reward': reward_dict['l2_reward'].mean().item(),
            'train/vector_similarity_reward': reward_dict['vector_similarity_reward'].mean().item(),
            'train/fingertip_distance_reward': reward_dict['fingertip_distance_reward'].mean().item(),
            # Add detailed loss components
            'train/l2_loss': loss_dict['l2_loss'].mean().item(),
            'train/vector_similarity_loss': loss_dict['vector_similarity_loss'].mean().item(),
            'train/fingertip_distance_loss': loss_dict['fingertip_distance_loss'].mean().item(),
        }
        self.log_step(step_info, self.global_steps)
        print(f"Scale: {mu_scale_now.detach().cpu().numpy()}")
        
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
