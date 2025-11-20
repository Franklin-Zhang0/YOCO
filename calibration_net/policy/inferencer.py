from calibration_net.policy.model import load_model
import torch
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.utils import Side
import numpy as np

class ModelInferencer():
    def __init__(self, model_dir, side, device='cuda' if torch.cuda.is_available() else 'cpu', betas=None, paired_poses:dict[str, np.array]=None, fixed_noise=True):
        assert side in ['left', 'right']
        self.device = device
        self.model, self.norm_stats, self.cfg = load_model(model_dir, device=device)
        self.n_comps = self.cfg['model'].get('input_size')
        self.cat_side = self.cfg['dataset'].get('cat_side', False)

        process_x = self.cfg['dataset']['preprocessing_x']
        process_y = self.cfg['dataset']['preprocessing_y']
        self.process_x_fn = build_preprocessing(process_x, 'x', norm_stats=self.norm_stats)
        self.process_y_fn = build_preprocessing(process_y, 'y', norm_stats=self.norm_stats)
        
        self.betas = torch.tensor(betas, dtype=torch.float32, device = self.device) if betas is not None else torch.zeros((10,), device=device)
        self.enc_side = Side.__dict__[side.upper()].value
        self.enc_side_tensor = torch.tensor([self.enc_side], device=self.device)
        if self.cat_side:
            self.shape = torch.cat([self.betas, self.enc_side_tensor], dim=0)
        else:
            self.shape = self.betas
        hyper_input = self.shape.unsqueeze(0)

        if paired_poses is not None:
            if hasattr(self.model, 'preference_extractor'):
                T = paired_poses['world_hand_joints'].shape[0]
                assert T == paired_poses['robot_joint_pose'].shape[0], "Mismatched time dimension in paired_poses"
                human_hand_joints = self.process_x_fn(paired_poses['world_hand_joints'])
                robot_joint_pose = self.process_y_fn(paired_poses['robot_joint_pose'])
                preference_input = np.concatenate([human_hand_joints.reshape(T, -1), robot_joint_pose.reshape(T, -1)], axis=-1)
                preference_input = torch.tensor(preference_input, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, D]
                attention_mask = torch.ones((1, T), dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    preference_feat = self.model.preference_extractor(preference_input, attention_mask)
                hyper_input = torch.cat([hyper_input, preference_feat], dim=-1)
            else:
                Warning("Preference extractor is not enabled in the model but paired_poses are provided.")
        
        if hasattr(self.model, 'hyperlora'):
            self.model.hyperlora.merge_weight(hyper_input)
        
        self.fixed_noise = fixed_noise
        self.noise = torch.randn(1, self.cfg['model']['output_size']).to(device)
        
    def inference(self, x: np.ndarray, batched: bool=False):
        self.model.eval()
        if not self.fixed_noise:
            self.noise = torch.randn(1, self.cfg['model']['output_size'], device=self.device)
        if batched: 
            B, T = x.shape[:2]
            self.noise = self.noise.repeat(B, 1)
        else:
            x = x[None, ...]

        x = self.process_x_fn(x)

        x = torch.as_tensor(x, device=self.device).float()
        with torch.no_grad():
            y_pred = self.model.predict(x, noisy_act=self.noise, side=self.enc_side_tensor)
            y_pred = y_pred.squeeze().cpu().numpy()
            y_pred = self.process_y_fn(y_pred)
        return y_pred