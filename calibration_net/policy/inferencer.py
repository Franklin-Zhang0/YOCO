from calibration_net.policy.model import load_model
import torch
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.utils import Side

class ModelInferencer():
    def __init__(self, model_dir, side, device='cuda' if torch.cuda.is_available() else 'cpu', betas=None, fixed_noise=True):
        assert side in ['left', 'right']
        self.device = device
        self.model, self.norm_stats, self.cfg = load_model(model_dir, device=device)
        self.n_comps = self.cfg['model'].get('input_size')
        self.cat_side = self.cfg['dataset'].get('cat_side', False)
        
        self.betas = torch.tensor(betas, dtype=torch.float32, device = self.device) if betas is not None else torch.zeros((10,), device=device)
        self.enc_side = Side.__dict__[side.upper()].value
        self.enc_side_tensor = torch.tensor([self.enc_side], device=self.device)
        if self.cat_side:
            self.shape = torch.cat([self.betas, self.enc_side_tensor], dim=0)
        else:
            self.shape = self.betas

        self.hyper_lora_ctx = self.model.hyperlora.use(self.shape.unsqueeze(0))
        process_x = self.cfg['dataset']['preprocessing_x']
        process_y = self.cfg['dataset']['preprocessing_y']
        self.process_x_fn = build_preprocessing(process_x, 'x', norm_stats=self.norm_stats)
        self.process_y_fn = build_preprocessing(process_y, 'y', norm_stats=self.norm_stats)
        self.fixed_noise = fixed_noise
        self.noise = torch.randn(1, self.cfg['model']['output_size']).to(device)
        
    def inference(self, x: torch.Tensor, batched: bool=False):
        self.model.eval()
        if not self.fixed_noise:
            self.noise = torch.randn(1, self.cfg['model']['output_size'], device=self.device)
        if batched: 
            self.noise = self.noise.repeat(x.shape[0], 1)
        else:
            x = x[None, ...]

        x = self.process_x_fn(x)
        assert x.ndim == 2
        
        x = torch.as_tensor(x, device=self.device).float()
        with torch.no_grad():
            with self.hyper_lora_ctx:
                y_pred = self.model.predict(x, noisy_act=self.noise, side=self.enc_side_tensor)
                y_pred = y_pred.squeeze().cpu().numpy()
            y_pred = self.process_y_fn(y_pred)
        return y_pred