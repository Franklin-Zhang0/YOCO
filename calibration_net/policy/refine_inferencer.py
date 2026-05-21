from calibration_net.policy.model import load_model
import torch
from calibration_net.policy.preprocessing import build_preprocessing
from calibration_net.policy.frozen_mano_canonicalizer import build_frozen_mano_canonicalizer
from calibration_net.utils import Side
import numpy as np


def resolve_torch_device(
    device: str | torch.device | None = None,
) -> torch.device:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resolved_device = torch.device(device)
    if resolved_device.type == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA requested but not available; falling back to CPU.')
            return torch.device('cpu')
        if (
            resolved_device.index is not None
            and resolved_device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f'Requested CUDA device {resolved_device} is unavailable; '
                f'found {torch.cuda.device_count()} CUDA device(s).'
            )

    return resolved_device


def get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device('cpu')


class RefineModelInferencer():
    def __init__(self, model_dir, side, device=None, paired_poses:dict[str, np.ndarray]=None, fixed_noise=True, loaded_model=None):
        assert side in ['left', 'right']
        resolved_device = resolve_torch_device(device)
        if loaded_model is not None:
            self.model, self.norm_stats, self.cfg = loaded_model
            self.model.to(resolved_device)
        else:
            self.model, self.norm_stats, self.cfg = load_model(
                model_dir,
                device=resolved_device,
            )
        self.device = get_module_device(self.model)
        self.n_comps = self.cfg['model'].get('input_size')
        self.cat_side = self.cfg['dataset'].get('cat_side', False)

        process_x = self.cfg['dataset']['preprocessing_x']
        process_y = []
        for process in self.cfg['dataset']['preprocessing_y']:
            if process == 'normalize':
                process_y.append('denormalize')
            elif process in {'joint_3d_world_to_local', 'flatten'}:
                # These transforms define the supervised target space. Model
                # outputs are already in that space, so reapplying them during
                # inference would corrupt direct keypoint predictions.
                continue
            else:
                process_y.append(process)
        self.process_x_fn = build_preprocessing(process_x, 'x', norm_stats=self.norm_stats)
        self.process_y_fn = build_preprocessing(process_y, 'y', norm_stats=self.norm_stats)

        self.input_canonicalizer = build_frozen_mano_canonicalizer(
            self.cfg.get('dataset', {}).get('frozen_mano_canonicalizer', {}),
            device=self.device,
        )
        if self.input_canonicalizer is not None:
            print(self.input_canonicalizer.status_line)

        self.set_side(side)
        self.update_condition(paired_poses)

        self.fixed_noise = fixed_noise
        self.noise = torch.randn(
            1,
            self.cfg['model']['output_size'],
            device=self.device,
        )

    def set_side(self, side: str):
        assert side in ['left', 'right']
        self.side = side
        self.enc_side = Side.__dict__[side.upper()].value
        self.enc_side_tensor = torch.tensor([self.enc_side], device=self.device)

    def canonicalizer_status(self) -> str | None:
        if self.input_canonicalizer is None:
            return None
        return self.input_canonicalizer.status_line

    def _canonicalize_raw_joints(self, joints):
        if self.input_canonicalizer is None:
            return joints
        return self.input_canonicalizer.canonicalize(joints, side=self.side)

    def update_condition(self, paired_poses:dict[str, np.ndarray]=None):
        hyper_input = None
        paired_poses = paired_poses or {}
        if hasattr(self.model, 'preference_extractor'):
            biased_key = 'world_hand_joints_biased'
            true_key_candidates = ['world_hand_joints', 'world_hand_joints_true']
            true_key = next((key for key in true_key_candidates if key in paired_poses.keys()), None)

            if biased_key in paired_poses and true_key is not None:
                biased_seq = self._canonicalize_raw_joints(paired_poses[biased_key])
                true_seq = self._canonicalize_raw_joints(paired_poses[true_key])
                biased_seq = self.process_x_fn(biased_seq)
                true_seq = self.process_x_fn(true_seq)

                seq_len = biased_seq.shape[0]
                num_pairs = min(self.cfg['dataset'].get('num_hypernet_pairs', 16), seq_len)

                if seq_len >= num_pairs:
                    rng = np.random.default_rng(seed=42)
                    sampled_indices = rng.choice(seq_len, num_pairs, replace=False)
                    actual_pairs = num_pairs
                else:
                    sampled_indices = np.arange(seq_len)
                    sampled_indices = np.pad(sampled_indices, (0, num_pairs - seq_len), mode='edge')
                    actual_pairs = seq_len

                paired_keypoints = np.stack([
                    biased_seq[sampled_indices],
                    true_seq[sampled_indices],
                ], axis=1)

                preference_pairs_tensor = torch.as_tensor(
                    paired_keypoints,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                preference_mask_tensor = torch.zeros(
                    1,
                    num_pairs,
                    dtype=torch.bool,
                    device=self.device,
                )
                preference_mask_tensor[:, :actual_pairs] = True

                with torch.no_grad():
                    pref_mod = self.model.preference_extractor
                    if hasattr(pref_mod, 'train_forward'):
                        preference_feat, _ = pref_mod.train_forward(preference_pairs_tensor, preference_mask_tensor)
                    else:
                        preference_feat = pref_mod(preference_pairs_tensor, preference_mask_tensor)

                hyper_input = preference_feat
            elif paired_poses:
                print('Warning: model has paired preference_extractor but paired_poses are missing required keys; skipping preference conditioning.')
        elif paired_poses:
            Warning('Preference extractor is not enabled in the model but paired_poses are provided.')

        if hasattr(self.model, 'hyperlora'):
            if hyper_input is not None:
                self.model.hyperlora.merge_weight(hyper_input)
            else:
                self.model.hyperlora.unmerge_weight()
        elif paired_poses:
            Warning('Model does not have hyperlora module but paired_poses are provided.')

    def inference(self, x: np.ndarray, batched: bool=False):
        self.model.eval()
        if not self.fixed_noise:
            self.noise = torch.randn(1, self.cfg['model']['output_size'], device=self.device)
        if batched:
            B, T = x.shape[:2]
            noise = self.noise.repeat(B, 1)
        else:
            noise = self.noise
            x = x[None, ...]

        x = self._canonicalize_raw_joints(x)
        x = self.process_x_fn(x)

        x = torch.as_tensor(x, device=self.device).float()
        with torch.no_grad():
            y_pred = self.model.predict(x)
            y_pred = y_pred.squeeze().cpu().numpy()
            y_pred = self.process_y_fn(y_pred)
        return y_pred
