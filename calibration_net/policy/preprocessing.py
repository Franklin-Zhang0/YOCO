from calibration_net.utils import transform_joint3d
import numpy as np
from typing import OrderedDict
import os
import torch

def get_normalize(norm_stats):
    def normalize(x):
        x = (x - norm_stats['mean']) / (norm_stats['std'] + 1e-6)
        return x
    return normalize


def get_denormalize(norm_stats):
    def denormalize(x):
        x = x * norm_stats['std'] + norm_stats['mean']
        return x
    return denormalize


def bone_normalize(joints_xyz: np.ndarray):
    MCP_indices = [1, 5, 9, 13, 17]
    
    MCP_pos = joints_xyz[..., MCP_indices, :]  # (..., 5, 3)
    root_pos = joints_xyz[..., 0:1, :]  # (..., 1, 3)
    
    bone_vec = MCP_pos - root_pos  # (..., 5, 3)
    # bone_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
    # mean_bone_len = np.mean(bone_len, axis=-2, keepdims=True)  # (..., 1, 3)
    bone_len = np.linalg.norm(bone_vec, axis=-1)  # (..., 5)
    mean_bone_len = np.mean(bone_len, axis=-1)
    normed_joints_xyz = joints_xyz / (mean_bone_len[..., None, None] + 1e-5) * 2.0
    return normed_joints_xyz

def tip_extractor(joints_xyz: np.ndarray):
    tip_indicies = [4, 8, 12, 16, 20]
    return joints_xyz[..., tip_indicies, :]

def get_delta():
    def delta(x):
        x = x[1:] - x[:-1]
        return x
    return delta

def get_original_and_delta():
    def original_and_delta(x):
        delta_x = x[1:] - x[:-1]
        return np.concatenate([x, delta_x], axis=0)
    return original_and_delta

def get_discard_wrist():
    def discard_wrist(x):
        assert x.shape[-2] == 21, f"The input shape should be [..., 21, 3], but got {x.shape}"
        return x[..., 1:, :]
    return discard_wrist

def get_layout_to_5x4():
    def layout_to_5x4(x):
        assert x.shape[-2] == 20, f"The input shape should be [..., 20, 3], but got {x.shape}"
        x_shape = x.shape
        return x.reshape(x_shape[:-2] + (5, 4, 3))
    return layout_to_5x4

def get_flatten():
    def flatten(x):
        return x.reshape(x.shape[0], -1)
    return flatten

class MANO_PCA:
    def __init__(self, num_components=6, side='right'):
        mano_root = os.environ.get('MANO_PATH')
        from manopth.manolayer import ready_arguments
        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

        smpl_data = ready_arguments(self.mano_path)
        
        self.hands_components = smpl_data['hands_components']
        self.num_components = num_components
        
    def transform(self, pose):
        '''
        INPUT:
            pose: (N, 45) np.array
        OUTPUT:
            pca_pose: (N, num_components) np.array
        '''
        if pose.ndim == 1:
            pose = pose[None, :]
        if isinstance(pose, np.ndarray):
            pca_pose = np.linalg.lstsq(self.hands_components[:self.num_components].T, pose.T, rcond=None)[0].T
        elif isinstance(pose, torch.Tensor):
            hands_componets = torch.tensor(self.hands_components[:self.num_components].T, device=pose.device, dtype=pca_pose.dtype)
            pca_pose = torch.linalg.lstsq(hands_componets, pose.T)
        return pca_pose
            
    def inv_transform(self, pca_pose):
        '''
        INPUT:
            pca_pose: (N, num_components) np.array
        OUTPUT:
            pose: (N, 45) np.array
        '''
        if pca_pose.ndim == 1:
            pca_pose = pca_pose[None, :]
        num_samples = pca_pose.shape[0]
        if isinstance(pca_pose, np.ndarray):
            pose = np.zeros((num_samples, 45), dtype=np.float32)
            pose[:,] = np.dot(pca_pose[:,], self.hands_components[:self.num_components])
        elif isinstance(pca_pose, torch.Tensor):
            pose = pca_pose @ torch.tensor(self.hands_components[:self.num_components], device=pca_pose.device, dtype=pca_pose.dtype)
        return pose

mano_pca = None    
def get_mano_pca(mano_angles, num_components=12, side='right'):
    global mano_pca
    if mano_pca is None:
        mano_pca = {
            'left': MANO_PCA(num_components=num_components, side='left'),
            'right': MANO_PCA(num_components=num_components, side='right'),
        }
    return mano_pca[side].transform(mano_angles)


TRANSFORM_REGISTRY = {
    'joint_3d_world_to_local': lambda: transform_joint3d,
    'bone_normalize': lambda: bone_normalize,
    'extract_tip': lambda: tip_extractor,
    'normalize': get_normalize,
    'denormalize': get_denormalize,
    'delta': get_delta,
    'original_and_delta': get_original_and_delta,
    'discard_wrist': get_discard_wrist,
    'joint_layout_to_5x4': get_layout_to_5x4,
    'flatten': get_flatten,
    'mano_pca': lambda: get_mano_pca,
}

def build_preprocessing(preprocess_types, key, norm_stats=None):
    combined_fn = OrderedDict()
    for cur_type in preprocess_types:
        assert cur_type in TRANSFORM_REGISTRY, f"Unknown preprocessing type: {cur_type}"
        if cur_type == 'normalize' or cur_type == 'denormalize':
            if norm_stats is None:
                break
            fn = TRANSFORM_REGISTRY[cur_type](norm_stats[key])
        else:
            fn = TRANSFORM_REGISTRY[cur_type]()
        combined_fn[cur_type] = fn
    if len(combined_fn) == 0:
        return lambda x, normalize=True: x
    else:
        def combined_fn_wrapper(x, normalize=True):
            for fn_type, fn in combined_fn.items():
                if fn_type == 'normalize' and not normalize:
                    continue
                if fn_type == 'denormalize' and normalize:
                    continue
                x = fn(x)
            return x
        return combined_fn_wrapper