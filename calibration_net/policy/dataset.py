import datasets
from torch.utils.data import Dataset
import torch
from calibration_net.utils import walk_dict, index_dict
import numpy as np
from .preprocessing import TRANSFORM_REGISTRY
from typing import OrderedDict
from functools import partial

class TeleopDataset(Dataset):  # point-wise dataset
    def __init__(self, path, input_keys, output_keys, side='left', augmentations={}):
        if ',' in path:
            path = path.split(',')
        else:
            path = [path]
        self.data = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in path])
        self.augmentations = augmentations
        self.side = side
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        self.normalization_stats = self.get_normalization_stats()

    def get_normalization_stats(self):
        def compute_stats(data, keys):
            x = data.map(lambda x: {'x': np.array(index_dict(dict(x), keys))})['x']
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            minimum = np.min(x, axis=0)
            maximum = np.max(x, axis=0)
            return {'mean': mean, 'std': std, 'min': minimum, 'max': maximum}
        return {
            'x': compute_stats(self.data, self.input_keys),
            'y': compute_stats(self.data, self.output_keys)
        }

    def __len__(self):
        return len(self.data)
    
    def __augment__(self, x, y):
        if 'x_noise_scale' in self.augmentations:
            x += torch.randn_like(x) * self.augmentations['x_noise_scale']
        if 'y_noise_scale' in self.augmentations:
            y += torch.randn_like(y) * self.augmentations['y_noise_scale']
        if 'x_dropout' in self.augmentations and self.augmentations['x_dropout'] > 0:
            mask = torch.rand_like(x) > self.augmentations['x_dropout']
            x = torch.where(mask, x, torch.zeros_like(x, dtype=x.dtype))
        if 'x_random' in self.augmentations and self.augmentations['x_random'] > 0:
            mask = torch.rand_like(x) > self.augmentations['x_random']
            x = torch.where(mask, x, torch.rand_like(x, dtype=x.dtype))
        return x, y

    def __getitem__(self, idx):
        data = self.data[idx]
        flat_data = {k:v for k,v in walk_dict(data)}
        x = torch.tensor(flat_data[self.input_keys])
        y = torch.tensor(flat_data[self.output_keys])
        x = (x - self.normalization_stats['x']['mean']) / (self.normalization_stats['x']['std'] + 1e-5)
        y = (y - self.normalization_stats['y']['mean']) / (self.normalization_stats['y']['std'] + 1e-5)
        
        x, y = self.__augment__(x, y)
        return {'input': x, 'output': y}
    
class SequenceDataset(Dataset):
    def __init__(self, 
                 path, 
                 input_keys, 
                 hypernet_input_keys,
                 output_keys, 
                 side='left', 
                 augmentations={}, 
                 preprocessing_x=[], 
                 preprocessing_y=[],
                 obs_horizon=1,
                 action_horizon=1,
                 ):
        if ',' in path:
            path = path.split(',')
        else:
            path = [path]

        self.data:datasets.Dataset = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in path])
        self.augmentations = augmentations
        self.side = side

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.hypernet_input_keys = hypernet_input_keys

        assert self.input_keys in self.data.column_names, f"Input keys '{self.input_keys}' not in dataset columns {self.data.column_names}"
        assert self.output_keys in self.data.column_names, f"Output keys '{self.output_keys}' not in dataset columns {self.data.column_names}"

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.length = self.get_length()
        self.mapping = self.create_mapping()
        
        # preprocessing without normalization to compute stats
        self.normalization_stats = {'x': None, 'y': None}
        self.preprocessing_x = self.build_preprocessing(preprocessing_x, 'x')
        self.preprocessing_y = self.build_preprocessing(preprocessing_y, 'y')
        
        self.normalization_stats = self.get_normalization_stats()
        self.preprocessing_x = self.build_preprocessing(preprocessing_x, 'x')
        self.preprocessing_y = self.build_preprocessing(preprocessing_y, 'y')
        
    def build_preprocessing(self, preprocess_types, key):
        combined_fn = OrderedDict()
        for cur_type in preprocess_types:
            assert cur_type in TRANSFORM_REGISTRY, f"Unknown preprocessing type: {cur_type}"
            if cur_type == 'normalize':
                fn = TRANSFORM_REGISTRY[cur_type](self.normalization_stats[key])
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
                    x = fn(x)
                return x
            return combined_fn_wrapper
        
                
    def create_mapping(self):
        print("Creating mapping for the dataset...")
        mapping = {}
        
        mapping['idx_to_demo_id'] = []
        mapping['idx_to_idx_in_demo'] = []
        
        for i in range(len(self.data)):
            seq_len = self.data['seq_len'][i]
            mapping['idx_to_demo_id'] += [i] * seq_len
            mapping['idx_to_idx_in_demo'] += list(range(seq_len))
            
        return mapping

    def get_normalization_stats(self):
        def compute_stats(data, keys, transform=None):
            print(f"Computing stats for {keys}")
            # x = data.map(lambda a: {'x': np.array(index_dict(dict(a), keys))}, num_proc=16)
            x = data.select_columns([keys]).rename_column(keys, 'x')
            if transform is not None:
                x = x.map(lambda a: {'x': transform(np.array(a['x']))}, num_proc=16)
            item_shape = np.array(x['x'][0]).shape
            axis_to_sum = tuple(range(len(item_shape)-1))
            num_dim = np.array(item_shape[1:]).prod()
            x_sum = x.map(lambda a: {'x': np.sum(a['x'], axis=axis_to_sum)}, num_proc=16) # sum a
            x_sum = np.sum(x_sum['x'], axis=0)
            x_mean = x_sum / self.length / num_dim 
            x_std = x.map(lambda a: {'x': np.sum((a['x']- x_mean) ** 2, axis=axis_to_sum)}, num_proc=16)
            x_std = np.sqrt(np.sum(x_std['x'], axis=0) / self.length / num_dim)
            return {'mean': x_mean, 'std': x_std}
        return {
            'x': compute_stats(self.data, self.input_keys, partial(self.preprocessing_x, normalize=False)),
            'y': compute_stats(self.data, self.output_keys, partial(self.preprocessing_y, normalize=False)),
        }
        
    def __chunking__(self, x:np.ndarray, chunk_range:slice):
        seqence_length = x.shape[0]
        chunk_indicies = np.arange(seqence_length)[:, None] + np.array(chunk_range)[None, :]
        chunk_indicies = np.clip(chunk_indicies, 0, seqence_length - 1)
        return x[chunk_indicies]
    
    def get_length(self):
        print("Calculating total length of the dataset...")
        length = self.data.map(lambda x: {'seq_len':len(index_dict(dict(x), self.input_keys))}, num_proc=16)['seq_len']
        self.data = self.data.add_column('seq_len', length)
        total_length = sum(length)
        return total_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        demo_id = self.mapping['idx_to_demo_id'][idx]
        idx_in_demo = self.mapping['idx_to_idx_in_demo'][idx]
        data = self.data[demo_id]
        
        x = np.array(index_dict(dict(data), self.input_keys))
        y = np.array(index_dict(dict(data), self.output_keys))
        
        x = self.preprocessing_x(x)
        y = self.preprocessing_y(y)
        
        x = self.__chunking__(x, range(-self.obs_horizon + 1, 1))
        y = self.__chunking__(y, range(0, self.action_horizon))
        
        x = torch.tensor(x[idx_in_demo], dtype=torch.float32)
        y = torch.tensor(y[idx_in_demo], dtype=torch.float32)
        
        # x, y = self.__augment__(x, y)
        item = {'input': x, 'output': y}

        if self.hypernet_input_keys is not None and self.hypernet_input_keys != '':
            hypernet_x = np.array(index_dict(dict(data), self.hypernet_input_keys))
            hypernet_x = torch.tensor(hypernet_x, dtype=torch.float32)
            item['hypernet_input'] = hypernet_x
        return item
    
if __name__ == '__main__':
    dataset = SequenceDataset(
        path='/mnt/sda/yuzhang/data/dex_ycb_retargeted/val_huggingface/',
        input_keys='mano_pose',
        output_keys='inspire_joint_pose',
        hypernet_input_keys='betas',
        preprocessing_x=[],
        preprocessing_y=['delta'],
    )
    print(dataset[0])