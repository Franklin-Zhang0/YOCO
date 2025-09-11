import datasets
from torch.utils.data import Dataset
import torch
from calibration_net.utils import walk_dict, index_dict
import numpy as np
from .preprocessing import TRANSFORM_REGISTRY, build_preprocessing
from typing import OrderedDict, Union
from functools import partial
from tqdm import tqdm
import h5py
import multiprocessing as mp

class ConcatHDF5():
    def __init__(self, paths:list[str]):
        self.paths = paths
        self.datasets = [h5py.File(p, 'r') for p in paths]
        self.lengths = [len(d) for d in self.datasets]
        self.cum_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        idx_in_file = idx - self.cum_lengths[file_idx]
        return self.datasets[file_idx][str(idx_in_file)]
    
    def map(self, func, num_proc=1):
        with mp.Pool(num_proc) as pool:
            results = list(tqdm(pool.imap(func, range(len(self))), total=len(self)))
        return results
        

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
                 data_caching=True,
                 cat_side=False,
                 ):
        if ',' in path:
            path = path.split(',')
        else:
            path = [path]
        if 'hdf5' in path[0]:
            self.data = ConcatHDF5(path)
            self.hdf5_mode = True
        else:
            self.data:datasets.Dataset = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in path])
            self.hdf5_mode = False
            
        self.augmentations = augmentations
        self.side = side
        self.data_caching = data_caching

        self.input_keys = input_keys
        self.cat_side = cat_side
        self.output_keys = output_keys
        self.hypernet_input_keys = hypernet_input_keys
        
        assert self.input_keys in self.data[0].keys(), f"Input keys '{self.input_keys}' not in dataset columns {self.data.column_names}"
        assert self.output_keys in self.data[0].keys(), f"Output keys '{self.output_keys}' not in dataset columns {self.data.column_names}"
            
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.length, self.mapping = self.create_mapping()
        
        # preprocessing without normalization to compute stats
        self.normalization_stats = {'x': None, 'y': None}
        self.preprocessing_x = build_preprocessing(preprocessing_x, 'x')
        self.preprocessing_y = build_preprocessing(preprocessing_y, 'y')
        
        self.normalization_stats = self.get_normalization_stats()
        self.preprocessing_x = build_preprocessing(preprocessing_x, 'x', norm_stats=self.normalization_stats)
        self.preprocessing_y = build_preprocessing(preprocessing_y, 'y', norm_stats=self.normalization_stats)
        self.data_cache = {}
        self.data_item_cache = {}
        
                
    def create_mapping(self):
        print("Creating mapping for the dataset...")
        mapping = {}
        
        mapping['idx_to_demo_id'] = []
        mapping['idx_to_idx_in_demo'] = []
        
        total_length = 0
        
        for i in range(len(self.data)):
            seq_len = len(self.data[i][self.input_keys])
            total_length += seq_len
            mapping['idx_to_demo_id'] += [i] * seq_len
            mapping['idx_to_idx_in_demo'] += list(range(seq_len))
            
        return total_length, mapping

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
        
        def compute_hdf5_stats(data, keys, transform=None):
            print(f"Computing stats for {keys}")
            x_sum = 0
            x_squared_sum = 0
            num_dim = None
            for i in tqdm(range(len(data))):
                item = data[i]
                x = np.array(item[keys])
                if transform is not None:
                    x = transform(x)
                if num_dim is None:
                    item_shape = x.shape
                    axis_to_sum = tuple(range(len(item_shape)-1))
                    num_dim = np.array(item_shape[1:]).prod()
                x_sum += np.sum(x, axis=axis_to_sum)
            x_mean = x_sum / self.length / num_dim
            for i in tqdm(range(len(data))):
                item = data[i]
                x = np.array(item[keys])
                if transform is not None:
                    x = transform(x)
                x_squared_sum += np.sum((x - x_mean) ** 2, axis=axis_to_sum)
            x_std = np.sqrt(x_squared_sum / self.length / num_dim)
            return {'mean': x_mean, 'std': x_std}
        
        if not self.hdf5_mode:
            return {
                'x': compute_stats(self.data, self.input_keys, partial(self.preprocessing_x, normalize=False)),
                'y': compute_stats(self.data, self.output_keys, partial(self.preprocessing_y, normalize=False)),
            }
        else:
            return {
                'x': compute_hdf5_stats(self.data, self.input_keys, partial(self.preprocessing_x, normalize=False)),
                'y': compute_hdf5_stats(self.data, self.output_keys, partial(self.preprocessing_y, normalize=False)),
            }
            
        
    def __chunking__(self, x:Union[np.ndarray, list], chunk_range:slice, idx:slice):
        sequence_length = len(x)
        chunk_indices = np.arange(sequence_length)[:, None] + np.array(chunk_range)[None, :]
        chunk_indices = np.clip(chunk_indices, 0, sequence_length - 1)
        if idx is not None:
            return x[chunk_indices[idx]]
        else:
            return x[chunk_indices]

    def __len__(self):
        return self.length

    def fetch_demo(self, demo_id):
        if self.data_caching: 
            if demo_id in self.data_cache:
                data = self.data_cache[demo_id]
            else:
                data = self.data[demo_id]
                self.data_cache[demo_id] = data
        else:
            data = self.data[demo_id]
        return data

    def __getitem__(self, idx):
        demo_id = self.mapping['idx_to_demo_id'][idx]
        idx_in_demo = self.mapping['idx_to_idx_in_demo'][idx]
        # data = self.fetch_demo(demo_id)
        data = self.data[demo_id]
        
        x = np.array(index_dict(dict(data), self.input_keys))
        y = np.array(index_dict(dict(data), self.output_keys))
        
        x = self.preprocessing_x(x)
        y = self.preprocessing_y(y)
        
        x = self.__chunking__(x, range(-self.obs_horizon + 1, 1), idx_in_demo)
        y = self.__chunking__(y, range(0, self.action_horizon), idx_in_demo)
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).squeeze()
        
        # x, y = self.__augment__(x, y)
        item = {'input': x, 'output': y}

        if self.hypernet_input_keys is not None and self.hypernet_input_keys != '':
            hypernet_x = np.array(index_dict(dict(data), self.hypernet_input_keys))
            hypernet_x = torch.tensor(hypernet_x, dtype=torch.float32)
            if self.cat_side:
                side = index_dict(dict(data), 'mano_side')[0]
                if isinstance(side, bytes):
                    side = side.decode('utf-8')
                assert type(side) == str
                side_x = torch.tensor([1.0 if side == 'left' else -1.0])
                hypernet_x = torch.cat([hypernet_x, side_x], dim=-1)
            item['hypernet_input'] = hypernet_x
            
        return item
    
if __name__ == '__main__':
    dataset = ConcatHDF5(['/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5','/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5'])
    dataset = SequenceDataset(
        path='/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5',
        input_keys='pca_mano_pose',
        output_keys='inspire_joint_pose',
        hypernet_input_keys='betas',
        preprocessing_x=[],
        preprocessing_y=[],
        cat_side=True,
    )
    breakpoint()