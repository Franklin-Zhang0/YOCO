import datasets
from torch.utils.data import Dataset
import torch
from calibration_net.utils import walk_dict, index_dict, Side
import numpy as np
from .preprocessing import build_preprocessing
from .augmentation import build_augmentation
from .bias_generator import BiasGenerator, BiasConfig
from typing import Union, Optional
from functools import partial
from tqdm import tqdm
import h5py
import multiprocessing as mp
import random
import warnings

class ConcatHDF5():
    def __init__(self, paths:list[str]):
        self.paths = paths
        self.datasets = [h5py.File(p, 'r') for p in paths]
        self.lengths = [len(d) for d in self.datasets]
        self.cum_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)
        self.seq_names = [list(d.keys()) for d in self.datasets]

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        idx_in_file = idx - self.cum_lengths[file_idx]
        seq_name = self.seq_names[file_idx][idx_in_file]
        return self.datasets[file_idx][seq_name]
    
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
                 augmentations={},
                 preprocessing_x=[],
                 preprocessing_y=[],
                 data_caching=True,
                 cat_side=False,
                 preference_cfg={},
                 is_train=True,
                 bias_cfg: Optional[dict] = None,
                 num_hypernet_pairs: int = 4,
                 pair_subsample: bool = False,
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
        self.augmentations = {k:build_augmentation(v) for k,v in augmentations.items()}
        self.data_caching = data_caching
        self.is_train = is_train
        self.preference_cfg = preference_cfg
        self.num_hypernet_pairs = num_hypernet_pairs
        self.pair_subsample = pair_subsample

        self.input_keys = input_keys
        self.cat_side = cat_side
        self.output_keys = output_keys
        self.hypernet_input_keys = hypernet_input_keys
        self.current_epoch = 0
        
        # Bias handling
        #
        # NOTE: Bias generation (MANO forward) is intentionally NOT done at runtime.
        # If enabled, we only *consume* precomputed columns:
        #   - f"{input_keys}_true"
        #   - f"{input_keys}_biased"
        #   - "mano_angles_biased" (optional, kept for provenance)
        if bias_cfg is not None:
            self.bias_generator = BiasGenerator(BiasConfig(**bias_cfg))
        else:
            self.bias_generator = None
        
        assert self.input_keys in self.data[0].keys(), f"Input keys '{self.input_keys}' not in dataset columns {self.data.column_names}"
        assert self.output_keys in self.data[0].keys(), f"Output keys '{self.output_keys}' not in dataset columns {self.data.column_names}"
            
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
        if self.data_caching:
            self.prefetch_data()

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def prefetch_data(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print("Prefetching data...")
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(self.fetch_demo, i): i for i in range(len(self.data))}
            for future in tqdm(as_completed(futures), total=len(futures)):
                demo_id = futures[future]
                try:
                    _ = future.result()
                except Exception as e:
                    print(f"Error precaching demo {demo_id}: {e}")
        print("Precaching done.")
        
                
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
                    num_dim = np.array(item_shape[1:-1]).prod()
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
        if self.data_caching and demo_id in self.data_cache:
            data = self.data_cache[demo_id]
        else:
            data = {}
            for key, value in walk_dict(self.data[demo_id]):
                data[key] = np.array(value)

            # Base preprocessing
            data[self.input_keys] = self.preprocessing_x(data[self.input_keys])
            data[self.output_keys] = self.preprocessing_y(data[self.output_keys])

            # Bias is consumed from precomputed columns only.
            if self.bias_generator is not None and self.bias_generator.config.enabled:
                true_key = f"{self.input_keys}_true"
                biased_key = f"{self.input_keys}_biased"

                if true_key not in data or biased_key not in data:
                    raise KeyError(
                        "Bias is enabled, but precomputed columns are missing. "
                        f"Expected '{true_key}' and '{biased_key}' in HDF5 demo {demo_id}. "
                        "Run the offline preprocessing via scripts/precompute_bias_hdf5.py first."
                    )

                data[true_key] = self.preprocessing_x(data[true_key])
                data[biased_key] = self.preprocessing_x(data[biased_key])

            if self.data_caching:
                self.data_cache[demo_id] = data
        return data
    
    def __augment__(self, item):
        for key, aug_fn in self.augmentations.items():
            if key in item:
                aug_result = aug_fn({'data': item[key]})
                item[key] = aug_result['data']

                aug_result.pop('data')
                for aug_key, aug_value in aug_result.items():
                    item[aug_key] = aug_value
        return item
            

    def __getitem__(self, idx):
        demo_id = self.mapping['idx_to_demo_id'][idx]
        idx_in_demo = self.mapping['idx_to_idx_in_demo'][idx]
        data = self.fetch_demo(demo_id)
        
        # Check if bias is enabled
        use_bias = self.bias_generator is not None and self.bias_generator.config.enabled
        
        if use_bias:
            # Use biased keypoints as input, true keypoints as ground truth reference
            x_biased = index_dict(dict(data), f'{self.input_keys}_biased')
            x_true = index_dict(dict(data), f'{self.input_keys}_true')
            
            # Check for stacked sampling dimension
            if x_biased.ndim == x_true.ndim + 1:
                num_samples = x_biased.shape[0]
                if self.is_train:
                    sample_idx = self.current_epoch % num_samples
                    x_biased = x_biased[sample_idx]
                else:
                    x_biased = x_biased[0]

            x = x_biased  # Model receives biased input
        else:
            x = index_dict(dict(data), self.input_keys)
            x_true = x  # No bias, true = input
        
        y = index_dict(dict(data), self.output_keys)
        
        inputs = torch.tensor(x[idx_in_demo], dtype=torch.float32)
        outputs = torch.tensor(y[idx_in_demo], dtype=torch.float32).squeeze()

        side = index_dict(dict(data), 'mano_side')[0]
        if isinstance(side, bytes):
            side = side.decode('utf-8')
        assert type(side) == str
        enc_side = Side.__dict__[side.upper()].value

        item = {'input': inputs, 'output': outputs, 'side': enc_side}
        
        # Store original (true) data for drift error computation
        if use_bias:
            item['original_data'] = torch.tensor(x_true[idx_in_demo], dtype=torch.float32)

        # Build HyperLoRA conditioning input from paired biased/true keypoints
        if self.hypernet_input_keys is not None and self.hypernet_input_keys != '':
            # Legacy mode: use specified hypernet_input_keys
            hypernet_x = index_dict(dict(data), self.hypernet_input_keys)
            hypernet_x = torch.tensor(hypernet_x, dtype=torch.float32)
            if self.cat_side:
                side_x = torch.tensor(enc_side, dtype=torch.float32).view(1)
                hypernet_x = torch.cat([hypernet_x, side_x], dim=-1)
            item['hypernet_input'] = hypernet_x
            
            total_length = self.preference_cfg.get('max_tokens', 10)
            rand_length = random.randint(self.preference_cfg.get('min_tokens', 0), self.preference_cfg.get('max_tokens', 10))
            rand_idx = random.choices(range(len(x)), k=rand_length)
            item['preference_x'] =  torch.tensor(x[rand_idx], dtype=torch.float32)
            item['preference_y'] = torch.tensor(y[rand_idx], dtype=torch.float32)
            
            pad_length = total_length - rand_length
            if pad_length > 0:
                item['preference_x'] = torch.cat([item['preference_x'], torch.zeros((pad_length, *item['preference_x'].shape[1:]), dtype=torch.float32)], dim=0)
                item['preference_y'] = torch.cat([item['preference_y'], torch.zeros((pad_length, *item['preference_y'].shape[1:]), dtype=torch.float32)], dim=0)
            item['preference_input'] = torch.cat([item['preference_x'].reshape(total_length, -1), item['preference_y'].reshape(total_length, -1)], dim=-1)
            item['preference_mask'] = torch.cat([torch.ones((rand_length,), dtype=torch.bool), torch.zeros((pad_length,), dtype=torch.bool)], dim=0)
        
        elif use_bias:
            # New mode: build paired-pose input for preference extractor
            # Sample K random frames from the same demo
            seq_len = len(x)
            if self.pair_subsample and self.is_train:
                num_pairs = np.random.normal(self.num_hypernet_pairs/2, self.num_hypernet_pairs/2)
                num_pairs = int(num_pairs)
            else:
                num_pairs = self.num_hypernet_pairs

            num_pairs = np.clip(num_pairs, 1, self.num_hypernet_pairs)
            
            # Sample indices (can include current frame)
            if seq_len > num_pairs:
                sampled_indices = np.random.choice(range(seq_len), num_pairs, replace=False)
            else:
                sampled_indices = np.arange(seq_len)
            
            # Pad to num_hypernet_pairs if needed
            actual_pairs = len(sampled_indices)
            if actual_pairs < self.num_hypernet_pairs:
                # Repeat last sample to pad 
                # sampled_indices += [sampled_indices[-1]] * (self.num_hypernet_pairs - actual_pairs)
                pad_width = self.num_hypernet_pairs - actual_pairs
                sampled_indices = np.pad(sampled_indices, (0, pad_width), mode='edge')
            
            # Get paired samples: (K, 2, 21, 3) where dim=1 is [biased, true]
            biased_samples = x_biased[sampled_indices]  # (K, 21, 3)
            true_samples = x_true[sampled_indices]      # (K, 21, 3)
            
            # Stack as pairs: (K, 2, 21, 3)
            paired_keypoints = np.stack([biased_samples, true_samples], axis=1)
            
            # Convert to tensor
            item['preference_pairs'] = torch.tensor(paired_keypoints, dtype=torch.float32)
            
            # Create mask: True for actual pairs, False for padding
            pair_mask = torch.zeros(self.num_hypernet_pairs, dtype=torch.bool)
            pair_mask[:actual_pairs] = True
            item['preference_mask'] = pair_mask

            # shuffle to prevent long tail problem
            if self.pair_subsample and self.is_train:
                random_order = np.arange(len(pair_mask))
                np.random.shuffle(random_order)
                item['preference_mask'] = item['preference_mask'][random_order]
                item['preference_pairs'] = item['preference_pairs'][random_order]
            
            # Optionally include side information (can be used by preference extractor)
            if self.cat_side:
                item['preference_side'] = torch.tensor(enc_side, dtype=torch.float32)

        if self.is_train:
            item = self.__augment__(item)
            
        return item
    
if __name__ == '__main__':
    # dataset = ConcatHDF5(['/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5','/mnt/sda/yuzhang/data/InterHand/train/hdf5/InterHand_retargeted_30fps_sliced.hdf5'])
    dataset = SequenceDataset(
        path='/data/250010203/franklin/data/hand_keypoint_data/InterHand/train/hdf5/omnihand/InterHand_retargeted_30fps_sliced.hdf5',
        input_keys='world_hand_joints',
        output_keys='mano_angles',
        preprocessing_x=["joint_3d_world_to_local"],
        preprocessing_y=[],
        hypernet_input_keys=None,
        bias_cfg={
            'enabled': True
        },
        data_caching=True
    )
    for data in tqdm(dataset):
        pass