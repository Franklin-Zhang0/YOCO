import datasets
from torch.utils.data import Dataset
import torch
from calibration_net.utils import walk_dict, index_dict, Side
import numpy as np
from .preprocessing import build_preprocessing
from .augmentation import build_augmentation
from .bias_generator import BiasGenerator, BiasConfig
from .frozen_mano_canonicalizer import build_frozen_mano_canonicalizer, decode_mano_side
from typing import Union, Optional
from tqdm import tqdm
import h5py
import multiprocessing as mp
import random

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
                 reference_poses_path: str = '',
                 frozen_mano_canonicalizer: Optional[dict] = None,
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
        self.augmentations = {k: build_augmentation(v) for k, v in augmentations.items()}
        self.data_caching = data_caching
        self.is_train = is_train
        self.preference_cfg = preference_cfg
        self.num_hypernet_pairs = num_hypernet_pairs
        self.pair_subsample = pair_subsample

        self.input_keys = input_keys
        self.cat_side = cat_side
        self.output_keys = output_keys
        self.hypernet_input_keys = hypernet_input_keys
        self.preprocessing_x_config = preprocessing_x
        self.preprocessing_y_config = preprocessing_y
        self.current_epoch = 0
        self.raw_data_cache: dict[int, dict[str, np.ndarray]] = {}
        self.precanonicalized_input_cache: dict[int, dict[str, np.ndarray]] = {}
        self.has_precanonicalized_inputs = False
        self.frozen_mano_canonicalizer = build_frozen_mano_canonicalizer(
            frozen_mano_canonicalizer
        )
        if self.frozen_mano_canonicalizer is not None:
            print(self.frozen_mano_canonicalizer.status_line)

        if bias_cfg is not None:
            self.bias_generator = BiasGenerator(BiasConfig(**bias_cfg))
        else:
            self.bias_generator = None

        available_keys = list(self.data[0].keys())
        assert self.input_keys in available_keys, (
            f"Input keys '{self.input_keys}' not in dataset columns {available_keys}"
        )
        assert self.output_keys in available_keys, (
            f"Output keys '{self.output_keys}' not in dataset columns {available_keys}"
        )

        self.length, self.mapping = self.create_mapping()

        self.normalization_stats = {'x': None, 'y': None}
        self.preprocessing_x_raw = build_preprocessing(preprocessing_x, 'x')
        self.preprocessing_y_raw = build_preprocessing(preprocessing_y, 'y')
        self.preprocessing_x = self.preprocessing_x_raw
        self.preprocessing_y = self.preprocessing_y_raw

        if self.frozen_mano_canonicalizer is not None:
            self._prefetch_raw_data()
            self._precanonicalize_dataset_inputs()

        self.normalization_stats = self.get_normalization_stats()
        self._rebuild_normalized_preprocessing()
        self.data_cache = {}
        self.data_item_cache = {}

        self.reference_poses_path = reference_poses_path
        self.reference_gt_poses = None
        if reference_poses_path and reference_poses_path != '':
            self._load_reference_poses(reference_poses_path)

        if self.data_caching:
            self.prefetch_data()

    def _rebuild_normalized_preprocessing(self):
        self.preprocessing_x = build_preprocessing(
            self.preprocessing_x_config,
            'x',
            norm_stats=self.normalization_stats,
        )
        self.preprocessing_y = build_preprocessing(
            self.preprocessing_y_config,
            'y',
            norm_stats=self.normalization_stats,
        )

    def set_normalization_stats(self, normalization_stats, refetch: bool = False):
        """Apply external normalization stats and invalidate processed caches."""
        self.normalization_stats = normalization_stats
        self._rebuild_normalized_preprocessing()
        self.data_cache = {}
        self.data_item_cache = {}
        if refetch and self.data_caching:
            self.prefetch_data()

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _decode_side(self, side_value) -> str:
        return decode_mano_side(side_value)

    def _get_item_array(self, item, key: str) -> np.ndarray:
        if self.hdf5_mode:
            return np.array(item[key])
        return np.array(index_dict(dict(item), key))

    def _get_item_side(self, item) -> str:
        return self._decode_side(self._get_item_array(item, 'mano_side'))

    def _load_raw_demo(self, demo_id: int) -> dict[str, np.ndarray]:
        if demo_id in self.raw_data_cache:
            return self.raw_data_cache[demo_id]

        data = {}
        for key, value in walk_dict(self.data[demo_id]):
            data[key] = np.array(value)

        if self.data_caching or self.frozen_mano_canonicalizer is not None:
            self.raw_data_cache[demo_id] = data
        return data

    def _prefetch_raw_data(self):
        print('Prefetching raw data before canonicalization...')
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(self._load_raw_demo, i): i for i in range(len(self.data))}
            for future in tqdm(as_completed(futures), total=len(futures)):
                demo_id = futures[future]
                try:
                    _ = future.result()
                except Exception as e:
                    print(f'Error raw-precaching demo {demo_id}: {e}')
        print('Raw precaching done.')

    def _get_canonicalizable_input_keys(self) -> list[str]:
        keys = [self.input_keys]
        if self.bias_generator is not None and self.bias_generator.config.enabled:
            keys.extend([
                f'{self.input_keys}_true',
                f'{self.input_keys}_biased',
            ])
        return list(dict.fromkeys(keys))

    def _get_precanonicalized_input(self, demo_id: int, key: str) -> np.ndarray | None:
        if demo_id not in self.precanonicalized_input_cache:
            return None
        return self.precanonicalized_input_cache[demo_id].get(key)

    def _flush_precanonicalized_batch(self, side: str, batch_entries: list[dict]):
        if not batch_entries:
            return
        arrays = [entry['array'] for entry in batch_entries]
        canonicalized_arrays = self.frozen_mano_canonicalizer.canonicalize_many(arrays, side=side)
        for entry, canonicalized in zip(batch_entries, canonicalized_arrays):
            demo_cache = self.precanonicalized_input_cache.setdefault(entry['demo_id'], {})
            demo_cache[entry['key']] = canonicalized

    def _precanonicalize_dataset_inputs(self, max_batch_frames: Optional[int] = None):
        if self.frozen_mano_canonicalizer is None:
            return

        print('Pre-canonicalizing input poses in batch...')
        if max_batch_frames is None:
            max_batch_frames = self.frozen_mano_canonicalizer.max_batch_frames

        candidate_keys = self._get_canonicalizable_input_keys()
        buffers = {
            'left': {'entries': [], 'num_frames': 0},
            'right': {'entries': [], 'num_frames': 0},
        }

        def flush(side: str):
            buffer = buffers[side]
            self._flush_precanonicalized_batch(side, buffer['entries'])
            buffer['entries'] = []
            buffer['num_frames'] = 0

        for demo_id in tqdm(range(len(self.data))):
            item = self._load_raw_demo(demo_id)
            side = self._get_item_side(item)
            for key in candidate_keys:
                if key not in item:
                    continue
                arr = self._get_item_array(item, key)
                if arr.shape[-2:] != (21, 3):
                    continue
                buffer = buffers[side]
                buffer['entries'].append({
                    'demo_id': demo_id,
                    'key': key,
                    'array': arr,
                })
                buffer['num_frames'] += int(np.prod(arr.shape[:-2]))
                if buffer['num_frames'] >= max_batch_frames:
                    flush(side)

        for side in buffers:
            flush(side)

        self.has_precanonicalized_inputs = True
        total_arrays = sum(len(cache) for cache in self.precanonicalized_input_cache.values())
        print(
            'Pre-canonicalized '
            f'{total_arrays} input arrays across {len(self.precanonicalized_input_cache)} demos.'
        )

    def _canonicalize_input(
        self,
        joints: np.ndarray,
        side: str,
        demo_id: Optional[int] = None,
        key: Optional[str] = None,
    ) -> np.ndarray:
        if demo_id is not None and key is not None:
            cached = self._get_precanonicalized_input(demo_id, key)
            if cached is not None:
                return cached
        if self.frozen_mano_canonicalizer is None:
            return joints
        return self.frozen_mano_canonicalizer.canonicalize(joints, side=side)

    def _apply_input_preprocessing(
        self,
        joints: np.ndarray,
        side: str,
        normalize: bool = True,
        demo_id: Optional[int] = None,
        key: Optional[str] = None,
    ):
        joints = self._canonicalize_input(joints, side, demo_id=demo_id, key=key)
        preprocess = self.preprocessing_x if normalize else self.preprocessing_x_raw
        return preprocess(joints, normalize=normalize)

    def _apply_output_preprocessing(self, values: np.ndarray, normalize: bool = True):
        preprocess = self.preprocessing_y if normalize else self.preprocessing_y_raw
        return preprocess(values, normalize=normalize)

    def _load_reference_poses(self, path: str):
        """Load reference GT poses from an external HDF5 file for proximity-based sampling."""
        f = h5py.File(path, 'r')
        all_gt = []
        try:
            keys = list(f.keys())
            try:
                keys = sorted(keys, key=lambda x: int(x))
            except Exception:
                keys = sorted(keys)

            for key in keys:
                group = f[key]
                gt = np.array(group['world_hand_joints'])
                side = None
                if 'mano_side' in group:
                    side = self._decode_side(np.array(group['mano_side']))
                elif 'mano_side' in group.attrs:
                    side = self._decode_side(group.attrs['mano_side'])

                if self.frozen_mano_canonicalizer is not None and side is None:
                    raise KeyError(
                        f"Reference poses in {path} are missing mano_side for group {key}."
                    )

                if side is not None:
                    gt = self._apply_input_preprocessing(gt, side, normalize=True)
                else:
                    gt = self.preprocessing_x(gt)
                all_gt.append(gt)
        finally:
            f.close()

        self.reference_gt_poses = np.concatenate(all_gt, axis=0)
        print(f"Loaded {len(self.reference_gt_poses)} reference GT poses from {path}")

    def prefetch_data(self):
        print('Prefetching data...')
        if self.frozen_mano_canonicalizer is not None and not self.has_precanonicalized_inputs:
            print('Using serial prefetch because the frozen MANO canonicalizer is enabled.')
            for demo_id in tqdm(range(len(self.data))):
                self.fetch_demo(demo_id)
            print('Precaching done.')
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(self.fetch_demo, i): i for i in range(len(self.data))}
            for future in tqdm(as_completed(futures), total=len(futures)):
                demo_id = futures[future]
                try:
                    _ = future.result()
                except Exception as e:
                    print(f'Error precaching demo {demo_id}: {e}')
        print('Precaching done.')

        if self.raw_data_cache:
            print('Releasing raw data cache after processed prefetch.')
            self.raw_data_cache.clear()

    def create_mapping(self):
        print('Creating mapping for the dataset...')
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
            print(f'Computing stats for {keys}')
            x = data.select_columns([keys]).rename_column(keys, 'x')
            if transform is not None:
                x = x.map(lambda a: {'x': transform(np.array(a['x']))}, num_proc=16)
            item_shape = np.array(x['x'][0]).shape
            axis_to_sum = tuple(range(len(item_shape) - 1))
            num_dim = np.array(item_shape[1:]).prod()
            x_sum = x.map(lambda a: {'x': np.sum(a['x'], axis=axis_to_sum)}, num_proc=16)
            x_sum = np.sum(x_sum['x'], axis=0)
            x_mean = x_sum / self.length / num_dim
            x_std = x.map(
                lambda a: {'x': np.sum((a['x'] - x_mean) ** 2, axis=axis_to_sum)},
                num_proc=16,
            )
            x_std = np.sqrt(np.sum(x_std['x'], axis=0) / self.length / num_dim)
            return {'mean': x_mean, 'std': x_std}

        def compute_iter_stats(data, keys, is_input: bool):
            print(f'Computing stats for {keys}')
            x_sum = 0
            x_squared_sum = 0
            num_dim = None
            axis_to_sum = None

            for i in tqdm(range(len(data))):
                item = self.raw_data_cache.get(i)
                if item is None:
                    item = self._load_raw_demo(i)
                if is_input:
                    x = self._get_precanonicalized_input(i, keys)
                    if x is None:
                        x = self._get_item_array(item, keys)
                    x = self._apply_input_preprocessing(
                        x,
                        self._get_item_side(item),
                        normalize=False,
                        demo_id=i,
                        key=keys,
                    )
                else:
                    x = self._get_item_array(item, keys)
                    x = self._apply_output_preprocessing(x, normalize=False)
                x = np.asarray(x)
                if num_dim is None:
                    item_shape = x.shape
                    axis_to_sum = tuple(range(len(item_shape) - 1))
                    num_dim = np.array(item_shape[1:-1]).prod()
                    if num_dim == 0:
                        num_dim = 1
                x_sum += np.sum(x, axis=axis_to_sum)

            x_mean = x_sum / self.length / num_dim

            for i in tqdm(range(len(data))):
                item = self.raw_data_cache.get(i)
                if item is None:
                    item = self._load_raw_demo(i)
                if is_input:
                    x = self._get_precanonicalized_input(i, keys)
                    if x is None:
                        x = self._get_item_array(item, keys)
                    x = self._apply_input_preprocessing(
                        x,
                        self._get_item_side(item),
                        normalize=False,
                        demo_id=i,
                        key=keys,
                    )
                else:
                    x = self._get_item_array(item, keys)
                    x = self._apply_output_preprocessing(x, normalize=False)
                x_squared_sum += np.sum((x - x_mean) ** 2, axis=axis_to_sum)

            x_std = np.sqrt(x_squared_sum / self.length / num_dim)
            return {'mean': x_mean, 'std': x_std}

        if not self.hdf5_mode and self.frozen_mano_canonicalizer is None:
            return {
                'x': compute_stats(
                    self.data, self.input_keys, lambda a: self.preprocessing_x_raw(a, normalize=False)
                ),
                'y': compute_stats(
                    self.data, self.output_keys, lambda a: self.preprocessing_y_raw(a, normalize=False)
                ),
            }

        return {
            'x': compute_iter_stats(self.data, self.input_keys, is_input=True),
            'y': compute_iter_stats(self.data, self.output_keys, is_input=False),
        }

    def __chunking__(self, x: Union[np.ndarray, list], chunk_range: slice, idx: slice):
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
            data = dict(self._load_raw_demo(demo_id))

            side = self._decode_side(index_dict(dict(data), 'mano_side'))
            raw_input = np.array(data[self.input_keys])
            raw_output = np.array(data[self.output_keys])
            data[self.input_keys] = self._apply_input_preprocessing(
                raw_input,
                side,
                demo_id=demo_id,
                key=self.input_keys,
            )
            data[self.output_keys] = self._apply_output_preprocessing(raw_output)

            if self.bias_generator is not None and self.bias_generator.config.enabled:
                true_key = f'{self.input_keys}_true'
                biased_key = f'{self.input_keys}_biased'

                if true_key not in data or biased_key not in data:
                    raise KeyError(
                        'Bias is enabled, but precomputed columns are missing. '
                        f"Expected '{true_key}' and '{biased_key}' in HDF5 demo {demo_id}. "
                        'Run the offline preprocessing via scripts/precompute_bias_hdf5.py first.'
                    )

                data[true_key] = self._apply_input_preprocessing(
                    np.array(data[true_key]),
                    side,
                    demo_id=demo_id,
                    key=true_key,
                )
                data[biased_key] = self._apply_input_preprocessing(
                    np.array(data[biased_key]),
                    side,
                    demo_id=demo_id,
                    key=biased_key,
                )

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

        use_bias = self.bias_generator is not None and self.bias_generator.config.enabled

        if use_bias:
            x_biased = index_dict(dict(data), f'{self.input_keys}_biased')
            x_true = index_dict(dict(data), f'{self.input_keys}_true')

            if x_biased.ndim == x_true.ndim + 1:
                num_samples = x_biased.shape[0]
                if self.is_train:
                    sample_idx = (self.current_epoch + idx) % num_samples
                    x_biased = x_biased[sample_idx]
                else:
                    x_biased = x_biased[0]

            x = x_biased
        else:
            x = index_dict(dict(data), self.input_keys)
            x_true = x

        y = index_dict(dict(data), self.output_keys)

        inputs = torch.tensor(x[idx_in_demo], dtype=torch.float32)
        outputs = torch.tensor(y[idx_in_demo], dtype=torch.float32).squeeze()

        side = self._decode_side(index_dict(dict(data), 'mano_side'))
        enc_side = Side.__dict__[side.upper()].value
        betas = np.asarray(index_dict(dict(data), 'betas'), dtype=np.float32)
        if betas.ndim == 1:
            frame_betas = betas
        elif betas.ndim == 2:
            if betas.shape[0] == 1:
                frame_betas = betas[0]
            else:
                frame_betas = betas[idx_in_demo]
        else:
            raise ValueError(f"Unsupported betas shape {betas.shape}")

        item = {
            'input': inputs,
            'output': outputs,
            'side': enc_side,
            'betas': torch.tensor(frame_betas, dtype=torch.float32),
        }

        if use_bias:
            item['original_data'] = torch.tensor(x_true[idx_in_demo], dtype=torch.float32)

        if self.hypernet_input_keys is not None and self.hypernet_input_keys != '':
            hypernet_x = index_dict(dict(data), self.hypernet_input_keys)
            hypernet_x = torch.tensor(hypernet_x, dtype=torch.float32)
            if self.cat_side:
                side_x = torch.tensor(enc_side, dtype=torch.float32).view(1)
                hypernet_x = torch.cat([hypernet_x, side_x], dim=-1)
            item['hypernet_input'] = hypernet_x

            total_length = self.preference_cfg.get('max_tokens', 10)
            rand_length = random.randint(
                self.preference_cfg.get('min_tokens', 0),
                self.preference_cfg.get('max_tokens', 10),
            )
            rand_idx = random.choices(range(len(x)), k=rand_length)
            item['preference_x'] = torch.tensor(x[rand_idx], dtype=torch.float32)
            item['preference_y'] = torch.tensor(y[rand_idx], dtype=torch.float32)

            pad_length = total_length - rand_length
            if pad_length > 0:
                item['preference_x'] = torch.cat([
                    item['preference_x'],
                    torch.zeros((pad_length, *item['preference_x'].shape[1:]), dtype=torch.float32),
                ], dim=0)
                item['preference_y'] = torch.cat([
                    item['preference_y'],
                    torch.zeros((pad_length, *item['preference_y'].shape[1:]), dtype=torch.float32),
                ], dim=0)
            item['preference_input'] = torch.cat([
                item['preference_x'].reshape(total_length, -1),
                item['preference_y'].reshape(total_length, -1),
            ], dim=-1)
            item['preference_mask'] = torch.cat([
                torch.ones((rand_length,), dtype=torch.bool),
                torch.zeros((pad_length,), dtype=torch.bool),
            ], dim=0)

        elif use_bias:
            seq_len = len(x)
            if self.pair_subsample and self.is_train:
                num_pairs = int(np.random.randint(1, self.num_hypernet_pairs + 1))
            else:
                num_pairs = self.num_hypernet_pairs

            num_pairs = np.clip(num_pairs, 1, self.num_hypernet_pairs)

            use_proximity = (
                self.reference_gt_poses is not None
                and self.is_train
                and np.random.random() < 0.5
            )

            if use_proximity:
                ref_poses = self.reference_gt_poses
                demo_poses = x_true

                diff = ref_poses[:, None] - demo_poses[None]
                distances = np.sqrt((diff ** 2).sum(axis=(-2, -1)))

                matched_indices = []
                used_frames = set()
                ref_order = np.argsort(distances.min(axis=1))
                for ref_idx in ref_order:
                    dists = distances[ref_idx].copy()
                    for used in used_frames:
                        dists[used] = np.inf
                    best_frame = np.argmin(dists)
                    matched_indices.append(best_frame)
                    used_frames.add(best_frame)
                    if len(matched_indices) >= seq_len:
                        break

                sampled_indices = np.array(matched_indices)

                if len(sampled_indices) > num_pairs:
                    subsample_idx = np.random.choice(len(sampled_indices), num_pairs, replace=False)
                    sampled_indices = sampled_indices[subsample_idx]
            else:
                if seq_len > num_pairs:
                    sampled_indices = np.random.choice(range(seq_len), num_pairs, replace=False)
                else:
                    sampled_indices = np.arange(seq_len)

            actual_pairs = len(sampled_indices)
            if actual_pairs < self.num_hypernet_pairs:
                pad_width = self.num_hypernet_pairs - actual_pairs
                sampled_indices = np.pad(sampled_indices, (0, pad_width), mode='edge')

            biased_samples = x_biased[sampled_indices]
            true_samples = x_true[sampled_indices]
            paired_keypoints = np.stack([biased_samples, true_samples], axis=1)

            item['preference_pairs'] = torch.tensor(paired_keypoints, dtype=torch.float32)

            pair_mask = torch.zeros(self.num_hypernet_pairs, dtype=torch.bool)
            pair_mask[:actual_pairs] = True
            item['preference_mask'] = pair_mask

            if self.pair_subsample and self.is_train:
                random_order = np.arange(len(pair_mask))
                np.random.shuffle(random_order)
                item['preference_mask'] = item['preference_mask'][random_order]
                item['preference_pairs'] = item['preference_pairs'][random_order]

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
