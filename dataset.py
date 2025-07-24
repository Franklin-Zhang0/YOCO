import datasets
from torch.utils.data import Dataset
import torch
from utils import walk_dict, index_dict
import numpy as np

class TeleopDataset(Dataset):
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
            x = x * mask
        return x, y

    def __getitem__(self, idx):
        data = self.data[idx]
        flat_data = {k:v for k,v in walk_dict(data)}
        x = torch.tensor(flat_data[self.input_keys])
        y = torch.tensor(flat_data[self.output_keys])
        # x = (x - self.normalization_stats['x']['mean']) / (self.normalization_stats['x']['std'] + 1e-5)
        x = (x - self.normalization_stats['x']['min']) / (self.normalization_stats['x']['max'] - self.normalization_stats['x']['min'] + 1e-5)
        y = (y - self.normalization_stats['y']['mean']) / (self.normalization_stats['y']['std'] + 1e-5)
        
        x, y = self.__augment__(x, y)
        return {'input': x, 'output': y}
    
