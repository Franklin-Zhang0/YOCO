from typing import OrderedDict
import numpy as np

def dropout(p=0.1):
    def dropout_fn(x):
        return np.where(np.random.rand(*x.shape) < p, 0, x)
    return dropout_fn

def gaussian_noise(sigma=0.01):
    def gaussian_noise_fn(x):
        noise = np.random.normal(0, sigma, x.shape)
        return x + noise
    return gaussian_noise_fn

AUGMENTATION_REGISTRY = {
    'dropout': dropout,
    'gaussian_noise': gaussian_noise,
}

def build_augmentation(preprocess_types):
    combined_fn = OrderedDict()
    for cur_type, args in preprocess_types.items():
        assert cur_type in AUGMENTATION_REGISTRY, f"Unknown preprocessing type: {cur_type}"
        fn = AUGMENTATION_REGISTRY[cur_type](**args)
        combined_fn[cur_type] = fn
    if len(combined_fn) == 0:
        return lambda x: x
    else:
        def combined_fn_wrapper(x):
            for fn_type, fn in combined_fn.items():
                x = fn(x)
            return x
        return combined_fn_wrapper