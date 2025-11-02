from typing import OrderedDict
import numpy as np

def dropout(p=0.1):
    def dropout_fn(x):
        data = x['data']
        x['data'] = np.where(np.random.rand(*data.shape) < p, 0, data)
        return x
    return dropout_fn

def gaussian_noise(sigma=0.01):
    def gaussian_noise_fn(x):
        data = x['data']
        scale = np.std(data)
        noise = np.random.normal(0, sigma, data.shape) * scale
        x['data'] = data + noise
        return x
    return gaussian_noise_fn

def finger_randomize(p=0.1):
    def finger_randomize_fn(x):
        finger_indices = [
            [1,2,3,4],    # Thumb
            [5,6,7,8],    # Index
            [9,10,11,12], # Middle
            [13,14,15,16],# Ring
            [17,18,19,20] # Pinky
        ]
        data = x['data']
        x['original_data'] = data.copy()
        mask = np.ones(5, dtype=bool)
        for i, finger in enumerate(finger_indices):
            if np.random.rand() < p:
                std, mean = np.std(data[finger]), np.mean(data[finger])
                random_values = np.random.normal(mean, std, data[finger].shape)
                data[finger] = random_values
                mask[i] = False
        x['data'] = data
        x['finger_mask'] = mask
        return x
    return finger_randomize_fn

def history_dropout(p=0.1):
    def history_dropout_fn(x):
        T = x['data'].shape[0]
        mask = np.random.rand(T) < p
        x['data'][mask] = 0
        return x
    return history_dropout_fn


AUGMENTATION_REGISTRY = {
    'dropout': dropout,
    'history_dropout': history_dropout,
    'gaussian_noise': gaussian_noise,
    'finger_randomize': finger_randomize,
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