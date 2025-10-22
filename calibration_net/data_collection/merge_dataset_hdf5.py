import os, argparse, pickle, numpy as np, h5py
from tqdm import tqdm

def iter_pickles(src):
    for p in os.path.expanduser(src).split(','):
        for fn in sorted(os.listdir(p)):
            if fn.endswith('.pkl'):
                with open(os.path.join(p, fn), 'rb') as f:
                    d = pickle.load(f)
                for k, v in list(d.items()):
                    if isinstance(v, np.ndarray) and v.dtype == np.float64:
                        d[k] = v.astype(np.float32)
                yield d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_dir', required=True)
    ap.add_argument('--save_path', required=True)
    args = ap.parse_args()
    args.save_path = os.path.expanduser(args.save_path)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    with h5py.File(args.save_path, 'w') as h5:
        for i, sample in tqdm(enumerate(iter_pickles(args.source_dir)), total=len(os.listdir(os.path.expanduser(args.source_dir)))):
            g = h5.create_group(str(i))
            for k, v in sample.items():
                g.create_dataset(k, data=v)

if __name__ == '__main__':
    main()
