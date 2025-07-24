import datasets
import os
import pickle
import argparse

def pickle_data_reader(path, is_continous=False):
    files = os.listdir(path)
    
    for file_name in files:
        if not file_name.endswith(".pkl"):
            continue
        with open(os.path.join(path, file_name), 'rb') as f:
            data = pickle.load(f)
        
        if is_continous:
            assert isinstance(data, list)
            for item in data:
                yield item
        else:
            yield data
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--is_continous', action='store_true', help="Whether the data is collected in continuous mode")
    args = parser.parse_args()

    data = datasets.Dataset.from_generator(lambda:pickle_data_reader(args.source_dir, args.is_continous))
    data.save_to_disk(args.save_path)
    
if __name__ == "__main__":
    main()