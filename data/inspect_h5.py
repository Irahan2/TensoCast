import h5py
import numpy as np
import os

h5_path = os.path.join(os.path.dirname(__file__), 'processed', 'METR-LA.h5')

def print_h5_structure(name, obj, indent=0):
    prefix = '  ' * indent
    if isinstance(obj, h5py.Dataset):
        print(f'{prefix}- {name} (Dataset)')
        print(f'{prefix}  Shape: {obj.shape}, Dtype: {obj.dtype}')
        try:
            sample = obj[()]
            if isinstance(sample, np.ndarray):
                print(f'{prefix}  Sample values: {sample.flatten()[:10]}')
            else:
                print(f'{prefix}  Sample value: {sample}')
        except Exception as e:
            print(f'{prefix}  Could not read sample: {e}')
    elif isinstance(obj, h5py.Group):
        print(f'{prefix}- {name} (Group)')
        for subkey in obj:
            print_h5_structure(subkey, obj[subkey], indent + 1)

with h5py.File(h5_path, 'r') as f:
    print('HDF5 file structure:')
    for key in f:
        print_h5_structure(key, f[key])
