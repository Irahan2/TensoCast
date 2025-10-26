import h5py
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def load_metr_la_data(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, 'r') as f:
        data = f['df/block0_values'][()]
        timestamps = f['df/axis1'][()]
        sensor_ids = f['df/axis0'][()]
    return data, timestamps, sensor_ids


def normalize_data(data: np.ndarray, method: str = 'standard',
                   scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    if method == 'none':
        return data, None
    
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data)
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        normalized_data = scaler.transform(data)
    
    return normalized_data, scaler


def train_test_split(data: np.ndarray, train_ratio: float = 0.7,
                     val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_sliding_windows(data: np.ndarray, window_size: int = 12,
                           horizon: int = 3, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(0, len(data) - window_size - horizon + 1, stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon])
    return np.array(X), np.array(y)


def handle_missing_values(data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
    if method == 'interpolate':
        df = pd.DataFrame(data)
        df = df.interpolate(method='linear', axis=0, limit_direction='both')
        return df.fillna(0).values
    elif method == 'zero':
        return np.nan_to_num(data, nan=0.0)
    elif method == 'mean':
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
        return data
    raise ValueError(f"Unknown method: {method}")
