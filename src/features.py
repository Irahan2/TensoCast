import numpy as np


def extract_temporal_features(data: np.ndarray, timestamps: np.ndarray = None):
    n_timesteps, n_sensors = data.shape
    features = {}
    
    if timestamps is not None:
        hour = timestamps.hour.values
        day_of_week = timestamps.dayofweek.values
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    features['rolling_mean'] = rolling_window_features(data, window=12, func='mean')
    features['rolling_std'] = rolling_window_features(data, window=12, func='std')
    
    return features


def rolling_window_features(data: np.ndarray, window: int = 12, func: str = 'mean'):
    n_timesteps, n_sensors = data.shape
    result = np.zeros_like(data)
    
    for i in range(window, n_timesteps):
        window_data = data[i-window:i, :]
        if func == 'mean':
            result[i, :] = np.mean(window_data, axis=0)
        elif func == 'std':
            result[i, :] = np.std(window_data, axis=0)
        elif func == 'max':
            result[i, :] = np.max(window_data, axis=0)
        elif func == 'min':
            result[i, :] = np.min(window_data, axis=0)
    
    result[:window, :] = result[window, :]
    return result


def add_lag_features(data: np.ndarray, lags: list = [1, 2, 3, 6, 12, 24]):
    features = []
    for lag in lags:
        lagged = np.roll(data, lag, axis=0)
        lagged[:lag, :] = lagged[lag, :]
        features.append(lagged)
    return np.stack(features, axis=-1)


def extract_spatial_features(data: np.ndarray):
    spatial_mean = np.mean(data, axis=1, keepdims=True)
    spatial_std = np.std(data, axis=1, keepdims=True)
    return {'spatial_mean': spatial_mean, 'spatial_std': spatial_std}
