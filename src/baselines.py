import numpy as np


class HistoricalAverage:
    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self
    
    def predict(self, X: np.ndarray, horizon: int = 3) -> np.ndarray:
        n_samples, window_size, n_sensors = X.shape
        predictions = np.zeros((n_samples, horizon, n_sensors))
        for i in range(n_samples):
            mean_val = np.mean(X[i], axis=0)
            predictions[i] = np.tile(mean_val, (horizon, 1))
        return predictions


class NaiveForecast:
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self
    
    def predict(self, X: np.ndarray, horizon: int = 3) -> np.ndarray:
        n_samples = len(X)
        last_value = X[:, -1, :]
        return np.tile(last_value[:, np.newaxis, :], (1, horizon, 1))
