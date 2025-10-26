import numpy as np
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = '') -> Dict[str, float]:
    return {
        f'{prefix}mae': mae(y_true, y_pred),
        f'{prefix}rmse': rmse(y_true, y_pred),
        f'{prefix}mape': mape(y_true, y_pred),
        f'{prefix}smape': smape(y_true, y_pred)
    }
