import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


class XGBForecaster:
    def __init__(self, horizon: int = 3, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        n_samples, seq_len, n_sensors = X_train.shape
        X_train_flat = X_train.reshape(n_samples, -1)
        
        self.models = []
        for h in range(self.horizon):
            y_train_h = y_train[:, h, :].reshape(n_samples, -1)
            base = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                n_jobs=-1,
            )
            model = MultiOutputRegressor(base)
            model.fit(X_train_flat, y_train_h)
            self.models.append(model)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_sensors = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        predictions = []
        for model in self.models:
            pred_h = model.predict(X_flat)
            predictions.append(pred_h.reshape(n_samples, n_sensors))
        
        return np.stack(predictions, axis=1)
