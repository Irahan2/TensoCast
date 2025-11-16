import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 1, horizon: int = 3, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim * horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(last_output)
        return output.view(-1, self.horizon, self.output_dim)


class LSTMForecaster:
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 horizon: int = 3, dropout: float = 0.2, learning_rate: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, input_dim, horizon, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, epochs: int = 50, batch_size: int = 32):
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
