import numpy as np
import os
import argparse
from src.data_utils import load_metr_la_data, normalize_data, train_test_split, create_sliding_windows
from src.decomposition import cp_decomposition, reconstruct_from_cp
from src.baselines import HistoricalAverage, NaiveForecast
from src.models.lstm_model import LSTMForecaster
from src.models.xgb_model import XGBForecaster
from src.metrics import calculate_all_metrics
from src.visualization import plot_predictions, plot_metrics_comparison, save_results_table


def run_pipeline(data_path: str = 'data/processed/METR-LA.h5', window_size: int = 12,
                horizon: int = 3, rank: int = 50, use_decomposition: bool = False,
                models: list = None, limit_samples: int = None,
                lstm_hidden: int = 128, lstm_layers: int = 2, lstm_epochs: int = 20,
                lstm_batch: int = 64, lstm_lr: float = 1e-3,
                xgb_estimators: int = 100, xgb_max_depth: int = 6, xgb_lr: float = 0.1):
    
    print("Loading data...")
    data, timestamps, sensor_ids = load_metr_la_data(data_path)
    print(f"Data shape: {data.shape}")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        data, window_size, horizon, use_decomposition, rank, limit_samples
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    results = {}
    
    if models is None:
        models = ['ha', 'naive', 'lstm', 'xgb']

    if 'ha' in models:
        print("\nTraining Historical Average baseline...")
        ha = HistoricalAverage(window_size=window_size)
        ha.fit(X_train, y_train)
        y_pred_ha = ha.predict(X_test, horizon=horizon)
        results['HistoricalAverage'] = calculate_all_metrics(y_test, y_pred_ha)
    
    if 'naive' in models:
        print("Training Naive baseline...")
        naive = NaiveForecast()
        naive.fit(X_train, y_train)
        y_pred_naive = naive.predict(X_test, horizon=horizon)
        results['Naive'] = calculate_all_metrics(y_test, y_pred_naive)
    
    if 'lstm' in models:
        print("\nTraining LSTM model...")
        lstm = LSTMForecaster(input_dim=X_train.shape[2], hidden_dim=lstm_hidden, num_layers=lstm_layers,
                              horizon=horizon, dropout=0.2, learning_rate=lstm_lr)
        lstm.fit(X_train, y_train, X_val, y_val, epochs=lstm_epochs, batch_size=lstm_batch, verbose=True)
        y_pred_lstm = lstm.predict(X_test)
        results['LSTM'] = calculate_all_metrics(y_test, y_pred_lstm)
    
    if 'xgb' in models:
        print("\nTraining XGBoost model...")
        xgb_model = XGBForecaster(horizon=horizon, n_estimators=xgb_estimators, max_depth=xgb_max_depth, learning_rate=xgb_lr)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        results['XGBoost'] = calculate_all_metrics(y_test, y_pred_xgb)
    
    print("\nResults:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    os.makedirs('results/figures', exist_ok=True)
    save_results_table(results, 'results/metrics.csv')
    
    fig = plot_metrics_comparison(results, save_path='results/figures/metrics_comparison.png')
    
    if 'lstm' in models:
        fig2 = plot_predictions(y_test, y_pred_lstm, sensor_idx=0, horizon_step=0,
                               n_samples=200, title='LSTM Predictions (Sensor 0, Horizon 1)')
        fig2.savefig('results/figures/lstm_predictions.png', dpi=300, bbox_inches='tight')
    
    print("\nResults saved to results/")
    return results


def prepare_data(data: np.ndarray, window_size: int, horizon: int,
                use_decomposition: bool, rank: int, limit_samples: int = None):

    data_normalized, scaler = normalize_data(data, method='standard')

    if use_decomposition:
        print(f"Applying CP decomposition (rank={rank})...")
        weights, factors, _ = cp_decomposition(data_normalized, rank=rank)
        data_normalized = reconstruct_from_cp(weights, factors)

    X, y = create_sliding_windows(data_normalized, window_size=window_size, horizon=horizon)

    if limit_samples is not None:
        X = X[:limit_samples]
        y = y[:limit_samples]

    X_train, X_val, X_test = train_test_split(X, train_ratio=0.7, val_ratio=0.1)
    y_train, y_val, y_test = train_test_split(y, train_ratio=0.7, val_ratio=0.1)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensoCast forecasting pipeline')
    parser.add_argument('--data-path', type=str, default='data/processed/METR-LA.h5')
    parser.add_argument('--window-size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--use-decomposition', action='store_true')
    parser.add_argument('--rank', type=int, default=50)
    parser.add_argument('--models', type=str, default='ha,naive,lstm,xgb', help='Comma-separated: ha,naive,lstm,xgb')
    parser.add_argument('--limit-samples', type=int, default=None)
    parser.add_argument('--quick', action='store_true', help='Fast smoke test: fewer samples/epochs')

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]

    if args.quick:
        # Quick settings: tiny, fast run
        limit_samples = args.limit_samples or 1000
        lstm_epochs = 1
        xgb_estimators = 50
    else:
        limit_samples = args.limit_samples
        lstm_epochs = 20
        xgb_estimators = 100

    results = run_pipeline(
        data_path=args.data_path,
        window_size=args.window_size,
        horizon=args.horizon,
        rank=args.rank,
        use_decomposition=args.use_decomposition,
        models=models,
        limit_samples=limit_samples,
        lstm_hidden=128,
        lstm_layers=2,
        lstm_epochs=lstm_epochs,
        lstm_batch=64,
        lstm_lr=1e-3,
        xgb_estimators=xgb_estimators,
        xgb_max_depth=6,
        xgb_lr=0.1,
    )
