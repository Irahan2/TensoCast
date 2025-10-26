import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, sensor_idx: int = 0, 
                    horizon_step: int = 0, n_samples: int = 100, title: str = None):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:n_samples, horizon_step, sensor_idx], label='True', alpha=0.7)
    plt.plot(y_pred[:n_samples, horizon_step, sensor_idx], label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(title or f'Predictions for Sensor {sensor_idx}, Horizon {horizon_step+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_decomposition_error(original: np.ndarray, reconstructed: np.ndarray):
    error = np.abs(original - reconstructed)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(original[:100, :50], aspect='auto', cmap='viridis')
    axes[0].set_title('Original')
    axes[0].set_xlabel('Sensors')
    axes[0].set_ylabel('Time')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(reconstructed[:100, :50], aspect='auto', cmap='viridis')
    axes[1].set_title('Reconstructed')
    axes[1].set_xlabel('Sensors')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(error[:100, :50], aspect='auto', cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Sensors')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict: dict, save_path: str = None):
    models = list(metrics_dict.keys())
    metric_keys = ['mae', 'rmse', 'mape', 'smape']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, key in enumerate(metric_keys):
        values = [metrics_dict[m].get(key) or metrics_dict[m].get(key.upper()) or metrics_dict[m].get(key.capitalize()) for m in models]
        axes[idx].bar(models, values)
        axes[idx].set_title(key.upper())
        axes[idx].set_ylabel('Value')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_sensor_heatmap(data: np.ndarray, title: str = 'Sensor Heatmap', n_timesteps: int = 500):
    plt.figure(figsize=(14, 6))
    plt.imshow(data[:n_timesteps, :].T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel('Time')
    plt.ylabel('Sensor ID')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def save_results_table(metrics_dict: dict, save_path: str):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(save_path, float_format='%.4f')
    return df
