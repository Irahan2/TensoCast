# TensoCast

Traffic forecasting using tensor decomposition and machine learning on METR-LA dataset.

## 1) Install

Requires Python 3.9+.

```powershell
pip install -r requirements.txt
```

## 2) Data

Place the dataset at:

- `data/processed/METR-LA.h5`



## 3) Run

Run as module:

Baselines only:

```powershell
python -m src.main --models ha,naive --quick
```

Quick LSTM (1 epoch):

```powershell
python -m src.main --models lstm --quick
```

Quick XGBoost:

```powershell
python -m src.main --models xgb --quick
```

Full run (all models, default settings):

```powershell
python -m src.main --models ha,naive,lstm,xgb
```

Use CP decomposition too (optional, slower):

```powershell
python -m src.main --models ha,naive,lstm --use-decomposition --rank 50
```

## CLI options

- `--data-path` (str): Path to H5 file. Default: `data/processed/METR-LA.h5`
- `--window-size` (int): History window length. Default: 12
- `--horizon` (int): Forecast steps. Default: 3
- `--models` (str): Comma-separated subset of `ha,naive,lstm,xgb`
- `--quick` (flag): Fast run (limits samples, LSTM 1 epoch, fewer XGB trees)
- `--limit-samples` (int): Manually cap samples used before split
- `--use-decomposition` (flag): Apply CP decomposition to normalized data
- `--rank` (int): CP rank if `--use-decomposition` is set



## Outputs

- `results/metrics.csv`: Summary metrics per model (mae, rmse, mape, smape)
- `results/figures/metrics_comparison.png`: Bar charts across models
- `results/figures/lstm_predictions.png`: Example plot (if LSTM selected)

## Project layout

- `src/main.py`: Orchestrates the pipeline and CLI
- `src/data_utils.py`: Load H5, normalize, sliding windows
- `src/decomposition.py`: CP/Tucker helpers (optional)
- `src/baselines.py`: Historical Average, Naive persistence
- `src/models/lstm_model.py`: PyTorch LSTM forecaster
- `src/models/xgb_model.py`: XGBoost forecaster (per-horizon, multi-output)
- `src/metrics.py`: mae, rmse, mape, smape
- `src/visualization.py`: Plots and results table

## Notes

- Run from repo root as module: `python -m src.main`
- Plots saved to `results/figures/`
