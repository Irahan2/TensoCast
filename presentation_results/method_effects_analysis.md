# Method Effects Analysis - TensoCast Project

## 1. Tensor Decomposition Effect (CP Decomposition)

### Baseline Performance (No Decomposition)
- Historical Average: MAE = 0.1381
- Naive Forecast: MAE = 0.1349

### With CP Decomposition (Rank=30)
- Historical Average: MAE = 0.1052 (**24% improvement**)
- Naive Forecast: MAE = 0.0882 (**35% improvement**)

**Observation**: Tensor decomposition significantly reduces error rates for traditional forecasting methods. The dimensionality reduction helps capture underlying traffic patterns while filtering noise.

---

## 2. Deep Learning vs Traditional Methods

### Traditional Baselines
- Historical Average: MAE = 0.1381 (simple, reliable)
- Naive Forecast: MAE = 0.1349 (persistence model)

### Machine Learning Methods
- XGBoost: MAE = 0.3498 (ensemble method)
- LSTM: MAE = 0.5401 (neural network, limited training)

**Observation**: Traditional methods outperform ML approaches on this normalized dataset with limited training. Traffic data shows strong temporal patterns that simple averages capture well.

---

## 3. Training Data Size Effect (LSTM)

### Quick Training (1 epoch, 1000 samples)
- LSTM: MAE = 0.5401

### Extended Training (20 epochs, 2000 samples)  
- LSTM: MAE = 0.4738 (**12% improvement**)

**Observation**: More training data and epochs improve LSTM performance, but still not competitive with simple baselines on this dataset. Neural networks may require hyperparameter tuning.

---

## 4. Method Comparison Summary

| Method | MAE | Relative Performance |
|--------|-----|---------------------|
| Naive + CP Decomposition | 0.0882 | **Best** |
| Historical Average + CP | 0.1052 | Very Good |
| Naive (baseline) | 0.1349 | Good |
| Historical Average | 0.1381 | Good |
| XGBoost | 0.3498 | Moderate |
| LSTM (extended) | 0.4738 | Needs tuning |
| LSTM (quick) | 0.5401 | Poor |

---

## Key Insights for Presentation

1. **Tensor decomposition provides the biggest improvement** - 24-35% error reduction
2. **Simple methods work well** for traffic forecasting when data is properly normalized
3. **Machine learning requires more effort** - hyperparameter tuning, feature engineering
4. **Preprocessing matters more than model complexity** for this type of data

## Practical Implications

- For real-time traffic systems: Use Historical Average + CP Decomposition
- For research purposes: Focus on better tensor decomposition ranks and methods
- For production: Simple baselines with good preprocessing often beat complex models
- LSTM potential exists but needs dedicated hyperparameter optimization

## Technical Notes

- All tests used 70/10/20 train/val/test split
- Quick mode: 1000 samples, 1 epoch for LSTM, 50 trees for XGBoost  
- CP decomposition rank=30 chosen empirically
- Data normalized with StandardScaler before processing