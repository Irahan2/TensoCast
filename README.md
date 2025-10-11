---

# 🧠 TensoCast: Tensor Decomposition for Time Series Forecasting

This project explores a modern approach to **multivariate time series forecasting** (such as traffic flow or energy demand) using **tensor decomposition** techniques.
Instead of flattening data and losing its structure, we aim to keep it in its natural multidimensional form — revealing deeper relationships hidden within the data.

---

### 🎯 Our Goal

Traditional models often simplify time series data by flattening it into 2D matrices, which breaks spatial and temporal relationships.
We want to change that by:

* **Preserving Structure:** Keep data as a multi-way tensor (e.g., `time × sensors × features`).
* **Extracting Hidden Patterns:** Use **CP** and **Tucker decomposition** to uncover latent relationships.
* **Improving Forecasts:** Feed these extracted factors into forecasting models like **LSTM** or **XGBoost**.
* **Evaluating Fairly:** Compare against classical baselines (VAR, GRU, Historical Average) using metrics like **MAE**, **RMSE**, and **MAPE**.

---

### 🧩 Project Plan

1. **Tensorization** — Transform raw time series into tensors using a sliding window.
2. **Decomposition** — Apply CP and Tucker to extract factor matrices and core tensors.
3. **Forecasting** — Use LSTM or XGBoost for the final prediction step.
4. **Comparison** — Evaluate against standard baselines and real-world datasets.

Our first goal is to build the pipeline with **PEMS-BAY** and **METR-LA** datasets.
Once stable, we’ll move on to **Zdunks’ dataset** for real-world validation and extend the model to **Online CP** (real-time updating).

---

### 🧠 Tech Stack

We’re building this project with a mix of classical and deep learning tools:

* **Python** – the main development language
* **TensorLy** – tensor decomposition and algebra
* **PyTorch / Keras** – deep learning for LSTM-based forecasting
* **Scikit-learn** – preprocessing, baselines, and evaluation
* **Pandas & NumPy** – data manipulation and tensor preparation
* **Matplotlib** – visualizations and plots

---

### 🧭 Next Steps

* [ ] Build the tensorization + decomposition pipeline
* [ ] Train and test the LSTM model
* [ ] Add online CP decomposition for streaming data
* [ ] Document and visualize results

---
