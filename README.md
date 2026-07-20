<div align="center">
  <h1>⚡ Swiss Day-Ahead Electricity Price Forecaster</h1>
  <p><i>A probabilistic time-series forecasting model designed for the highly volatile European/Swiss energy spot market.</i></p>
  
  <img src="reports/rolling_forecast_comparison.png" alt="Rolling Day-Ahead Forecast Plot" width="1000"/>
</div>

---

## 📖 Overview

The **Swiss Electricity Price Forecaster** predicts the next day's hourly electricity prices (EPEX SPOT CH) by analyzing historical price data, localized weather forecasts, grid load estimations, and cross-border energy flows.

This project implements an end-to-end Machine Learning pipeline utilizing advanced Deep Sequence Modeling (Temporal Fusion Transformers) to provide both highly accurate and explainable predictions.

## ✨ Features

- **Automated Data Pipelines:** Fetches live data from the ENTSO-E Transparency Platform and Open-Meteo APIs.
- **Deep Sequence Modeling:** Leverages PyTorch Forecasting and `darts` to train a Temporal Fusion Transformer (TFT).
- **Walk-forward Backtesting:** Simulates real-world trading P&L on the EPEX SPOT CH bidding zone.
- **FastAPI Backend:** A robust REST API to serve predictions and handle CORS.
- **Interactive UI Demo:** A Gradio dashboard to visualize predictions in real-time.

## 📈 Model Performance

Evaluating time-series models on a single validation split can be overly optimistic or pessimistic depending on the specific week chosen. Therefore, we evaluate our models using both a standard single-split and a robust **Walk-Forward Backtesting** approach against a strong `NaiveSeasonal (K=24)` baseline (due to the high daily seasonality of electricity prices).

### 1. Single-Split Validation (Week-Ahead Auto-Regression)
This represents the error on a single static 7-day test set. Because this is a Day-Ahead model (trained to predict 24 hours), forecasting a full 7 days (168 hours) in one shot forces the model to use **auto-regression** (feeding its own predictions back into itself to predict further into the future). This naturally degrades accuracy compared to a true 24-hour prediction.
| Model | MAE (CHF/MWh) | RMSE (CHF/MWh) |
|-------|---------------|----------------|
| Linear Regression Baseline | ~42.47 | ~48.57 |
| LightGBM Baseline | ~38.32 | ~45.43 |
| Temporal Fusion Transformer | **~34.75** | **~45.33** |

### 2. Rolling Day-Ahead Backtest (Walk-Forward)
This is a much more robust and realistic metric. We give the model the history up to Day $T$, ask for Day $T+1$, and then slide the window forward by 24 hours. We repeat this for the entire 7-day validation set. This evaluates the model's true **Day-Ahead** performance without the penalty of 7-day auto-regression!
| Model | MAE (CHF/MWh) | RMSE (CHF/MWh) |
|-------|---------------|----------------|
| Temporal Fusion Transformer | **~29.53** | **~40.33** |

*(Note: Exact metrics vary based on the specific historical volatility and the duration of the dataset used).*

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have Python >= 3.11 installed. This project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/swiss_electricity_price_forecaster.git
cd swiss_electricity_price_forecaster

# Install dependencies using uv
uv sync
```

### 2. Configuration

We use the free and open **Energy-Charts API** and **Open-Meteo API** to fetch real electricity and weather data automatically. **No API keys are required!**

However, it is recommended to set up your local environment file:

```bash
cp .env.example .env
```

### 3. Running the Pipeline

You can run the entire machine learning pipeline end-to-end with a single command:

```bash
uv run python run_pipeline.py
```

This unified script will sequentially execute:
1. Data Acquisition (ENTSO-E & Open-Meteo)
2. Feature Engineering
3. Baseline Evaluation
4. Deep Sequence Modeling (TFT)
5. Walk-forward Backtesting
6. Comparison Plot Generation

### 4. Running the Demo Application

Start the backend API and the frontend dashboard in separate terminal windows:

**Terminal 1 (Backend API):**
```bash
uv run python -m src.api.main
```
The API will be available at `http://localhost:8000`.

**Terminal 2 (Gradio UI):**
```bash
uv run python -m src.api.app
```
Open your browser to `http://localhost:7860` to interact with the forecast demo.

## 🛠️ Tech Stack

- **Data Processing:** `pandas`, `numpy`
- **Forecasting:** `darts`, `PyTorch`
- **APIs:** `openmeteo-requests`, `requests`
- **Deployment:** `FastAPI`, `Gradio`, `Docker`

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/swiss_electricity_price_forecaster/issues).
