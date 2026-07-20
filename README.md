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

Evaluating time-series models on a single validation split can be overly optimistic or pessimistic depending on the specific week chosen. Therefore, we evaluate our models using both a standard single-split and a robust **Walk-Forward Backtesting** approach against strong Linear Regression and LightGBM baselines.

### 1. Single-Split Validation (Week-Ahead Auto-Regression)
This represents the error on a single static 7-day test set. Because this is a Day-Ahead model (trained to predict 24 hours), forecasting a full 7 days (168 hours) in one shot forces the model to use **auto-regression** (feeding its own predictions back into itself to predict further into the future). This naturally degrades accuracy compared to a true 24-hour prediction.
| Model | MAE (CHF/MWh) | RMSE (CHF/MWh) |
|-------|---------------|----------------|
| Linear Regression Baseline | ~42.45 | ~48.55 |
| LightGBM Baseline | ~40.16 | ~48.10 |
| Temporal Fusion Transformer | **~36.88** | **~47.71** |

### 2. Rolling Day-Ahead Backtest (Walk-Forward)
This is a much more robust and realistic metric. We give the model the history up to Day $T$, ask for Day $T+1$, and then slide the window forward by 24 hours. We repeat this for the entire 7-day validation set. This evaluates the model's true **Day-Ahead** performance without the penalty of 7-day auto-regression!
| Model | MAE (CHF/MWh) | RMSE (CHF/MWh) |
|-------|---------------|----------------|
| Linear Regression Baseline | ~15.15 | ~17.35 |
| LightGBM Baseline | ~9.33 | ~11.66 |
| Temporal Fusion Transformer | ~24.50 | ~33.19 |

*(Note: Exact metrics vary based on the specific historical volatility and the duration of the dataset used).*

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have Python >= 3.11 installed.

> [!WARNING]
> While you can use tools like `uv` for fast dependency management, it currently struggles to resolve **PyTorch Nightly** builds. If you are using an RTX 50-series GPU (e.g., RTX 5070) which requires PyTorch Nightly for CUDA 12.4 support, you must use standard `pip` as shown below.

```bash
# Clone the repository
git clone https://github.com/yourusername/swiss_electricity_price_forecaster.git
cd swiss_electricity_price_forecaster

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# Install regular dependencies
pip install .

# IF you have an RTX 50-series GPU, install PyTorch Nightly (CUDA 12.4):
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall
```

### 2. Configuration

We use the free and open **Energy-Charts API** and **Open-Meteo API** to fetch real electricity and weather data automatically. **No API keys required**

### 3. Running the Pipeline

You can run the entire machine learning pipeline end-to-end with a single command:

```bash
python run_pipeline.py
```

This unified script will sequentially execute:
1. Data Acquisition (ENTSO-E & Open-Meteo)
2. Feature Engineering
3. Baseline Evaluation
4. Deep Sequence Modeling (TFT)
5. Walk-forward Backtesting
6. Comparison Plot Generation

### 4. Running the Demo Application

You have two options for running the demo: locally via Python or containerized via Docker.

**Important:** Because model weights (`models/`) and datasets (`data/`) are not committed to Git, you **must** complete Step 3 (`python run_pipeline.py`) to generate them before attempting to start the API or UI.

#### Option A: Local Python Environment
Start the backend API and the frontend dashboard in separate terminal windows:

**Terminal 1 (Backend API):**
```bash
python -m src.api.main
```
The API will be available at `http://localhost:8000`.

**Terminal 2 (Gradio UI):**
```bash
python -m src.api.app
```
Open your browser to `http://localhost:7860`.

#### Option B: Docker Compose
If you prefer to run the API and UI in isolated containers, simply run:
```bash
docker-compose up --build
```
This will spin up both the backend and frontend simultaneously. Open your browser to `http://localhost:7860`.

## 🛠️ Tech Stack

- **Data Processing:** `pandas`, `numpy`
- **Forecasting:** `darts`, `PyTorch`
- **APIs:** `openmeteo-requests`, `requests`
- **Deployment:** `FastAPI`, `Gradio`, `Docker`

## Architecture & Implementation Notes

### Deep Learning Architecture
- **Multi-Horizon Forecasting:** To avoid the compounding errors inherent in standard auto-regressive LSTMs (which predict one step iteratively), the Temporal Fusion Transformer (TFT) utilizes a seq-to-seq architecture. It processes a 72-hour lookback window to directly output the entire 24-hour day-ahead curve in a single shot.
- **Probabilistic Forecasting:** Energy markets are prone to extreme price spikes. Instead of minimizing standard Mean Squared Error (which is highly sensitive to outliers), the TFT is configured with `QuantileRegression`. The model outputs a probabilistic distribution (e.g., 10th, 50th, 90th percentiles), quantifying uncertainty rather than forcing a deterministic point forecast.
- **Cyclic Feature Encoding:** Electricity prices exhibit strong daily and weekly seasonality. Raw time features (hour, day of week) were encoded using sine/cosine transformations to ensure the attention mechanism perceives time continuously without artificial zero-hour discontinuities.
- **Strict Covariate Separation:** The data pipeline strictly separates `past_covariates` (historical load/prices) and `future_covariates` (weather, calendar events). Scalers are fitted exclusively on the training split to prevent look-ahead bias and data leakage during the walk-forward backtest.

### MLOps & Deployment
- **Hardware-Agnostic Inference:** The model was trained using PyTorch Lightning with `CUDAAccelerator` (RTX 50-series). To deploy the API on CPU environments, the model's inner `trainer_params` are dynamically intercepted and overridden in memory during hydration (`accelerator='cpu'`).
- **Secure Deserialization:** PyTorch 2.6 defaults to `weights_only=True`, which blocks custom likelihood classes. The `QuantileRegression` distribution was added to `torch.serialization.add_safe_globals` to ensure secure loading in production.
- **Dynamic Backtesting API:** The `/predict` endpoint supports retroactive queries by dynamically slicing the historical time-series dataframe 72 hours prior to the requested timestamp, returning a merged payload of predictions vs. actuals for frontend visualization.
