<div align="center">
  <h1>⚡ Multi-Country Electricity Price Forecaster</h1>
  <p><i>A probabilistic day-ahead electricity price forecasting model for European bidding zones — Switzerland, Portugal, Spain, and more.</i></p>
</div>

---

## 📖 Overview

The **Electricity Price Forecaster** predicts the next day's hourly electricity prices for one or more countries by analyzing historical price data, localized weather, and grid load. It implements an end-to-end ML pipeline using a **Temporal Fusion Transformer (TFT)** for probabilistic, uncertainty-aware forecasts, and can **compare forecasts across countries** side by side.

Supported out of the box (config-driven — add more in `src/config.py`):

| Code | Country      | Bidding zone | Weather station |
|------|--------------|--------------|-----------------|
| CH   | Switzerland  | CH           | Zurich          |
| PT   | Portugal     | PT           | Lisbon          |
| ES   | Spain        | ES           | Madrid          |

## ✨ Features

- **Multi-country:** Train and serve a separate model per country; compare them in one view.
- **Config-driven countries:** Add a bidding zone by adding one entry to `src/config.py`.
- **Automated data pipelines:** Fetches live data from the **Energy-Charts API** (EPEX SPOT / ENTSO-E transparency aggregator) and **Open-Meteo**. **No API keys required.**
- **Deep sequence modeling:** Temporal Fusion Transformer via `darts` + PyTorch, with quantile outputs (10/50/90).
- **Walk-forward backtesting:** Rolling day-ahead evaluation against Linear Regression and LightGBM baselines.
- **FastAPI backend:** REST endpoints for single-country forecasts, cross-country comparison, per-country metrics, and price summaries.
- **Interactive Gradio UI:** Country selector, overlay plots, metrics & summary tables.

## 🔌 Data sources

- **Day-ahead prices & actual load** — [Energy-Charts API](https://api.energy-charts.info) (`/price?bzn=<ZONE>` and `/public_power?country=<cc>`). This aggregates EPEX SPOT / ENTSO-E Transparency Platform data. **No API key.**
- **Weather** — [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive). **No API key.**

> **Note on resolution.** Some bidding zones (e.g. ES, PT) publish day-ahead prices at 15-minute resolution, while others (e.g. CH) are hourly. The download layer resamples all series to a common **hourly** grid so features stay aligned. All prices are in **EUR/MWh** for every zone (Energy-Charts reports CH in EUR/MWh too).

## 🚀 Getting Started

### 1. Environment setup

Ensure you have Python >= 3.11 installed.

> [!WARNING]
> If you use an RTX 50-series GPU requiring PyTorch Nightly (CUDA 12.4+), `uv` may struggle to resolve it — use standard `pip` as shown below.

```bash
# Clone the repository
git clone https://github.com/pmasousa/EU-Electricity-Price-Forecaster.git
cd EU-Electricity-Price-Forecaster

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install .

# (RTX 50-series only) PyTorch Nightly for CUDA 12.4:
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall
```

> **Repo rename:** if you renamed the GitHub repository in Settings, update the clone URL above to match.

### 2. Configuration

Countries live in `src/config.py`. To add a new country, append one entry:

```python
COUNTRIES = {
    ...
    "FR": {"name": "France", "bzn": "FR", "country": "fr",
           "lat": 48.8566, "lon": 2.3522, "tz": "Europe/Paris"},
}
```

The bidding-zone code (`bzn`) and country code (`country`) must match values accepted by the Energy-Charts API. Run a quick check before trusting a new zone:

```bash
curl "https://api.energy-charts.info/price?bzn=FR&start=2026-07-01&end=2026-07-02"
```

### 3. Running the pipeline

Run the full multi-country pipeline (downloads ~3 years of data, builds features, trains baselines + a TFT per country, and generates comparison plots):

```bash
# All countries in src/config.py
python run_pipeline.py

# Or a subset
python run_pipeline.py --countries CH,PT,ES

# Resume from a specific stage
python run_pipeline.py --start-from src/models/train_tft.py --countries PT
```

The pipeline runs, **per country**: data download → feature engineering → baseline evaluation → TFT training → comparison plots. Artifacts are namespaced by country:

- Data: `data/raw/entsoe_prices_{CC}.csv`, `entsoe_load_{CC}.csv`, `weather_{CC}.csv`
- Features: `data/processed/features_{CC}.csv`
- Models: `models/tft_model_{CC}.pt`, `models/scaler_target_{CC}.pkl`, `models/scaler_future_{CC}.pkl`
- Plots: `reports/forecast_comparison_{CC}.png`, `reports/rolling_forecast_comparison_{CC}.png`, ...
- Metrics: `reports/metrics.txt` (per-country sections), `reports/backtest_metrics.txt`

### 4. Running the demo

> Models (`models/`) and datasets (`data/`) are gitignored — run the pipeline (Step 3) first.

#### Option A: Local

```bash
# Terminal 1 — backend API (http://localhost:8000)
python -m src.api.main

# Terminal 2 — Gradio UI (http://localhost:7860)
python -m src.api.app
```

#### Option B: Docker Compose

```bash
docker-compose up --build
```

The UI reads the backend base URL from `API_URL` (defaults to `http://127.0.0.1:8000`; set to `http://api:8000` under Docker).

## 🌐 API reference

All prices are in **EUR/MWh**. The API loads every country model that has artifacts at startup and skips the rest.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check; lists loaded and available countries. |
| `GET /predict?country=PT&target_date=YYYY-MM-DD` | 24h forecast for one country with 10/50/90 quantile bands. `country` defaults to the first loaded country; `target_date` is optional (retroactive comparison). |
| `GET /compare?countries=CH,PT,ES&target_date=YYYY-MM-DD` | Forecasts for multiple countries in one payload, for overlay plots. Countries without a loaded model are reported in `skipped`. |
| `GET /metrics` | Per-country model metrics (MAE/RMSE) parsed from `reports/metrics.txt`. |
| `GET /summary?countries=CH,PT,ES` | Price-level summary per country: mean/median/min/max forecast price and peak hour. |

Response field names (note the EUR currency):

```jsonc
{
  "country": "PT", "country_name": "Portugal",
  "forecast": [
    { "timestamp": "...", "predicted_price_eur_mwh": 62.4, "q10": 50.1, "q90": 74.8,
      "actual_price_eur_mwh": 61.0 }   // actual only present for retroactive queries
  ]
}
```

## 🛠️ Tech stack

- **Data processing:** `pandas`, `numpy`
- **Forecasting:** `darts`, `PyTorch` (Temporal Fusion Transformer, `QuantileRegression`)
- **Weather:** `openmeteo-requests`
- **Serving:** `FastAPI`, `Gradio`, `Docker`

## Architecture & implementation notes

- **Per-country models.** Each country gets its own TFT, scalers, and feature table, isolated by a country suffix. This keeps performance per market independent and makes adding a country a config-only change at the data layer.
- **Multi-horizon forecasting.** The TFT processes a 168-hour (7-day) lookback and directly outputs the 24-hour day-ahead curve in one shot, avoiding auto-regressive error compounding.
- **Probabilistic forecasting.** Configured with `QuantileRegression`, the model outputs a distribution (10/50/90 percentiles) to quantify uncertainty — important in spike-prone markets.
- **Hourly normalization.** Download resamples any resolution to hourly so the per-country feature columns are consistent (6 base columns + calendar + cyclic encodings = 17 features).
- **Strict covariate separation.** `future_covariates` (weather, calendar) are separated from the target, and scalers are fitted only on the training split to prevent leakage during walk-forward backtesting.
- **Hardware-agnostic inference.** `trainer_params` are overridden to `accelerator='cpu'` at load time, so GPU-trained models serve on CPU.
- **Secure deserialization.** `QuantileRegression` and `GlobalTimerCallback` are registered via `torch.serialization.add_safe_globals` for safe unpickling (PyTorch 2.6+ `weights_only` default).

> **Known environment caveat (torch nightly vs darts 0.45).** Loading a previously-saved TFT checkpoint can fail with `PLForecastingModule.__init__() got an unexpected keyword argument 'weights_only'` on very recent PyTorch nightlies. This is a darts/torch version interaction, not a project bug — pin to a compatible torch release (e.g. a stable 2.x) if you hit it. Re-running `python run_pipeline.py --start-from src/models/train_tft.py` retrains against the installed versions.
