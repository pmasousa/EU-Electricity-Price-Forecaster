# Swiss Day-Ahead Electricity Price Forecaster

## Overview
A time-series forecasting model designed for the highly volatile European/Swiss energy spot market. The project predicts the next day's hourly electricity prices (EPEX SPOT CH) by analyzing historical price data, localized weather forecasts, grid load estimations, and cross-border energy flows.

## Objectives
- Handle complex, highly volatile time-series data with **probabilistic forecasting**.
- Integrate external APIs for real-time market and weather data.
- Build **explainable** deep learning sequence models tailored for energy trading.
- Implement an **MLOps pipeline** for automated training and deployment.

## Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch (Temporal Fusion Transformers - TFT, LSTMs)
- **Data Sourcing:** ENTSO-E Transparency Platform API, Open-Meteo API
- **Time Series Processing:** Pandas, Darts / PyTorch Forecasting
- **MLOps & Deployment:** MLflow/DVC (Versioning), FastAPI/Gradio (Serving), Docker
- **Evaluation:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Quantile Loss

## Technical Architecture
1. **Data Ingestion & Integration:**
   - Automated scripts fetching hourly load, cross-border flows, and price data from ENTSO-E.
   - Fetching weather forecasts (temperature, wind, solar irradiation) via Open-Meteo.
2. **Feature Engineering:**
   - Cross-border flows (CH imports/exports).
   - Nuclear/hydro availability schedules (crucial for the Swiss grid).
   - Calendar features: Swiss public holidays, weekend effects, heating/cooling degree days.
   - Exogenous regressors: Fuel prices (gas, carbon ETS).
   - Cyclical time encoding (sine/cosine for hours/days).
3. **Forecasting Model:**
   - **TFT Network:** A sequence-to-sequence architecture that takes multivariate data to predict the next 24 hours of prices, emphasizing probabilistic forecasting (prediction intervals) and explainability (attention-based feature importance).
4. **Backtesting Framework:**
   - Walk-forward validation simulating real-world trading P&L on the EPEX SPOT CH bidding zone.
5. **MLOps Pipeline:**
   - Model versioning, scheduled inference/retraining, and predictions served via a REST API.

## Evaluation Baseline

| Model | MAE (CHF/MWh) | RMSE | Training Time |
|-------|---------------|------|---------------|
| Naive persistence | — | — | — |
| ARIMA | — | — | — |
| LSTM | — | — | — |
| TFT | — | — | — |

## Project Roadmap
1. **Phase 0: Environment Setup:** `pyproject.toml` / `requirements.txt`, reproducible env (Docker), `.env` for API keys, project structure scaffolding.
2. **Phase 1: Data Acquisition:** Set up API access and build data downloading scripts (ENTSO-E, Open-Meteo).
3. **Phase 2: EDA & Feature Engineering:** Perform Exploratory Data Analysis (EDA) on price spikes, seasonality, and build domain-specific features.
4. **Phase 3: Data Preparation:** Build data loaders and sequence generators for PyTorch.
5. **Phase 4: Baselines:** Train baseline models (e.g., ARIMA or naive persistence).
6. **Phase 5: Deep Sequence Modeling:** Train and tune the deep sequence model (TFT), analyzing feature importance and probabilistic bounds.
7. **Phase 6: Backtesting:** Implement the backtesting engine and calculate simulated trading P&L.
8. **Phase 7: Deployment & Demo:** Containerize the inference pipeline, host a small dashboard (Streamlit/Gradio) to serve next-day predictions, and write a results summary.
