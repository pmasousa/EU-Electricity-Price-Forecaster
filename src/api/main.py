import os
import sys
import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
import uvicorn
import datetime
import pandas as pd
import pickle
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models.torch import QuantileRegression
import torch
from pytorch_lightning.callbacks import Callback
from contextlib import asynccontextmanager

# Allow running as a module: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import COUNTRIES, DEFAULT_COUNTRIES, get_country, parse_countries

# Define dummy callback to allow pickle to load the model. Training scripts save
# the TFT with a GlobalTimerCallback in their callbacks; on load, pickle looks
# the class up by module path. To stay robust whether we're invoked as a script
# (`python -m src.api.main`) or via uvicorn (`__main__` == uvicorn), we register
# the class on __main__ AND in torch's safe globals.
class GlobalTimerCallback(Callback):
    pass


def _register_callback_for_unpickling():
    """Make GlobalTimerCallback findable by pickle/torch regardless of __main__."""
    import __main__
    setattr(__main__, "GlobalTimerCallback", GlobalTimerCallback)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([QuantileRegression, GlobalTimerCallback])


def _load_country_model(country: str):
    """Load one country's TFT model + scalers. Returns None if missing or load fails.

    A failed load is logged and treated as "not available" rather than crashing
    startup, so the API can still serve other countries and the metadata-only
    endpoints (``/``, ``/metrics``). This is important because model deserialization
    is sensitive to the exact torch/pytorch-lightning versions present at runtime.
    """
    model_path = f"models/tft_model_{country}.pt"
    scaler_target_path = f"models/scaler_target_{country}.pkl"
    scaler_future_path = f"models/scaler_future_{country}.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scaler_target_path)
            and os.path.exists(scaler_future_path)):
        return None

    try:
        _register_callback_for_unpickling()
        model = TFTModel.load(model_path, map_location="cpu", weights_only=False)
        if hasattr(model, 'trainer_params'):
            model.trainer_params['accelerator'] = 'cpu'
            model.trainer_params['devices'] = 1

        with open(scaler_target_path, "rb") as f:
            scaler_target = pickle.load(f)
        with open(scaler_future_path, "rb") as f:
            scaler_future = pickle.load(f)

        return {"model": model, "scaler_target": scaler_target, "scaler_future": scaler_future}
    except Exception as e:
        print(f"Warning: failed to load model for {country} ({type(e).__name__}: {e}). "
              f"Skipping — retrain with `python run_pipeline.py --countries {country}` "
              f"against the installed library versions to use it.")
        return None


# Map of country code -> {"model", "scaler_target", "scaler_future"} for loaded countries.
MODELS: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make custom callback + likelihood classes resolvable during unpickling.
    _register_callback_for_unpickling()

    for country in DEFAULT_COUNTRIES:
        loaded = _load_country_model(country)
        if loaded is not None:
            MODELS[country] = loaded
            print(f"Loaded model + scalers for {country}.")
        else:
            print(f"Warning: model artifacts for {country} not found "
                  f"(run the pipeline for this country first). Skipping.")

    if not MODELS:
        print("Warning: No country models loaded. API will fail until the pipeline is run.")
    else:
        print(f"Ready. Loaded models for: {sorted(MODELS)}")
    yield
    # Clean up (if any) goes here


app = FastAPI(title="Electricity Price Forecaster", lifespan=lifespan)

# Handle CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "message": "Electricity Price Forecaster API is running",
        "countries_loaded": sorted(MODELS),
        "countries_available": sorted(COUNTRIES),
    }


def create_cyclic_features(df, col_name, period):
    df[f"{col_name}_sin"] = np.sin(2 * np.pi * df[col_name] / period)
    df[f"{col_name}_cos"] = np.cos(2 * np.pi * df[col_name] / period)
    return df


def _forecast_one(country: str, target_date: Optional[str] = None) -> dict:
    """Core single-country forecast. Returns a payload dict with quantile bands."""
    if country not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"No loaded model for country '{country}'. Loaded: {sorted(MODELS)}",
        )

    bundle = MODELS[country]
    model = bundle["model"]
    scaler_target = bundle["scaler_target"]
    scaler_future = bundle["scaler_future"]

    try:
        # Load the latest data for this country
        features_path = f"data/processed/features_{country}.csv"
        if not os.path.exists(features_path):
            raise HTTPException(
                status_code=404,
                detail=f"Features for country '{country}' not found at {features_path}.",
            )
        df = pd.read_csv(features_path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # History window must cover the model's input_chunk_length (168h = 7 days,
        # set in train_tft.py). Using less raises "series too short" at predict.
        history_len = 168
        forecast_len = 24

        df_actual = pd.DataFrame()

        if target_date:
            try:
                target_start = pd.to_datetime(target_date).normalize()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

            if target_start <= df.index[0] + pd.Timedelta(hours=history_len):
                raise HTTPException(status_code=400, detail="Date too early, not enough historical data.")

            history_end = target_start - pd.Timedelta(hours=1)
            df_history = df.loc[:history_end].iloc[-history_len:].copy()

            target_end = target_start + pd.Timedelta(hours=forecast_len - 1)
            df_actual = df.loc[target_start:target_end].copy()
            actual_prices = (
                {ts.isoformat(): float(v) for ts, v in df_actual['price'].items()}
                if not df_actual.empty else {}
            )
        else:
            df_history = df.iloc[-history_len:].copy()
            actual_prices = {}

        # Create future 24 hours of covariates
        last_date = df_history.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_len, freq='h')

        df_future_24 = pd.DataFrame(index=future_dates)

        if target_date and not df_actual.empty and len(df_actual) == forecast_len:
            # Use actual weather + load if available for historical comparison
            df_future_24['load'] = df_actual['load'].values
            df_future_24['temperature_2m'] = df_actual['temperature_2m'].values
            df_future_24['relative_humidity_2m'] = df_actual['relative_humidity_2m'].values
            df_future_24['wind_speed_10m'] = df_actual['wind_speed_10m'].values
            df_future_24['direct_radiation'] = df_actual['direct_radiation'].values
        else:
            # Naive forecast for weather + load (copy yesterday's values)
            yesterday_covs = df_history.iloc[-forecast_len:][
                ['load', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'direct_radiation']
            ]
            df_future_24['load'] = yesterday_covs['load'].values
            df_future_24['temperature_2m'] = yesterday_covs['temperature_2m'].values
            df_future_24['relative_humidity_2m'] = yesterday_covs['relative_humidity_2m'].values
            df_future_24['wind_speed_10m'] = yesterday_covs['wind_speed_10m'].values
            df_future_24['direct_radiation'] = yesterday_covs['direct_radiation'].values

        # Recompute calendar features for future dates
        df_future_24['hour'] = df_future_24.index.hour
        df_future_24['day_of_week'] = df_future_24.index.dayofweek
        df_future_24['day_of_month'] = df_future_24.index.day
        df_future_24['month'] = df_future_24.index.month
        df_future_24['is_weekend'] = df_future_24['day_of_week'].isin([5, 6]).astype(int)

        df_future_24 = create_cyclic_features(df_future_24, "hour", 24)
        df_future_24 = create_cyclic_features(df_future_24, "day_of_week", 7)
        df_future_24 = create_cyclic_features(df_future_24, "month", 12)

        # Combine past and future for the TFT model (history + forecast horizon for future_covariates).
        # Only 'price' is the target; all other columns (incl. load, weather, calendar)
        # are future covariates the model expects at every timestep.
        df_future_full = pd.concat([df_history.drop(columns=['price']), df_future_24])

        series = TimeSeries.from_series(df_history['price'])

        future_cols = [c for c in df_history.columns if c != 'price']
        future_covs = TimeSeries.from_dataframe(df_future_full, value_cols=future_cols)

        # Scale
        series_scaled = scaler_target.transform(series)
        future_scaled = scaler_future.transform(future_covs)

        # Predict 24 hours into the future using probabilistic sampling
        pred_scaled = model.predict(
            n=24,
            series=series_scaled,
            future_covariates=future_scaled,
            num_samples=100
        )

        # Inverse transform
        pred_real = scaler_target.inverse_transform(pred_scaled)

        # Extract quantiles
        q10_vals = pred_real.quantile(0.1).values().flatten()
        q50_vals = pred_real.quantile(0.5).values().flatten()
        q90_vals = pred_real.quantile(0.9).values().flatten()

        # Format the response
        results = []
        for i, ts in enumerate(pred_real.time_index):
            ts_iso = ts.isoformat()
            res = {
                "timestamp": ts_iso,
                "predicted_price_eur_mwh": float(q50_vals[i]),
                "q10": float(q10_vals[i]),
                "q90": float(q90_vals[i]),
            }
            if ts_iso in actual_prices:
                res["actual_price_eur_mwh"] = actual_prices[ts_iso]
            results.append(res)

        return {
            "country": country,
            "country_name": get_country(country)["name"],
            "forecast": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction for {country}: {e}")
        raise HTTPException(status_code=500, detail=f"[{country}] {e}")


@app.get("/predict")
def predict_next_day(
    country: str = Query(default=None, description="Country code, e.g. CH/PT/ES. "
                          "Defaults to the first loaded country."),
    target_date: Optional[str] = None,
):
    """Single-country 24h forecast (EUR/MWh) with 10/50/90 quantile bands."""
    country = (country or (sorted(MODELS)[0] if MODELS else None))
    if country is None:
        raise HTTPException(status_code=503, detail="No models loaded. Run the pipeline first.")
    country = country.upper()
    if country not in COUNTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown country '{country}'. Available: {sorted(COUNTRIES)}",
        )
    return _forecast_one(country, target_date)


@app.get("/compare")
def compare_countries(
    countries: str = Query(default=None,
                           description="Comma-separated country codes, e.g. CH,PT,ES. "
                                       "Defaults to all loaded countries."),
    target_date: Optional[str] = None,
):
    """Return 24h forecasts for multiple countries in one payload (for overlay plots).

    Each entry mirrors the /predict response. Countries without a loaded model are
    reported in ``skipped`` rather than failing the whole request.
    """
    requested = parse_countries(countries) if countries else sorted(MODELS)
    results = []
    skipped = []
    for c in requested:
        if c not in MODELS:
            skipped.append({"country": c, "reason": "model not loaded"})
            continue
        try:
            results.append(_forecast_one(c, target_date))
        except HTTPException as e:
            skipped.append({"country": c, "reason": e.detail})
    return {"forecasts": results, "skipped": skipped}


def _parse_metrics_file(path: str = "reports/metrics.txt") -> list[dict]:
    """Parse reports/metrics.txt into a list of per-country metric records.

    The file is written as sections beginning with ``=== Country: <CODE> ===`` and
    containing ``MAE:`` / ``RMSE:`` lines under model-name headers. We capture the
    latest MAE/RMSE per (country, model) and label single-shot vs rolling.
    """
    if not os.path.exists(path):
        return []

    with open(path) as f:
        text = f.read()

    records = []
    current_country = None
    current_model = None
    # Track the last seen MAE/RMSE for each (country, model) — later lines win.
    index: dict[tuple, dict] = {}
    order: list[tuple] = []

    # Detect old-format files (no country sections). In that case attribute rows
    # to the first configured country so historical metrics aren't lost.
    has_country_sections = "=== Country:" in text
    fallback_country = DEFAULT_COUNTRIES[0]

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"===\s*Country:\s*([A-Z]{2})\s*===", line)
        if m:
            current_country = m.group(1)
            continue
        # Model header lines look like a name possibly followed by "(...)".
        if line.startswith("MAE:") or line.startswith("RMSE:"):
            country = current_country or (fallback_country if not has_country_sections else None)
            key = (country, current_model)
            if key not in index:
                index[key] = {"country": country, "model": current_model,
                              "mae": None, "rmse": None}
                order.append(key)
            if line.startswith("MAE:"):
                try:
                    index[key]["mae"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            else:
                try:
                    index[key]["rmse"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        else:
            # Treat as a model header (e.g. "Linear Regression", "TFT Model (Rolling Day-Ahead)").
            current_model = line

    for key in order:
        rec = index[key]
        if rec["country"] and rec["model"]:
            records.append(rec)
    return records


@app.get("/metrics")
def get_metrics():
    """Per-country model metrics (MAE/RMSE in EUR/MWh) parsed from reports/metrics.txt."""
    records = _parse_metrics_file()
    return {"metrics": records}


@app.get("/summary")
def price_summary(
    countries: str = Query(default=None,
                           description="Comma-separated country codes. Defaults to all loaded."),
    target_date: Optional[str] = None,
):
    """Price-level summary per country: mean/median/min/max forecast price and peak hour."""
    requested = parse_countries(countries) if countries else sorted(MODELS)
    summaries = []
    for c in requested:
        if c not in MODELS:
            continue
        fc = _forecast_one(c, target_date)["forecast"]
        prices = np.array([h["predicted_price_eur_mwh"] for h in fc], dtype=float)
        if prices.size == 0:
            continue
        peak_idx = int(np.argmax(prices))
        summaries.append({
            "country": c,
            "country_name": get_country(c)["name"],
            "mean": float(np.mean(prices)),
            "median": float(np.median(prices)),
            "min": float(np.min(prices)),
            "max": float(np.max(prices)),
            "peak_hour": fc[peak_idx]["timestamp"],
            "peak_price": float(prices[peak_idx]),
            "n_hours": int(prices.size),
        })
    return {"summary": summaries}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
