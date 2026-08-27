import json
import os
import pickle
import re
import sys
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
import uvicorn
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models.torch import QuantileRegression
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pytorch_lightning.callbacks import Callback

# Allow running as a module: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import (
    COUNTRIES,
    DEFAULT_COUNTRIES,
    DEFAULT_COUNTRY,
    get_country,
    parse_countries,
)
from src.data.honest import COV_COLUMNS, future_covariate_rows, with_load_lags


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
    __main__.GlobalTimerCallback = GlobalTimerCallback
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([QuantileRegression, GlobalTimerCallback])


def _load_country_model(country: str):
    """Load one country's serving bundle (built by src/models/build_serving.py).

    Returns None if missing or load-failed. A failed load is logged and treated
    as "not available" rather than crashing startup, so the API can still serve
    other countries and the metadata-only endpoints.
    """
    serving_dir = f"models/serving_{country}"
    model_path = os.path.join(serving_dir, "tft_model.pt")
    scaler_target_path = os.path.join(serving_dir, "scaler_target.pkl")
    scaler_cov_path = os.path.join(serving_dir, "scaler_cov.pkl")
    config_path = os.path.join(serving_dir, "config.json")

    if not all(os.path.exists(p) for p in
               [model_path, scaler_target_path, scaler_cov_path, config_path]):
        return None

    try:
        _register_callback_for_unpickling()
        # No weights_only kwarg: darts 0.45 forwards extra kwargs into PL's
        # load_from_checkpoint where they collide with the module constructor.
        model = TFTModel.load(model_path, map_location="cpu")
        if hasattr(model, 'trainer_params'):
            model.trainer_params['accelerator'] = 'cpu'
            model.trainer_params['devices'] = 1
            # Drop the training-time CSVLogger: it points at the training
            # machine's log folder, which a serving host doesn't have. darts
            # rebuilds a throwaway logger for predict.
            model.trainer_params.pop('logger', None)

        with open(scaler_target_path, "rb") as f:
            scaler_target = pickle.load(f)
        with open(scaler_cov_path, "rb") as f:
            scaler_cov = pickle.load(f)
        with open(config_path) as f:
            config = json.load(f)

        return {
            "model": model,
            "scaler_target": scaler_target,
            "scaler_cov": scaler_cov,
            "config": config,
        }
    except Exception as e:
        print(f"Warning: failed to load serving bundle for {country} "
              f"({type(e).__name__}: {e}). Skipping — rebuild with "
              f"`python src/models/build_serving.py --country {country}`.")
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


def _forecast_one(country: str, target_date: str | None = None) -> dict:
    """Core single-country forecast. Returns a payload dict with quantile bands."""
    if country not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"No loaded model for country '{country}'. Loaded: {sorted(MODELS)}",
        )

    bundle = MODELS[country]
    model = bundle["model"]
    scaler_target = bundle["scaler_target"]
    scaler_cov = bundle["scaler_cov"]

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
                raise HTTPException(
                    status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
                ) from None

            # The honest covariates lag load by 24h/168h, and the 168h history
            # window must itself carry valid lags -> earliest target is
            # data_start + 168h (lags) + 168h (history).
            if target_start <= df.index[0] + pd.Timedelta(hours=history_len + 168):
                raise HTTPException(
                    status_code=400, detail="Date too early, not enough historical data."
                ) from None

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

        # Covariates through the shared honest builder (src/data/honest.py) —
        # identical construction to what the model was trained on: calendar +
        # weather (actuals for retroactive dates, else the last-24h proxy) +
        # load lagged 24h/168h (realized load at forecast time is not knowable).
        last_date = df_history.index[-1]
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(hours=1), periods=forecast_len, freq="h"
        )

        if target_date and not df_actual.empty and len(df_actual) == forecast_len:
            weather = df_actual[["temperature_2m", "relative_humidity_2m",
                                 "wind_speed_10m", "direct_radiation"]]
        else:
            weather = None

        honest_frame = with_load_lags(df)
        cov_history = honest_frame.loc[df_history.index[0]:last_date, COV_COLUMNS]
        cov_future = future_covariate_rows(df, future_index, weather=weather)
        cov_full = pd.concat([cov_history, cov_future])

        series = TimeSeries.from_series(df_history['price'])
        future_covs = TimeSeries.from_dataframe(cov_full, value_cols=COV_COLUMNS)

        # Scale
        series_scaled = scaler_target.transform(series)
        future_scaled = scaler_cov.transform(future_covs)

        # Predict 24 hours into the future using probabilistic sampling
        pred_scaled = model.predict(
            n=24,
            series=series_scaled,
            future_covariates=future_scaled,
            num_samples=100
        )

        # Inverse transform
        pred_real = scaler_target.inverse_transform(pred_scaled)

        # Extract quantiles (and fail loudly instead of serving NaNs)
        q10_vals = pred_real.quantile(0.1).values().flatten()
        q50_vals = pred_real.quantile(0.5).values().flatten()
        q90_vals = pred_real.quantile(0.9).values().flatten()
        if not (np.isfinite(q10_vals).all() and np.isfinite(q50_vals).all()
                and np.isfinite(q90_vals).all()):
            raise HTTPException(
                status_code=500,
                detail=f"[{country}] non-finite forecast values — check the "
                       f"covariate history for gaps.",
            )

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
        raise HTTPException(status_code=500, detail=f"[{country}] {e}") from e


@app.get("/predict")
def predict_next_day(
    country: str = Query(default=None, description="Country code, e.g. CH/PT/ES. "
                          "Defaults to the first loaded country."),
    target_date: str | None = None,
):
    """Single-country 24h forecast (EUR/MWh) with 10/50/90 quantile bands."""
    country = (
        country
        or (DEFAULT_COUNTRY if DEFAULT_COUNTRY in MODELS else None)
        or (sorted(MODELS)[0] if MODELS else None)
    )
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
    target_date: str | None = None,
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


def _parse_benchmark_tables(latest_dir: str = "reports/latest") -> list[dict]:
    """Parse reports/latest/benchmark_*.txt into per-country metric records.

    Table rows are fixed-width (name, MAE, RMSE, rMAE); the parser splits on
    2+ spaces so it survives format tweaks. Only the point-forecast table is
    read — pinball/coverage sections are skipped.
    """
    records: list[dict] = []
    if not os.path.isdir(latest_dir):
        return records

    for fname in sorted(os.listdir(latest_dir)):
        if not (fname.startswith("benchmark_") and fname.endswith(".txt")):
            continue
        path = os.path.join(latest_dir, fname)
        country = None
        with open(path) as f:
            for raw in f:
                line = raw.rstrip("\n")
                m = re.match(r"Benchmark ([A-Z]{2}) ", line)
                if m:
                    country = m.group(1)
                    continue
                if line.startswith("Pinball"):
                    break  # point table ends here
                if country and re.match(r"\S.*\s{2,}-?\d+\.\d{2}\s+-?\d+\.\d{2}", line):
                    parts = re.split(r"\s{2,}", line.strip())
                    if len(parts) >= 3:
                        try:
                            records.append({
                                "country": country,
                                "model": parts[0],
                                "mae": float(parts[1]),
                                "rmse": float(parts[2]),
                                **({"rmae": float(parts[3])} if len(parts) > 3 else {}),
                                "source": f"reports/latest/{fname}",
                            })
                        except ValueError:
                            continue
    return records


@app.get("/metrics")
def get_metrics():
    """Per-country benchmark metrics (MAE/RMSE/rMAE in EUR/MWh) from the
    latest harness tables in reports/latest/."""
    return {"metrics": _parse_benchmark_tables()}


@app.get("/summary")
def price_summary(
    countries: str = Query(default=None,
                           description="Comma-separated country codes. Defaults to all loaded."),
    target_date: str | None = None,
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
