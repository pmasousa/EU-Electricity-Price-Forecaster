"""Scalers must be fit on the train split ONLY (no leakage from val/test).

Tests ``src/data/dataset.py::load_and_prepare_data`` against a synthetic
features table with EXACTLY the columns ``src/features/build_features.py``
produces.

CRITICAL GUARDRAIL: ``load_and_prepare_data`` overwrites
``models/scaler_target_{cc}.pkl`` and ``models/scaler_future_{cc}.pkl`` as a
side effect, and the serving API depends on those files. We snapshot the CH
scaler files (raw bytes, or None when absent) before the call and restore them
in a ``finally`` block, so the repo state is byte-identical afterwards.

Leakage detection: the synthetic series carries a huge positive AND negative
price spike inside the final 14 days (val + test), so the full-series min/max
differ from the train-only min/max. We then assert both structurally (the
fitted MinMaxScaler's data_min_/data_max_ equal the train stats) and
behaviorally (transform(val) escapes [0, 1], impossible if the scaler had seen
the val spike at fit time).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries

from src.data.dataset import load_and_prepare_data

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# Exact column layout written by src/features/build_features.py (build order:
# renamed raw columns, calendar features, then the sin/cos cyclic encodings).
FEATURE_COLUMNS = [
    "price",
    "load",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "direct_radiation",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

N_DAYS = 90
TEST_DAYS = 7  # load_and_prepare_data defaults
VAL_DAYS = 7  # load_and_prepare_data defaults
TEST_LEN = TEST_DAYS * 24
VAL_LEN = VAL_DAYS * 24
TOTAL_LEN = N_DAYS * 24
TRAIN_LEN = TOTAL_LEN - TEST_LEN - VAL_LEN

SPIKE_HIGH = 999.0  # >> train max (~65)
SPIKE_LOW = -777.0  # << train min (~35)


def _make_features_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # tz-aware UTC index, exactly like build_features.py writes it
    # (dataset.py strips tz with tz_localize(None) after reading).
    idx = pd.date_range("2024-01-01", periods=TOTAL_LEN, freq="h", tz="UTC")
    t = np.arange(TOTAL_LEN)

    price = (
        50.0
        + 10.0 * np.sin(2 * np.pi * t / 24)
        + 5.0 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0.0, 2.0, TOTAL_LEN)
    )
    # Spikes strictly inside the val/test window (last 14 days): the full-series
    # min/max must exceed the train-only min/max in BOTH directions.
    price[TOTAL_LEN - VAL_LEN - TEST_LEN + 100] = SPIKE_HIGH  # inside val
    price[TOTAL_LEN - 20] = SPIKE_LOW  # inside test

    load = 8000.0 + 1500.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0.0, 200.0, TOTAL_LEN)
    temperature_2m = (
        10.0 + 8.0 * np.sin(2 * np.pi * (t - 6 * 24) / (24 * 365))
        + rng.normal(0.0, 2.0, TOTAL_LEN)
    )
    relative_humidity_2m = (
        60.0 + 20.0 * np.sin(2 * np.pi * (t - 3 * 24) / (24 * 30))
        + rng.normal(0.0, 5.0, TOTAL_LEN)
    )
    wind_speed_10m = np.abs(
        6.0 + 4.0 * np.sin(2 * np.pi * t / (24 * 3)) + rng.normal(0.0, 1.5, TOTAL_LEN)
    )
    direct_radiation = np.clip(
        120.0 * np.sin(np.pi * (idx.hour.values - 6) / 12), 0.0, None
    )

    df = pd.DataFrame(index=idx)
    df["price"] = price
    df["load"] = load
    df["temperature_2m"] = temperature_2m
    df["relative_humidity_2m"] = relative_humidity_2m
    df["wind_speed_10m"] = wind_speed_10m
    df["direct_radiation"] = direct_radiation
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"] = idx.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # Same cyclic encodings as create_cyclic_features() in build_features.py.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    assert list(df.columns) == FEATURE_COLUMNS
    return df


def _snapshot(path: Path):
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, blob):
    if blob is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(blob)


def test_scalers_fit_on_train_only(tmp_path):
    features_path = tmp_path / "features_CH.csv"
    _make_features_frame().to_csv(features_path)

    # ---- GUARDRAIL: snapshot the production CH scaler artifacts ----
    target_pkl = MODELS_DIR / "scaler_target_CH.pkl"
    future_pkl = MODELS_DIR / "scaler_future_CH.pkl"
    snap_target = _snapshot(target_pkl)
    snap_future = _snapshot(future_pkl)

    try:
        result = load_and_prepare_data(
            country="CH", processed_data_dir=str(tmp_path)
        )
    finally:
        _restore(target_pkl, snap_target)
        _restore(future_pkl, snap_future)

    # ---- (a) split lengths match the test_days=7 / val_days=7 defaults ----
    assert len(result["train"]) == TRAIN_LEN
    assert len(result["val"]) == VAL_LEN
    assert len(result["test"]) == TEST_LEN

    # ---- (b) the target scaler saw ONLY the train split ----
    # Reload exactly the way dataset.py does (tz stripped).
    df = pd.read_csv(features_path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    price = df["price"]

    train_price = price.iloc[:TRAIN_LEN]
    val_price = price.iloc[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
    assert train_price.min() > SPIKE_LOW and train_price.max() < SPIKE_HIGH, (
        "test fixture broken: spikes must land outside the train split"
    )
    # Sanity: full-series stats really do differ from train-only stats.
    assert price.max() == SPIKE_HIGH > train_price.max()
    assert price.min() == SPIKE_LOW < train_price.min()

    scaler_target = result["scaler_target"]
    fitted_stats = None
    fitted_params = getattr(scaler_target, "_fitted_params", None)
    if isinstance(fitted_params, (list, tuple)) and len(fitted_params) > 0:
        sklearn_scaler = fitted_params[0]
        if hasattr(sklearn_scaler, "data_min_") and hasattr(sklearn_scaler, "data_max_"):
            fitted_stats = (
                float(np.min(sklearn_scaler.data_min_)),
                float(np.max(sklearn_scaler.data_max_)),
            )

    if fitted_stats is not None:
        # Structural assertion: fitted MinMaxScaler stats == train-only stats.
        fitted_min, fitted_max = fitted_stats
        assert np.isclose(fitted_min, train_price.min()), (
            f"scaler data_min_ {fitted_min} != train min {train_price.min()} "
            f"(full-series min is {price.min()} — leakage?)"
        )
        assert np.isclose(fitted_max, train_price.max()), (
            f"scaler data_max_ {fitted_max} != train max {train_price.max()} "
            f"(full-series max is {price.max()} — leakage?)"
        )
    else:
        # Behavioral fallback: if the minmax had been fit on data including
        # val, every transformed value would sit inside [0, 1].
        val_series = TimeSeries.from_series(val_price)
        val_scaled = scaler_target.transform(val_series).values().flatten()
        assert np.any(val_scaled > 1.0) or np.any(val_scaled < 0.0), (
            "transform(val) stays inside [0, 1] — scaler appears to have seen "
            "val data at fit time (leakage)"
        )

    # Behavioral check in addition to the structural one (whichever ran above):
    # the val spike (SPIKE_HIGH, inside val) must transform to > 1 under a
    # train-only minmax fit.
    val_series = TimeSeries.from_series(val_price)
    val_scaled = scaler_target.transform(val_series).values().flatten()
    assert np.any(val_scaled > 1.0), (
        "expected transform(val) values above 1.0 from the val-week spike; "
        "got max "
        f"{val_scaled.max()}"
    )

    # The returned val/test splits are transforms (not re-fits) of raw data —
    # they must contain out-of-range values too, i.e. no hidden refit happened.
    assert np.any(result["val"].values().flatten() > 1.0)
