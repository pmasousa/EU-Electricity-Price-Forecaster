"""Honest covariate frame — one definition shared by benchmarking and serving.

Realized load is NOT known at day-ahead gate closure, so the frame replaces it
with ``load_lag24`` / ``load_lag168`` (values from <= forecast origin, always
knowable). Weather at forecast time is realized weather used as a day-ahead
forecast proxy (stated assumption, standard in the EPF literature).

Consumers: ``src/evaluation/backtest.py`` (training/evaluation) and
``src/api/main.py`` (live serving). Both must build identical covariate frames
in an identical column order — that order is ``COV_COLUMNS`` below.
"""

import os

import numpy as np
import pandas as pd

COV_COLUMNS = [
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
    "load_lag24",
    "load_lag168",
]

WEATHER_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "direct_radiation"]


def read_features(country: str, processed_data_dir: str = "data/processed") -> pd.DataFrame:
    """Load the raw feature table (tz-stripped, hourly DatetimeIndex)."""
    path = os.path.join(processed_data_dir, f"features_{country}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run src/features/build_features.py first."
        )
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df


def with_load_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Full honest frame (no row dropping): raw load replaced by its lags,
    columns in canonical [price] + COV_COLUMNS order. First 168 rows carry NaN
    lags — callers must slice past them."""
    df = df.sort_index()
    df["load_lag24"] = df["load"].shift(24)
    df["load_lag168"] = df["load"].shift(168)
    df = df.drop(columns=["load"])
    return df[["price"] + COV_COLUMNS]


def load_honest_frame(country: str, processed_data_dir: str = "data/processed") -> pd.DataFrame:
    """Training/evaluation frame: raw load replaced by its lags, canonical order.

    Refuses to run on smoke-test-sized data (< 180 days).
    """
    df = read_features(country, processed_data_dir)
    span_days = (df.index[-1] - df.index[0]).days
    if span_days < 180:
        raise SystemExit(
            f"features_{country}.csv spans only {span_days} days — refusing to "
            f"benchmark on smoke-test data. Re-download with --days >= 365*2."
        )
    return with_load_lags(df).dropna()


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=index)
    cal["hour"] = index.hour
    cal["day_of_week"] = index.dayofweek
    cal["day_of_month"] = index.day
    cal["month"] = index.month
    cal["is_weekend"] = cal["day_of_week"].isin([5, 6]).astype(int)
    cal["hour_sin"] = np.sin(2 * np.pi * cal["hour"] / 24)
    cal["hour_cos"] = np.cos(2 * np.pi * cal["hour"] / 24)
    cal["day_of_week_sin"] = np.sin(2 * np.pi * cal["day_of_week"] / 7)
    cal["day_of_week_cos"] = np.cos(2 * np.pi * cal["day_of_week"] / 7)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)
    return cal


def future_covariate_rows(
    features: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Covariate rows for the forecast horizon, in canonical COV_COLUMNS order.

    ``features`` is the FULL raw feature table (with raw ``load``). Loads are
    lagged from history (t-24 / t-168 always lie at or before the forecast
    origin). ``weather``: actuals aligned to ``future_index`` for retroactive
    dates, or None to use the day-ahead proxy (copy of the last 24 realized
    hours).
    """
    features = features.sort_index()

    if weather is None:
        last_day = features[WEATHER_COLUMNS].iloc[-24:]
        n = len(future_index)
        weather = pd.DataFrame(
            np.tile(last_day.values, (n // 24 + 1, 1))[:n],
            index=future_index,
            columns=WEATHER_COLUMNS,
        )

    cal = _calendar_features(future_index)

    load = features["load"].reindex(features.index.union(future_index))
    lags = pd.DataFrame(index=future_index)
    lags["load_lag24"] = load.shift(24).reindex(future_index)
    lags["load_lag168"] = load.shift(168).reindex(future_index)

    frame = pd.concat([weather[WEATHER_COLUMNS], cal, lags], axis=1)
    return frame[COV_COLUMNS]
