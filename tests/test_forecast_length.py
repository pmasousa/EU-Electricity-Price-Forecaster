"""Day-ahead forecast length contract: every forecast is exactly 24 hourly steps.

Mirrors the serving protocol used across the project (``HORIZON = 24`` in
``src/evaluation/backtest.py`` and ``forecast_len = 24`` in ``src/api/main.py``):
a frozen fitted model (``retrain=False``) emitting contiguous, non-overlapping
``forecast_horizon=24, stride=24`` blocks over the tail of an hourly series.

Darts 0.45 notes (verified against this repo):
- ``historical_forecasts`` defaults to ``last_points_only=True`` which keeps only
  the final point of each horizon; we pass ``last_points_only=False`` to get the
  full 24-step blocks (a list of TimeSeries, one per forecast origin).
- ``historical_forecasts(retrain=False)`` requires the model to be fitted first.
- TimeSeries expose ``.to_series()`` (``pd_series()`` was removed).
"""

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel

HORIZON = 24
N_POINTS = 600
START = 480  # first forecast origin; the tail of the series is covered below


def _make_target_and_covariate():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=N_POINTS, freq="h")
    t = np.arange(N_POINTS)
    y = (
        50.0
        + 10.0 * np.sin(2 * np.pi * t / 24)
        + 5.0 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0.0, 1.0, N_POINTS)
    )
    # A "known future" covariate must extend HORIZON steps past the target end:
    # darts requires future covariates to cover every predicted step.
    cov_len = N_POINTS + HORIZON
    t_cov = np.arange(cov_len)
    cov_idx = pd.date_range("2024-01-01", periods=cov_len, freq="h")
    cov = (
        100.0
        + 20.0 * np.cos(2 * np.pi * t_cov / 168)
        + rng.normal(0.0, 2.0, cov_len)
    )
    target = TimeSeries.from_series(pd.Series(y, index=idx, name="price"))
    future_covariate = TimeSeries.from_series(pd.Series(cov, index=cov_idx, name="load"))
    return idx, target, future_covariate


def test_day_ahead_forecast_is_multiple_of_24_covering_tail():
    idx, target, future_covariate = _make_target_and_covariate()

    model = LinearRegressionModel(lags=24, lags_future_covariates=[0])
    # retrain=False below requires a fitted model.
    model.fit(series=target, future_covariates=future_covariate)

    forecasts = model.historical_forecasts(
        series=target,
        future_covariates=future_covariate,
        forecast_horizon=HORIZON,
        stride=HORIZON,
        retrain=False,
        start=START,
        last_points_only=False,
        verbose=False,
        show_warnings=False,
    )

    # last_points_only=False -> one TimeSeries per forecast origin.
    assert isinstance(forecasts, list) and len(forecasts) > 0

    # Each block is a full day-ahead horizon of exactly 24 steps.
    assert all(len(fc) == HORIZON for fc in forecasts)

    # Number of non-overlapping 24h blocks that fit in the tail
    # [START, N_POINTS): ceil((N_POINTS - START) / stride).
    expected_blocks = len(range(START, N_POINTS - HORIZON + 1, HORIZON))
    assert len(forecasts) == expected_blocks

    # Concatenated forecast length is the matching multiple of 24...
    fc_all = forecasts[0]
    for fc in forecasts[1:]:
        fc_all = fc_all.append(fc)
    assert len(fc_all) == expected_blocks * HORIZON == 120
    assert len(fc_all) % HORIZON == 0

    # ...covering exactly the tail of the series with no gaps: the forecast
    # index equals the target's hourly index sliced over the covered window,
    # which by construction is hourly-contiguous.
    fc_index = pd.DatetimeIndex(fc_all.time_index)
    expected_index = idx[START: START + expected_blocks * HORIZON]
    assert list(fc_index) == list(expected_index)
    diffs = pd.Series(fc_index).diff().dropna()
    assert len(diffs) == len(fc_index) - 1  # no missing steps
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_forecast_values_finite_and_index_reconstructible():
    """The forecast round-trips to a pandas series (repo uses ``.to_series()``)."""
    _, target, future_covariate = _make_target_and_covariate()

    model = LinearRegressionModel(lags=24, lags_future_covariates=[0])
    model.fit(series=target, future_covariates=future_covariate)

    forecast = model.predict(n=HORIZON, series=target, future_covariates=future_covariate)
    assert len(forecast) == HORIZON
    values = forecast.to_series().to_numpy()
    assert np.isfinite(values).all()
