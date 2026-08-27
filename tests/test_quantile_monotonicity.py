"""Quantile outputs must satisfy q10 <= q50 <= q90 at every horizon step.

Trains a deliberately tiny TFTModel on CPU (the GPU may be busy with a real
training run) with a QuantileRegression(quantiles=(0.1, 0.5, 0.9)) likelihood on
a deterministic synthetic hourly series, then checks that sampled predictions
produce monotone empirical 10/50/90 percentiles across the 24 horizon steps —
the same invariant the serving API relies on when it reports quantile bands
from ``pred_real.quantile(...)`` in ``src/api/main.py``.
"""

import time

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

N_POINTS = 800
QUANTILES = (0.1, 0.5, 0.9)
NUM_SAMPLES = 200
RUNTIME_BUDGET_SECONDS = 90.0


def _make_series() -> TimeSeries:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=N_POINTS, freq="h")
    t = np.arange(N_POINTS)
    y = (
        50.0
        + 10.0 * np.sin(2 * np.pi * t / 24)
        + 3.0 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0.0, 1.5, N_POINTS)
    )
    return TimeSeries.from_series(pd.Series(y, index=idx, name="price"))


def test_tft_quantile_outputs_are_monotone():
    series = _make_series()

    model = TFTModel(
        input_chunk_length=24,
        output_chunk_length=24,
        hidden_size=8,
        num_attention_heads=2,
        lstm_layers=1,
        n_epochs=2,
        batch_size=32,
        random_state=42,
        add_relative_index=True,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
        },
        likelihood=QuantileRegression(quantiles=QUANTILES),
    )

    t0 = time.monotonic()
    model.fit(series, verbose=False)
    pred = model.predict(n=24, num_samples=NUM_SAMPLES)
    elapsed = time.monotonic() - t0

    assert len(pred) == 24
    # all_values() -> ndarray of shape (n_time, n_components, n_samples)
    samples = pred.all_values(copy=False)[:, 0, :]
    assert samples.shape == (24, NUM_SAMPLES)
    assert np.isfinite(samples).all()

    q10 = np.percentile(samples, 10, axis=1)
    q50 = np.percentile(samples, 50, axis=1)
    q90 = np.percentile(samples, 90, axis=1)

    for step in range(24):
        assert q10[step] <= q50[step], (
            f"step {step}: q10 ({q10[step]:.4f}) > q50 ({q50[step]:.4f})"
        )
        assert q50[step] <= q90[step], (
            f"step {step}: q50 ({q50[step]:.4f}) > q90 ({q90[step]:.4f})"
        )

    # Cheap regression guard: this tiny-CPU configuration must stay fast so the
    # whole suite keeps its ~3-minute budget.
    assert elapsed < RUNTIME_BUDGET_SECONDS, (
        f"TFT fit+predict took {elapsed:.1f}s (budget {RUNTIME_BUDGET_SECONDS}s)"
    )
