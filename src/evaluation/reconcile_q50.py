"""Reconcile the q50 anomaly: deterministic-median MAE vs sampled-empirical-median MAE.

The MAE table scores darts' deterministic predict (the q0.5 head output read
directly); the pinball pass scores the empirical median of num_samples draws
from the fitted distribution. If those two estimators disagree systematically,
the MAE table is scoring probabilistic models with the worse of their medians
and the tables must adopt one consistent estimator.

CPU-only on purpose (GPU may be busy training). Usage:
    python src/evaluation/reconcile_q50.py [--country CH] [--holdout-weeks 8]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
torch.manual_seed(42)
np.random.seed(42)

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel, TFTModel
from sklearn.preprocessing import StandardScaler

from src.evaluation.backtest import (
    HORIZON,
    GlobalTimerCallback,  # noqa: F401 — pickled checkpoints reference it in __main__
    _sample_quantile,
    load_honest_frame,
)


def main(country: str, holdout_weeks: int):
    torch.set_num_threads(8)
    df = load_honest_frame(country)
    y_raw = TimeSeries.from_series(df["price"])
    cov_cols = [c for c in df.columns if c != "price"]
    cov_raw = TimeSeries.from_dataframe(df, value_cols=cov_cols)

    hold_n = holdout_weeks * 168
    val_n = 3 * 168
    train_n = len(df) - hold_n - val_n
    start = train_n + val_n
    n_origins = hold_n // HORIZON

    cov_scaler = Scaler(StandardScaler())
    cov_scaler.fit(cov_raw[:train_n])
    cov_scaled = cov_scaler.transform(cov_raw)
    tsc = Scaler()
    tsc.fit(y_raw[:train_n])
    y_scaled = tsc.transform(y_raw)

    actual = df["price"].iloc[start:].values

    def run(model, name, probabilistic_lgbm=False):
        det_errs, emp200_errs, emp1000_errs = [], [], []
        below10, above90, n_cov = 0, 0, 0
        for k in range(n_origins):
            origin = start + k * HORIZON
            ctx = y_scaled[:origin]
            det = model.predict(
                n=HORIZON, series=ctx, future_covariates=cov_scaled, show_warnings=False
            )
            det_real = tsc.inverse_transform(det).values().flatten()
            det_errs.append(np.abs(actual[k * HORIZON:(k + 1) * HORIZON] - det_real))

            s200 = model.predict(
                n=HORIZON, series=ctx, future_covariates=cov_scaled,
                num_samples=200, show_warnings=False,
            )
            m200 = tsc.inverse_transform(_sample_quantile(s200, 0.5)).values().flatten()
            emp200_errs.append(np.abs(actual[k * HORIZON:(k + 1) * HORIZON] - m200))

            s1000 = model.predict(
                n=HORIZON, series=ctx, future_covariates=cov_scaled,
                num_samples=1000, show_warnings=False,
            )
            m1000 = tsc.inverse_transform(_sample_quantile(s1000, 0.5)).values().flatten()
            emp1000_errs.append(np.abs(actual[k * HORIZON:(k + 1) * HORIZON] - m1000))

            q10 = tsc.inverse_transform(_sample_quantile(s1000, 0.1)).values().flatten()
            q90 = tsc.inverse_transform(_sample_quantile(s1000, 0.9)).values().flatten()
            y = actual[k * HORIZON:(k + 1) * HORIZON]
            below10 += int((y < q10).sum())
            above90 += int((y > q90).sum())
            n_cov += len(y)

        det_mae = float(np.concatenate(det_errs).mean())
        m200 = float(np.concatenate(emp200_errs).mean())
        m1000 = float(np.concatenate(emp1000_errs).mean())
        print(f"\n=== {name} ===")
        print(f"MAE deterministic head:        {det_mae:6.2f}")
        print(f"MAE empirical median (200 s):  {m200:6.2f}")
        print(f"MAE empirical median (1000 s): {m1000:6.2f}")
        print(
            f"coverage: below q10 {100.0 * below10 / n_cov:5.1f}% "
            f"| above q90 {100.0 * above90 / n_cov:5.1f}% (targets: 10% | 10%)"
        )
        return det_mae, m1000

    print(f"[{country}] loading TFT (CPU)...")
    import glob as _glob
    candidates = sorted(
        _glob.glob(f"reports/runs/{country}_minmax/*/tft_model_{country}_minmax.pt")
    )
    if not candidates:
        raise SystemExit("no archived minmax model found — run the harness first")
    tft = TFTModel.load(candidates[-1], map_location="cpu")
    tft.trainer_params["accelerator"] = "cpu"
    tft.trainer_params["devices"] = "auto"
    tft.trainer_params.pop("logger", None)
    run(tft, f"TFT (minmax) — {country}")

    print(f"[{country}] fitting quantile LightGBM...")
    qlgbm = LightGBMModel(
        lags=168,
        lags_future_covariates=[0],
        likelihood="quantile",
        quantiles=[0.1, 0.5, 0.9],
        verbose=-1,
    )
    qlgbm.fit(series=y_scaled[:train_n], future_covariates=cov_scaled)
    run(qlgbm, f"LightGBM (quantile) — {country}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="CH")
    p.add_argument("--holdout-weeks", type=int, default=8)
    args = p.parse_args()
    main(args.country, args.holdout_weeks)
