"""Shared walk-forward benchmark harness — the single source of truth for model comparison.

Every model (naive, linear, LightGBM, TFT) is evaluated through the exact same
protocol: trained on the same data, given the same covariates, and scored on
the same day-ahead walk-forward pass over the holdout
(`forecast_horizon=24, stride=24, retrain=False`).

Honesty rules baked in:
- Realized load is NOT known at day-ahead gate closure, so the covariate frame
  carries only `load_lag24` / `load_lag168` (yesterday's / last week's load).
- Weather at forecast time t is realized weather used as a proxy for the
  day-ahead weather forecast (stated assumption, standard in the literature).
- Target scalers (and covariate scaler) are fit on the train split only.
- The TFT target transform is configurable (`minmax` vs `asinh`): 3 years of CH
  prices span -464..+319 EUR/MWh, so plain min-max squashes the typical band
  and lets spike tails dominate the quantile loss. `asinh` + z-score is the
  standard spike-robust alternative.

Baselines always use the minmax pipeline regardless of --target-transform, so
variant runs move only the TFT row.

Run folders: every run writes ALL of its outputs (benchmark table, forecast
series, comparison plot, loss curve, trained model, feature-table snapshot)
directly into ``reports/runs/{country}_{label}/{timestamp}/`` at run time —
nothing is ever overwritten and every run's evidence lives in exactly one
folder. ``reports/latest/`` holds a convenience copy of each variant's most
recent benchmark table; ``models/`` holds the current model per variant.
"""

import argparse
import datetime
import glob
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")
# joblib/loky counts physical cores by shelling out to `wmic`, which no longer
# exists on Windows 11 26200+; its fallback (logical cores) is fine for us.
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
torch.set_float32_matmul_precision("high")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse
from darts.models import LightGBMModel, LinearRegressionModel, TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler

from src.config import get_country
from src.data.honest import load_honest_frame  # shared with the serving API

HORIZON = 24
LGBM_SILENCE = {"verbose": -1}

# Matches the darts TFT default quantile set — the control configuration.
DEFAULT_QUANTILES = (
    "0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99"
)


class GlobalTimerCallback(Callback):
    def __init__(self):
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        done = trainer.current_epoch + 1
        remaining = (trainer.max_epochs - done) * (elapsed / done)
        print(
            f"[Timer] epoch {done}/{trainer.max_epochs} | "
            f"elapsed {datetime.timedelta(seconds=int(elapsed))} | "
            f"eta {datetime.timedelta(seconds=int(remaining))}",
            flush=True,
        )


# load_honest_frame lives in src/data/honest.py — one definition shared by the
# benchmark harness and the serving API. Imported above.


def build_tft(
    target_transform: str,
    quantiles: list[float],
    epochs: int,
    country: str,
    label: str,
    hidden_size: int = 64,
    dropout: float = 0.3,
    es_patience: int = 10,
    es_min_delta: float = 1e-3,
) -> TFTModel:
    return TFTModel(
        input_chunk_length=168,
        output_chunk_length=HORIZON,
        hidden_size=hidden_size,
        lstm_layers=2,
        num_attention_heads=8,
        dropout=dropout,
        batch_size=256,
        n_epochs=epochs,
        add_relative_index=True,
        random_state=42,
        likelihood=QuantileRegression(quantiles=quantiles),
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={"patience": 4, "factor": 0.5},
        pl_trainer_kwargs={
            "logger": CSVLogger("reports/logs", name=f"tft_logs_{country}_{label}"),
            "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else "auto",
            "callbacks": [
                GlobalTimerCallback(),
                EarlyStopping(
                    monitor="val_loss", patience=es_patience, min_delta=es_min_delta, mode="min"
                ),
            ],
        },
    )


def walk_forward(model, series, start: int, future_covariates=None) -> TimeSeries:
    """One code path for every model: frozen-model day-ahead backtest.

    ``last_points_only=False`` returns one series per 24h window; we stitch
    them into a single contiguous forecast. darts' default (True) would keep
    ONLY the last horizon hour of each window — silently turning an all-hours
    day-ahead metric into an hour-23-only metric.
    """
    kwargs = {}
    if future_covariates is not None:
        kwargs["future_covariates"] = future_covariates
    forecasts = model.historical_forecasts(
        series=series,
        start=start,
        forecast_horizon=HORIZON,
        stride=HORIZON,
        retrain=False,
        verbose=False,
        show_warnings=False,
        last_points_only=False,
        **kwargs,
    )
    fc = forecasts[0]
    for window in forecasts[1:]:
        fc = fc.append(window)
    return fc


def sinh_inverse(ts: TimeSeries) -> TimeSeries:
    freq = None
    try:
        freq = ts.freq
    except Exception:
        pass
    return TimeSeries.from_times_and_values(ts.time_index, np.sinh(ts.values()), freq=freq)


PINBALL_TAUS = (0.1, 0.5, 0.9)


def _sample_quantile(pred: TimeSeries, tau: float) -> TimeSeries:
    """Per-step quantile across samples; darts <0.46 uses .quantile(), newer
    versions .quantile_timeseries()."""
    if hasattr(pred, "quantile_timeseries"):
        return pred.quantile_timeseries(tau)
    vals = pred.all_values(copy=False)  # (time, components, samples)
    q = np.quantile(vals, tau, axis=-1)
    return TimeSeries.from_times_and_values(pred.time_index, q)


def pinball_walk_forward(model, series, start: int, scaler, future_covs=None, inverse_fn=None,
                         num_samples: int = 200, taus=PINBALL_TAUS) -> dict[float, TimeSeries]:
    """Probabilistic walk-forward: per-origin sampled predictions reduced to
    empirical quantile curves. Returns {tau: quantile forecast series} in EUR."""
    per_tau: dict[float, list] = {tau: [] for tau in taus}
    n_origins = (len(series) - start) // HORIZON
    for k in range(n_origins):
        origin = start + k * HORIZON
        pred = model.predict(
            n=HORIZON,
            series=series[:origin],
            future_covariates=future_covs,
            num_samples=num_samples,
            show_warnings=False,
        )
        for tau in taus:
            q_real = scaler.inverse_transform(_sample_quantile(pred, tau))
            if inverse_fn is not None:
                q_real = inverse_fn(q_real)
            per_tau[tau].append(q_real)
    out = {}
    for tau, parts in per_tau.items():
        merged = parts[0]
        for nxt in parts[1:]:
            merged = merged.append(nxt)
        out[tau] = merged
    return out


def pinball_scores(actual: TimeSeries, quantile_fcs: dict) -> dict:
    """Mean pinball (quantile) loss per tau plus their average, in EUR/MWh.
    Also reports empirical breach rates (coverage) for the outer quantiles —
    a q10 target of 10% actuals-below / q90 target of 10% actuals-above."""
    y = actual.values().flatten()
    scores = {}
    for tau, fc in quantile_fcs.items():
        q = fc.values().flatten()
        diff = y - q
        scores[tau] = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
    scores["mean"] = float(np.mean([scores[t] for t in quantile_fcs]))
    if 0.1 in quantile_fcs and 0.9 in quantile_fcs:
        scores["cov_below_q10"] = float(np.mean(y < quantile_fcs[0.1].values().flatten()) * 100.0)
        scores["cov_above_q90"] = float(np.mean(y > quantile_fcs[0.9].values().flatten()) * 100.0)
    return scores


def plot_loss_curve(country: str, label: str, out_path: str, title_extra: str = "") -> str | None:
    """Train/val loss curve for the latest fitting of this variant, with the
    best-val epoch marked. Returns the metrics directory (or None)."""
    log_root = os.path.join("reports", "logs", f"tft_logs_{country}_{label}")
    versions = sorted(
        (d for d in glob.glob(os.path.join(log_root, "version_*")) if os.path.isdir(d)),
        key=lambda d: int(d.rsplit("version_", 1)[-1]),
    )
    with_metrics = [d for d in versions if os.path.exists(os.path.join(d, "metrics.csv"))]
    if not with_metrics:
        return None
    m = pd.read_csv(os.path.join(with_metrics[-1], "metrics.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    if "train_loss" in m.columns:
        tr = m[["epoch", "train_loss"]].dropna().groupby("epoch").mean()
        ax.plot(tr.index, tr["train_loss"], color="tab:blue", lw=1.5, label="train_loss")
    if "val_loss" in m.columns:
        vl = m[["epoch", "val_loss"]].dropna().groupby("epoch").mean()
        ax.plot(vl.index, vl["val_loss"], color="tab:orange", lw=2.0, label="val_loss")
        best_ep = int(vl["val_loss"].idxmin())
        ax.axvline(best_ep, color="gray", ls=":", lw=1)
        ax.annotate(
            f"best val {vl['val_loss'].min():.4f} @ epoch {best_ep}",
            xy=(0.02, 0.95), xycoords="axes fraction", va="top",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"[{country}] TFT training curve — {label}{title_extra}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return with_metrics[-1]


def run_benchmark(
    country: str,
    holdout_weeks: int,
    val_weeks: int,
    epochs: int,
    target_transform: str,
    quantiles: list[float],
    skip_tft: bool,
    load_tft: str | None = None,
    label: str | None = None,
    hidden_size: int = 64,
    dropout: float = 0.3,
    es_patience: int = 10,
    es_min_delta: float = 1e-3,
):
    if label is None:
        label = target_transform if not skip_tft else f"{target_transform}-baselines"
    run_dir = os.path.join(
        "reports", "runs", f"{country}_{label}",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"[{country}] run folder: {run_dir}", flush=True)
    get_country(country)

    df = load_honest_frame(country)
    print(f"[{country}] rows {len(df)} | {df.index[0]} -> {df.index[-1]}", flush=True)

    y_raw = TimeSeries.from_series(df["price"])
    cov_cols = [c for c in df.columns if c != "price"]
    cov_raw = TimeSeries.from_dataframe(df, value_cols=cov_cols)

    hold_n = holdout_weeks * 168
    val_n = val_weeks * 168
    train_n = len(df) - hold_n - val_n
    start = train_n + val_n
    print(
        f"[{country}] split: train {train_n}h | val {val_n}h | holdout {hold_n}h "
        f"({df.index[start]} -> {df.index[-1]})",
        flush=True,
    )

    # Covariates: z-scored, fit on train only, covering the full span.
    cov_scaler = Scaler(StandardScaler())
    cov_scaler.fit(cov_raw[:train_n])
    cov_scaled = cov_scaler.transform(cov_raw)

    # Baseline pipeline: minmax on raw prices (fit on train only) — fixed across
    # all TFT variants so baselines are directly comparable run-to-run.
    base_scaler = Scaler()
    base_scaler.fit(y_raw[:train_n])
    y_base = base_scaler.transform(y_raw)

    # TFT pipeline: configurable target transform (fit on train only).
    if target_transform == "asinh":
        y_tft_space = TimeSeries.from_series(np.arcsinh(df["price"]))
        tft_scaler = Scaler(StandardScaler())
    else:
        y_tft_space = y_raw
        tft_scaler = Scaler()
    tft_scaler.fit(y_tft_space[:train_n])
    y_tft = tft_scaler.transform(y_tft_space)

    results: dict[str, dict] = {}

    def score(name: str, fc_real: TimeSeries):
        mae_score = mae(y_raw[start:], fc_real)
        rmse_score = rmse(y_raw[start:], fc_real)
        results[name] = {"mae": mae_score, "rmse": rmse_score, "forecast": fc_real}
        # rMAE vs the t-24 naive (scored first) — the benchmark-relative skill
        # metric from the EPF literature; <1 means beats yesterday's curve.
        naive_mae = results.get("Naive persistence (t-24)", {}).get("mae")
        rmae = mae_score / naive_mae if naive_mae else float("nan")
        results[name]["rmae"] = rmae
        print(
            f"[{country}] {name:<28} MAE {mae_score:7.2f}  RMSE {rmse_score:8.2f}  "
            f"rMAE {rmae:5.2f}",
            flush=True,
        )

    def evaluate(name: str, model, series, scaler, future_covs, inverse_fn=None):
        t0 = time.time()
        fc_real = scaler.inverse_transform(walk_forward(model, series, start, future_covs))
        if inverse_fn is not None:
            fc_real = inverse_fn(fc_real)
        score(name, fc_real)
        print(f"    (backtest took {time.time() - t0:.0f}s)", flush=True)

    print(f"[{country}] --- naive baselines (pandas shifts on actuals) ---", flush=True)
    price_pd = df["price"]
    score("Naive persistence (t-24)", TimeSeries.from_series(price_pd.shift(24).iloc[start:]))
    score("Naive weekly (t-168)", TimeSeries.from_series(price_pd.shift(168).iloc[start:]))

    print(f"[{country}] --- classical baselines ---", flush=True)
    lr = LinearRegressionModel(lags=168, lags_future_covariates=[0])
    lr.fit(series=y_base[:train_n], future_covariates=cov_scaled)
    evaluate("Linear Regression", lr, y_base, base_scaler, cov_scaled)

    lgbm = LightGBMModel(lags=168, lags_future_covariates=[0], **LGBM_SILENCE)
    lgbm.fit(series=y_base[:train_n], future_covariates=cov_scaled)
    evaluate("LightGBM", lgbm, y_base, base_scaler, cov_scaled)

    tft = None
    inv = None
    model_path = None
    if load_tft:
        if not os.path.exists(load_tft):
            raise SystemExit(f"--load-tft: {load_tft} not found")
        print(f"[{country}] --- TFT loaded from {load_tft} (no retraining) ---", flush=True)
        tft = TFTModel.load(load_tft, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model_path = load_tft
        inv = sinh_inverse if target_transform == "asinh" else None
        evaluate(f"TFT ({target_transform})", tft, y_tft, tft_scaler, cov_scaled, inverse_fn=inv)
    elif not skip_tft:
        print(
            f"[{country}] --- TFT (transform={target_transform}, "
            f"quantiles={len(quantiles)}, max epochs={epochs}) ---",
            flush=True,
        )
        tft = build_tft(
            target_transform,
            quantiles,
            epochs,
            country,
            label,
            hidden_size=hidden_size,
            dropout=dropout,
            es_patience=es_patience,
            es_min_delta=es_min_delta,
        )
        t0 = time.time()
        tft.fit(
            series=y_tft[:train_n],
            future_covariates=cov_scaled,
            val_series=y_tft[train_n - 168 : train_n].append(y_tft[train_n : train_n + val_n]),
            val_future_covariates=cov_scaled,
            verbose=True,
        )
        print(f"[{country}] TFT trained in {time.time() - t0:.0f}s", flush=True)
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join(run_dir, f"tft_model_{country}_{label}.pt")
        tft.save(model_path)
        # darts writes the fitted weights to a sibling .ckpt (fit-time cwd) —
        # the run folder must keep BOTH or the archived model is weightless.
        ckpt_sibling = model_path + ".ckpt"
        if os.path.exists(ckpt_sibling):
            pass
        else:
            fit_ckpt = os.path.join("models", os.path.basename(model_path) + ".ckpt")
            if os.path.exists(fit_ckpt):
                shutil.copy2(fit_ckpt, ckpt_sibling)
        shutil.copy2(model_path, os.path.join("models", os.path.basename(model_path)))
        if os.path.exists(ckpt_sibling):
            shutil.copy2(ckpt_sibling, os.path.join("models", os.path.basename(ckpt_sibling)))
        print(f"[{country}] TFT saved to {model_path} (current copy in models/)", flush=True)
        metrics_dir = plot_loss_curve(
            country, label, os.path.join(run_dir, f"loss_curve_{country}_{label}.png"),
            title_extra=f" (hidden {hidden_size}, dropout {dropout})",
        )
        if metrics_dir:
            for fname in ("metrics.csv", "hparams.yaml"):
                src = os.path.join(metrics_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, run_dir)

        inv = sinh_inverse if target_transform == "asinh" else None
        evaluate(f"TFT ({target_transform})", tft, y_tft, tft_scaler, cov_scaled, inverse_fn=inv)

    # --- Probabilistic evaluation: pinball loss at q10/50/90 ---
    pinball_rows: dict[str, dict] = {}
    print(f"[{country}] --- probabilistic evaluation (pinball @ q10/50/90) ---", flush=True)
    t0 = time.time()
    qlgbm = LightGBMModel(
        lags=168,
        lags_future_covariates=[0],
        likelihood="quantile",
        quantiles=list(PINBALL_TAUS),
        **LGBM_SILENCE,
    )
    qlgbm.fit(series=y_base[:train_n], future_covariates=cov_scaled)
    pinball_rows["LightGBM (quantile)"] = pinball_scores(
        y_raw[start:], pinball_walk_forward(qlgbm, y_base, start, base_scaler, cov_scaled)
    )
    if tft is not None:
        pinball_rows[f"TFT ({label})"] = pinball_scores(
            y_raw[start:],
            pinball_walk_forward(tft, y_tft, start, tft_scaler, cov_scaled, inverse_fn=inv),
        )
    for name, s in pinball_rows.items():
        print(
            f"[{country}] {name:<28} pinball q10 {s[0.1]:6.2f}  q50 {s[0.5]:6.2f}  "
            f"q90 {s[0.9]:6.2f}  mean {s['mean']:6.2f}  | "
            f"breaches: <q10 {s.get('cov_below_q10', float('nan')):4.1f}% "
            f">q90 {s.get('cov_above_q90', float('nan')):4.1f}% (targets 10/10)",
            flush=True,
        )
    print(f"    (pinball pass took {time.time() - t0:.0f}s)", flush=True)

    # ---- outputs (all inside this run's folder) ----

    lines = [
        f"Benchmark {country} — generated {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"Data: {len(df)} hourly rows, {df.index[0]} -> {df.index[-1]}",
        f"Splits: train {train_n}h / val {val_n}h (TFT early stop) / holdout {hold_n}h",
        f"Protocol: day-ahead walk-forward, horizon {HORIZON}h, stride {HORIZON}h, "
        f"retrain=False (frozen models), MAE/RMSE on EUR/MWh",
        "Covariates: calendar + weather-at-t (realized weather as day-ahead forecast "
        "proxy) + load_lag24/load_lag168 (realized load excluded — not known day-ahead)",
        f"TFT target transform: {target_transform}; quantiles: {quantiles}",
        "",
        f"{'Model':<30}{'MAE':>10}{'RMSE':>10}{'rMAE':>8}",
        "-" * 58,
    ]
    for name, r in results.items():
        lines.append(f"{name:<30}{r['mae']:>10.2f}{r['rmse']:>10.2f}{r['rmae']:>8.2f}")
    if pinball_rows:
        lines += [
            "",
            "Pinball loss (EUR/MWh), taus 0.1/0.5/0.9. Breach rates should be ~10%/10%;",
            "higher >q90 breach = the model underestimates upside tail risk.",
            f"{'Model':<30}{'q10':>8}{'q50':>8}{'q90':>8}{'mean':>8}{'<q10%':>8}{'>q90%':>8}",
            "-" * 78,
        ]
        for name, s in pinball_rows.items():
            lines.append(
                f"{name:<30}{s[0.1]:>8.2f}{s[0.5]:>8.2f}{s[0.9]:>8.2f}{s['mean']:>8.2f}"
                f"{s.get('cov_below_q10', float('nan')):>8.1f}"
                f"{s.get('cov_above_q90', float('nan')):>8.1f}"
            )
    bench_path = os.path.join(run_dir, f"benchmark_{country}_{label}.txt")
    with open(bench_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{country}] wrote {bench_path}", flush=True)

    # Per-model forecast series for downstream plots/tests.
    fc_df = pd.DataFrame(index=y_raw[start:].time_index)
    fc_df.index.name = "timestamp"
    fc_df["actual"] = y_raw[start:].values().flatten()
    for name, r in results.items():
        fc_df[name] = r["forecast"].to_series()
    fc_path = os.path.join(run_dir, f"forecasts_{country}_{label}.csv")
    fc_df.to_csv(fc_path)
    plot_path = os.path.join(run_dir, f"benchmark_comparison_{country}_{label}.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fc_df.plot(ax=ax1, alpha=0.8)
    ax1.set_title(f"[{country}] Day-ahead walk-forward over {holdout_weeks}-week holdout")
    ax1.set_ylabel("EUR/MWh")
    ax1.grid(True, alpha=0.3)
    names = list(results.keys())
    maes = [results[n]["mae"] for n in names]
    rmses = [results[n]["rmse"] for n in names]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, maes, 0.4, label="MAE", color="skyblue")
    ax2.bar(x + 0.2, rmses, 0.4, label="RMSE", color="salmon")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    for i, (m, r_) in enumerate(zip(maes, rmses, strict=True)):
        ax2.annotate(f"{m:.1f}", (i - 0.2, m), ha="center", va="bottom", fontsize=9)
        ax2.annotate(f"{r_:.1f}", (i + 0.2, r_), ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("EUR/MWh")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close("all")

    # Data snapshot (the exact feature table behind these numbers) + latest-pointer.
    shutil.copy2(
        f"data/processed/features_{country}.csv",
        os.path.join(run_dir, f"features_{country}.csv"),
    )
    latest_dir = os.path.join("reports", "latest")
    os.makedirs(latest_dir, exist_ok=True)
    shutil.copy2(bench_path, os.path.join(latest_dir, f"benchmark_{country}_{label}.txt"))
    print(f"[{country}] run complete — all artifacts in {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared walk-forward benchmark harness.")
    parser.add_argument("--countries", type=str, default="CH",
                        help="Comma-separated country codes (default: CH).")
    parser.add_argument("--holdout-weeks", type=int, default=8)
    parser.add_argument("--val-weeks", type=int, default=3,
                        help="Early-stop validation weeks before the holdout.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--target-transform", choices=["minmax", "asinh"], default="minmax",
                        help="TFT target pipeline (baselines unaffected).")
    parser.add_argument("--quantiles", type=str, default=DEFAULT_QUANTILES,
                        help="Comma-separated TFT quantile list.")
    parser.add_argument("--skip-tft", action="store_true",
                        help="Naive + classical baselines only.")
    parser.add_argument("--load-tft", type=str, default=None,
                        help="Path to a saved TFT .pt — evaluate it without retraining.")
    parser.add_argument("--label", type=str, default=None,
                        help="Run label for artifact/model file names.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--es-min-delta", type=float, default=1e-3)
    args = parser.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",")]
    for country in [c.strip().upper() for c in args.countries.split(",") if c.strip()]:
        run_benchmark(
            country=country,
            holdout_weeks=args.holdout_weeks,
            val_weeks=args.val_weeks,
            epochs=args.epochs,
            target_transform=args.target_transform,
            quantiles=quantiles,
            skip_tft=args.skip_tft,
            load_tft=args.load_tft,
            label=args.label,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            es_patience=args.es_patience,
            es_min_delta=args.es_min_delta,
        )

    # Hard exit prevents Windows C++ heap corruption during PyTorch teardown.
    os._exit(0)
