"""Assemble the serving bundle for one country from benchmarked artifacts.

Serves EXACTLY what was benchmarked: copies the TFT from a harness run and
regenerates the scalers it was trained with (deterministic: same data file,
same split recipe — minmax on the target, z-score on covariates, both fit on
the train split only). Writes ``models/serving_{country}/``:

    tft_model.pt       the benchmarked TFT (loaded by the API)
    scaler_target.pkl  darts Scaler (minmax), fit on the train split
    scaler_cov.pkl     darts Scaler (StandardScaler), fit on the train split
    config.json        split recipe, covariate columns, provenance

Usage:
    python src/models/build_serving.py --country CH --variant minmax \
        [--holdout-weeks 8 --val-weeks 3]
"""

import argparse
import datetime
import json
import os
import pickle
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

from src.data.honest import COV_COLUMNS, load_honest_frame


def build(country: str, variant: str, holdout_weeks: int, val_weeks: int,
          model_path: str | None = None) -> None:
    df = load_honest_frame(country)
    hold_n = holdout_weeks * 168
    val_n = val_weeks * 168
    train_n = len(df) - hold_n - val_n

    y = TimeSeries.from_series(df["price"])
    covs = TimeSeries.from_dataframe(df, value_cols=COV_COLUMNS)

    # Deterministic regeneration of the training scalers — fit on train only,
    # exactly as the harness did.
    scaler_target = Scaler()
    scaler_target.fit(y[:train_n])
    scaler_cov = Scaler(StandardScaler())
    scaler_cov.fit(covs[:train_n])

    src_model = model_path or f"models/tft_model_{country}_{variant}.pt"
    if not os.path.exists(src_model):
        raise SystemExit(
            f"{src_model} not found — pass --model-path pointing at the run "
            f"archive (reports/runs/{country}_{variant}/<timestamp>/) or run "
            f"the benchmark harness first."
        )

    out_dir = f"models/serving_{country}"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(src_model, os.path.join(out_dir, "tft_model.pt"))
    # darts keeps the trained weights in a companion .ckpt — without it the
    # model loads weightless and predict() refuses to run. Look next to the
    # .pt first, then in models/ (where fit() writes it).
    ckpt_candidates = [
        src_model + ".ckpt",
        os.path.join("models", os.path.basename(src_model) + ".ckpt"),
    ]
    src_ckpt = next((c for c in ckpt_candidates if os.path.exists(c)), None)
    if src_ckpt is None:
        raise SystemExit(
            f"no .ckpt found for {src_model} (tried {ckpt_candidates}) — "
            f"the checkpoint holds the weights."
        )
    shutil.copy2(src_ckpt, os.path.join(out_dir, "tft_model.pt.ckpt"))
    with open(os.path.join(out_dir, "scaler_target.pkl"), "wb") as f:
        pickle.dump(scaler_target, f)
    with open(os.path.join(out_dir, "scaler_cov.pkl"), "wb") as f:
        pickle.dump(scaler_cov, f)

    config = {
        "country": country,
        "variant": variant,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "fit_through": str(df.index[train_n - 1]),
        "data_through": str(df.index[-1]),
        "split": {"holdout_weeks": holdout_weeks, "val_weeks": val_weeks},
        "input_chunk_length": 168,
        "forecast_horizon": 24,
        "target_transform": "minmax" if variant == "minmax" else variant,
        "cov_columns": COV_COLUMNS,
        "notes": (
            "Serves the benchmarked model as-is (no post-selection retrain). "
            "Benchmark evidence: reports/runs/ and reports/latest/."
        ),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{country}] serving bundle -> {out_dir}")
    print(f"    model: {src_model} (variant '{variant}')")
    print(f"    fit through {config['fit_through']} | data through {config['data_through']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="CH",
                   help="Country code, or comma-separated list (default: CH).")
    p.add_argument("--variant", default="minmax",
                   help="Harness label of the model to serve (default: minmax = V0).")
    p.add_argument("--holdout-weeks", type=int, default=8)
    p.add_argument("--val-weeks", type=int, default=3)
    p.add_argument("--model-path", default=None,
                   help="Explicit path to a benchmarked .pt (e.g. inside "
                        "reports/runs/...). Defaults to models/tft_model_<CC>_<variant>.pt.")
    args = p.parse_args()
    for country in [c.strip().upper() for c in args.country.split(",") if c.strip()]:
        build(country, args.variant, args.holdout_weeks, args.val_weeks, args.model_path)
