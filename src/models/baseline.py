import os
import sys
import argparse
import torch
import warnings

warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")
torch.set_float32_matmul_precision('high')

# Add project root to sys.path if not running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import DEFAULT_COUNTRIES, get_country
from src.data.dataset import load_and_prepare_data
from darts.models import LinearRegressionModel, LightGBMModel
from darts.metrics import mae, rmse
import os

def train_and_evaluate_baselines(country: str = "CH"):
    """Train baselines for one country and append metrics to per-country reports."""
    get_country(country)  # validate
    print(f"[{country}] Loading data for baseline evaluation...")
    try:
        data = load_and_prepare_data(country=country)
    except Exception as e:
        print(f"[{country}] Error loading data: {e}")
        return

    train = data["train"]
    val = data["val"]
    val_len = len(val)

    future_train = data["future_train"]
    future_val = data["future_val"]

    scaler_target = data["scaler_target"]
    actual_val = scaler_target.inverse_transform(val)

    baselines = {
        "Linear Regression": LinearRegressionModel(lags=168, lags_future_covariates=[0]),
        "LightGBM": LightGBMModel(lags=168, lags_future_covariates=[0])
    }

    os.makedirs("reports", exist_ok=True)
    # Truncate the main metrics file on the first (CH) country so old runs don't pile up;
    # subsequent countries append to the same file under their own section.
    mode = "w" if country == DEFAULT_COUNTRIES[0] else "a"
    with open("reports/metrics.txt", mode) as f, open("reports/backtest_metrics.txt", mode) as bf:
        print(f"\n--- [{country}] Baseline Model Evaluation ---")
        f.write(f"=== Country: {country} ===\n")

        for name, model in baselines.items():
            print(f"[{country}] Training {name} baseline...")
            model.fit(series=train, future_covariates=future_train)
            pred_val_scaled = model.predict(n=val_len, future_covariates=future_train.append(future_val))

            pred_val = scaler_target.inverse_transform(pred_val_scaled)
            mae_score = mae(actual_val, pred_val)
            rmse_score = rmse(actual_val, pred_val)

            print(f"Model: {name}")
            print(f"Validation MAE:  {mae_score:.2f} EUR/MWh")
            print(f"Validation RMSE: {rmse_score:.2f} EUR/MWh\n")

            f.write(f"{name}\n")
            f.write(f"MAE: {mae_score:.2f}\n")
            f.write(f"RMSE: {rmse_score:.2f}\n\n")

            # --- Walk-Forward Backtest ---
            print(f"[{country}] Running historical backtest (Walk-forward) for {name}...")
            future_covs = future_train.append(future_val)
            all_series = train.append(val)

            forecasts = model.historical_forecasts(
                series=all_series,
                future_covariates=future_covs,
                start=len(train),
                forecast_horizon=24,
                stride=24,
                retrain=False,
                verbose=False
            )
            forecasts_real = scaler_target.inverse_transform(forecasts)

            backtest_mae = mae(actual_val, forecasts_real)
            backtest_rmse = rmse(actual_val, forecasts_real)

            print(f"Backtest MAE:  {backtest_mae:.2f} EUR/MWh")
            print(f"Backtest RMSE: {backtest_rmse:.2f} EUR/MWh\n")

            # Write to backtest_metrics.txt as well
            bf.write(f"=== Country: {country} ===\n")
            bf.write(f"Backtest: {name}\n")
            bf.write(f"MAE: {backtest_mae:.2f}\n")
            bf.write(f"RMSE: {backtest_rmse:.2f}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-country baseline models.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in config).")
    args = parser.parse_args()

    if args.countries:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
        for c in countries:
            get_country(c)
    else:
        countries = DEFAULT_COUNTRIES

    for country in countries:
        train_and_evaluate_baselines(country=country)
