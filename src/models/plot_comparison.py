import os
import sys
import argparse
import torch
import warnings
import matplotlib.pyplot as plt

# Suppress PyTorch Lightning pytree and Tensor Core warnings robustly
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")

# Enable Tensor Cores for massive speedup
torch.set_float32_matmul_precision('high')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import DEFAULT_COUNTRIES, get_country
from src.data.dataset import load_and_prepare_data
from darts.models import NaiveSeasonal, TFTModel
from darts.metrics import mae, rmse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import numpy as np
import time
import datetime
from pytorch_lightning.callbacks import Callback

class GlobalTimerCallback(Callback):
    def __init__(self):
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pass

def generate_comparison_plot(country: str = "CH", epochs: int = 100):
    """Generate the model-comparison plots for one country.

    All outputs are suffixed with the country code (e.g.
    ``reports/forecast_comparison_{country}.png``) so multi-country runs don't
    clobber each other. Prices are labelled EUR/MWh (Energy-Charts currency).
    ``epochs`` only matters if no saved model exists and a fallback is trained.
    """
    get_country(country)  # validate
    print(f"[{country}] Loading data...")
    data = load_and_prepare_data(country=country)
    train = data["train"]
    val = data["val"]
    future_train = data["future_train"]
    future_val = data["future_val"]
    scaler_target = data["scaler_target"]
    val_real = scaler_target.inverse_transform(val)
    train_real = scaler_target.inverse_transform(train)

    # 1. Baselines
    print(f"[{country}] Running Baselines...")
    from darts.models import LinearRegressionModel, LightGBMModel

    baseline_lr = LinearRegressionModel(lags=168, lags_future_covariates=[0])
    baseline_lr.fit(series=train, future_covariates=future_train)
    pred_val_lr = baseline_lr.predict(n=len(val), future_covariates=future_train.append(future_val))
    pred_val_lr_real = scaler_target.inverse_transform(pred_val_lr)

    baseline_lgbm = LightGBMModel(lags=168, lags_future_covariates=[0])
    baseline_lgbm.fit(series=train, future_covariates=future_train)
    pred_val_lgbm = baseline_lgbm.predict(n=len(val), future_covariates=future_train.append(future_val))
    pred_val_lgbm_real = scaler_target.inverse_transform(pred_val_lgbm)

    # Run rolling backtest for baselines
    print(f"[{country}] Running Baseline Walk-Forward Backtests for plot...")
    all_series = train.append(val)
    future_covs = future_train.append(future_val)

    lr_rolling = baseline_lr.historical_forecasts(series=all_series, future_covariates=future_covs, start=len(train), forecast_horizon=24, stride=24, retrain=False, verbose=False)
    lr_rolling_real = scaler_target.inverse_transform(lr_rolling)
    mae_lr_rolling = mae(val_real, lr_rolling_real)
    rmse_lr_rolling = rmse(val_real, lr_rolling_real)

    lgbm_rolling = baseline_lgbm.historical_forecasts(series=all_series, future_covariates=future_covs, start=len(train), forecast_horizon=24, stride=24, retrain=False, verbose=False)
    lgbm_rolling_real = scaler_target.inverse_transform(lgbm_rolling)
    mae_lgbm_rolling = mae(val_real, lgbm_rolling_real)
    rmse_lgbm_rolling = rmse(val_real, lgbm_rolling_real)

    # 2. TFT Model
    print(f"[{country}] Running TFT...")
    tft_path = f"models/tft_model_{country}.pt"
    if os.path.exists(tft_path):
        tft = TFTModel.load(tft_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        tft.trainer_params["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
        tft.trainer_params["devices"] = [0] if torch.cuda.is_available() else "auto"
    else:
        print(f"[{country}] TFT Model not found. Training a quick one...")
        tft = TFTModel(
            input_chunk_length=24*7,
            output_chunk_length=24,
            hidden_size=64,
            lstm_layers=2,
            num_attention_heads=4,
            dropout=0.3,
            batch_size=128,
            n_epochs=epochs,
            add_relative_index=True,
            random_state=42,
            optimizer_kwargs={"lr": 1e-3},
            lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_kwargs={"patience": 4, "factor": 0.5},
            pl_trainer_kwargs={
                "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
                "devices": [0] if torch.cuda.is_available() else "auto",
                "callbacks": [
                    EarlyStopping(monitor="val_loss", patience=15, min_delta=0.001, mode="min")
                ]
            }
        )
        tft.fit(series=train, future_covariates=future_train, val_series=train[-168:].append(val), val_future_covariates=future_train[-168:].append(future_val))
        os.makedirs("models", exist_ok=True)
        tft.save(tft_path)

    future_combined = future_train.append(future_val)
    pred_val_tft = tft.predict(n=len(val), future_covariates=future_combined, series=train, show_warnings=False)
    pred_val_tft_real = scaler_target.inverse_transform(pred_val_tft)

    # Calculate Single-Shot Metrics
    mae_lr = mae(val_real, pred_val_lr_real)
    rmse_lr = rmse(val_real, pred_val_lr_real)
    mae_lgbm = mae(val_real, pred_val_lgbm_real)
    rmse_lgbm = rmse(val_real, pred_val_lgbm_real)
    mae_tft = mae(val_real, pred_val_tft_real)
    rmse_tft = rmse(val_real, pred_val_tft_real)

    import glob

    # ---------------- Plot 1: Forecast ----------------
    print(f"[{country}] Generating Forecast Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot only the last 3 days of train for context + validation
    history_len = 24 * 3
    train_plot = scaler_target.inverse_transform(train[-history_len:])

    val_start = val_real.time_index[0]
    val_end = val_real.time_index[-1]

    # Subplot 1: LR vs TFT
    ax1.plot(train_plot.time_index, train_plot.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax1.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax1.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax1.plot(pred_val_lr_real.time_index, pred_val_lr_real.values(), label="Linear Regression", color="orange", linestyle="--", linewidth=1.5)
    ax1.plot(pred_val_tft_real.time_index, pred_val_tft_real.values(), label="TFT Forecast", color="blue", linestyle="--", linewidth=2.0)
    ax1.set_title(f"[{country}] Week-Ahead Forecast: Linear Regression (MAE: {mae_lr:.2f}) vs TFT (MAE: {mae_tft:.2f})", fontsize=14)
    ax1.set_ylabel("Price (EUR/MWh)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: LightGBM vs TFT
    ax2.plot(train_plot.time_index, train_plot.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax2.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax2.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax2.plot(pred_val_lgbm_real.time_index, pred_val_lgbm_real.values(), label="LightGBM", color="purple", linestyle="--", linewidth=1.5)
    ax2.plot(pred_val_tft_real.time_index, pred_val_tft_real.values(), label="TFT Forecast", color="blue", linestyle="--", linewidth=2.0)
    ax2.set_title(f"[{country}] Week-Ahead Forecast: LightGBM (MAE: {mae_lgbm:.2f}) vs TFT (MAE: {mae_tft:.2f})", fontsize=14)
    ax2.set_ylabel("Price (EUR/MWh)", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    os.makedirs("reports", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"reports/forecast_comparison_{country}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{country}] Forecast plot saved.")

    # ---------------- Plot 2: Learning Curve ----------------
    print(f"[{country}] Generating Learning Curve Plot...")
    plt.figure(figsize=(10, 5))

    # Find latest metrics.csv for this country
    try:
        log_dirs = glob.glob(f"reports/logs/tft_logs_{country}/version_*")
        valid_log_dirs = [d for d in log_dirs if os.path.exists(os.path.join(d, "metrics.csv"))]
        if valid_log_dirs:
            latest_log_dir = sorted(valid_log_dirs, key=lambda x: int(x.split('version_')[-1]))[-1]
            metrics_path = os.path.join(latest_log_dir, "metrics.csv")

            metrics_df = pd.read_csv(metrics_path)
            if 'train_loss' in metrics_df.columns:
                train_loss = metrics_df[['epoch', 'train_loss']].dropna().groupby('epoch').mean()
                plt.plot(train_loss.index, train_loss['train_loss'], label="Train Loss", color="blue", linewidth=2)
            if 'val_loss' in metrics_df.columns:
                val_loss = metrics_df[['epoch', 'val_loss']].dropna().groupby('epoch').mean()
                plt.plot(val_loss.index, val_loss['val_loss'], label="Validation Loss", color="orange", linewidth=2)
            plt.title(f"[{country}] TFT Learning Curve (Loss per Epoch)", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)

            plt.savefig(f"reports/learning_curve_{country}.png", dpi=300, bbox_inches='tight')
            print(f"[{country}] Learning curve saved.")
        else:
            print(f"[{country}] No training logs with metrics.csv found. Skipping learning curve.")
    except Exception as e:
        print(f"[{country}] Error loading learning curve: {e}")
    finally:
        plt.close()

    # ---------------- Plot 3: Rolling Day-Ahead Forecast ----------------
    print(f"[{country}] Generating Rolling Day-Ahead Forecast...")
    rolling_forecasts = []

    num_days = len(val) // 24
    for i in range(num_days):
        if i == 0:
            current_train = train
        else:
            current_train = train.append(val[:i*24])

        pred = tft.predict(n=24, series=current_train, future_covariates=future_combined, show_warnings=False)
        rolling_forecasts.append(pred)

    pred_val_tft_rolling = rolling_forecasts[0]
    for p in rolling_forecasts[1:]:
        pred_val_tft_rolling = pred_val_tft_rolling.append(p)

    pred_val_tft_rolling_real = scaler_target.inverse_transform(pred_val_tft_rolling)
    mae_tft_rolling = mae(val_real, pred_val_tft_rolling_real)
    rmse_tft_rolling = rmse(val_real, pred_val_tft_rolling_real)

    with open("reports/metrics.txt", "a") as f:
        f.write(f"TFT Model (Rolling Day-Ahead)\n")
        f.write(f"MAE: {mae_tft_rolling:.2f}\n")
        f.write(f"RMSE: {rmse_tft_rolling:.2f}\n\n")

    print(f"[{country}] Generating Rolling Forecast Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(train_plot.time_index, train_plot.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax1.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax1.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax1.plot(pred_val_lr_real.time_index, pred_val_lr_real.values(), label=f"Linear Regression (Week-Ahead MAE: {mae_lr:.2f})", color="orange", linestyle="--", linewidth=1.5)
    for idx, p in enumerate(rolling_forecasts):
        p_real = scaler_target.inverse_transform(p)
        label = f"TFT Rolling Day-Ahead (MAE: {mae_tft_rolling:.2f})" if idx == 0 else None
        ax1.plot(p_real.time_index, p_real.values(), label=label, color="green", linestyle="-", linewidth=2.0)
        if idx > 0:
            ax1.axvline(p_real.time_index[0], color='gray', linestyle=':', alpha=0.5)
    ax1.set_title(f"[{country}] Rolling Day-Ahead: Linear Regression vs TFT", fontsize=14)
    ax1.set_ylabel("Price (EUR/MWh)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_plot.time_index, train_plot.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax2.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax2.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax2.plot(pred_val_lgbm_real.time_index, pred_val_lgbm_real.values(), label=f"LightGBM (Week-Ahead MAE: {mae_lgbm:.2f})", color="purple", linestyle="--", linewidth=1.5)
    for idx, p in enumerate(rolling_forecasts):
        p_real = scaler_target.inverse_transform(p)
        label = f"TFT Rolling Day-Ahead (MAE: {mae_tft_rolling:.2f})" if idx == 0 else None
        ax2.plot(p_real.time_index, p_real.values(), label=label, color="green", linestyle="-", linewidth=2.0)
        if idx > 0:
            ax2.axvline(p_real.time_index[0], color='gray', linestyle=':', alpha=0.5)
    ax2.set_title(f"[{country}] Rolling Day-Ahead: LightGBM vs TFT", fontsize=14)
    ax2.set_ylabel("Price (EUR/MWh)", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"reports/rolling_forecast_comparison_{country}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{country}] Rolling forecast plot saved.")

    # Generate Error Comparison Plots
    print(f"[{country}] Generating Error Comparison Plots...")

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    # Plot 1: Week-Ahead (Single-Split)
    models_week = ['Linear Reg.', 'LightGBM', 'TFT']
    mae_scores_week = [mae_lr, mae_lgbm, mae_tft]
    rmse_scores_week = [rmse_lr, rmse_lgbm, rmse_tft]

    x_week = np.arange(len(models_week))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x_week - width/2, mae_scores_week, width, label='MAE', color='skyblue')
    rects2 = ax.bar(x_week + width/2, rmse_scores_week, width, label='RMSE', color='salmon')

    ax.set_ylabel('Error (EUR/MWh)')
    ax.set_title(f'[{country}] Single-Shot (Week-Ahead) Error Comparison')
    ax.set_xticks(x_week)
    ax.set_xticklabels(models_week)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    plt.tight_layout()
    plt.savefig(f"reports/error_comparison_week_ahead_{country}.png", dpi=300)
    plt.close('all')

    # Plot 2: Day-Ahead (Rolling Walk-Forward)
    models_day = ['Linear Reg.', 'LightGBM', 'TFT']
    mae_scores_day = [mae_lr_rolling, mae_lgbm_rolling, mae_tft_rolling]
    rmse_scores_day = [rmse_lr_rolling, rmse_lgbm_rolling, rmse_tft_rolling]

    x_day = np.arange(len(models_day))

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x_day - width/2, mae_scores_day, width, label='MAE', color='skyblue')
    rects2 = ax.bar(x_day + width/2, rmse_scores_day, width, label='RMSE', color='salmon')

    ax.set_ylabel('Error (EUR/MWh)')
    ax.set_title(f'[{country}] Walk-Forward (Rolling Day-Ahead) Error Comparison')
    ax.set_xticks(x_day)
    ax.set_xticklabels(models_day)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    plt.tight_layout()
    plt.savefig(f"reports/error_comparison_day_ahead_{country}.png", dpi=300)
    plt.close('all')

    print(f"[{country}] Error comparison plots saved.")

    print(f"[{country}] Script finished successfully. Exiting cleanly to avoid PyTorch teardown crashes.")
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-country comparison plots.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in config).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs if a fallback TFT must be trained (default: 100).")
    args = parser.parse_args()

    if args.countries:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
        for c in countries:
            get_country(c)
    else:
        countries = DEFAULT_COUNTRIES

    for country in countries:
        generate_comparison_plot(country=country, epochs=args.epochs)
