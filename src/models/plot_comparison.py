import os
import sys
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

from src.data.dataset import load_and_prepare_data
from darts.models import NaiveSeasonal, TFTModel
from darts.metrics import mae, rmse
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

def generate_comparison_plot():
    print("Loading data...")
    data = load_and_prepare_data()
    train = data["train"]
    val = data["val"]
    past_train = data["past_train"]
    past_val = data["past_val"]
    future_train = data["future_train"]
    future_val = data["future_val"]
    scaler_target = data["scaler_target"]
    
    # 1. Baselines
    print("Running Baselines...")
    from darts.models import LinearRegressionModel, LightGBMModel
    
    baseline_lr = LinearRegressionModel(lags=168, lags_past_covariates=24, lags_future_covariates=[0])
    baseline_lr.fit(series=train, past_covariates=past_train, future_covariates=future_train)
    pred_val_lr = baseline_lr.predict(n=len(val), past_covariates=past_train.append(past_val), future_covariates=future_train.append(future_val))
    pred_val_lr_real = scaler_target.inverse_transform(pred_val_lr)
    
    baseline_lgbm = LightGBMModel(lags=168, lags_past_covariates=24, lags_future_covariates=[0])
    baseline_lgbm.fit(series=train, past_covariates=past_train, future_covariates=future_train)
    pred_val_lgbm = baseline_lgbm.predict(n=len(val), past_covariates=past_train.append(past_val), future_covariates=future_train.append(future_val))
    pred_val_lgbm_real = scaler_target.inverse_transform(pred_val_lgbm)
    
    # 2. TFT Model
    print("Running TFT...")
    tft_path = "models/tft_model.pt"
    if os.path.exists(tft_path):
        tft = TFTModel.load(tft_path)
    else:
        print("TFT Model not found. Training a quick one...")
        tft = TFTModel(
            input_chunk_length=24*3,
            output_chunk_length=24,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=1024,
            n_epochs=2,
            add_relative_index=False,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
                "devices": [0] if torch.cuda.is_available() else "auto"
            }
        )
        tft.fit(series=train, past_covariates=past_train, future_covariates=future_train, val_series=val, val_past_covariates=past_val, val_future_covariates=future_val)
        os.makedirs("models", exist_ok=True)
        tft.save(tft_path)
        
    past_combined = past_train.append(past_val)
    future_combined = future_train.append(future_val)
    pred_val_tft = tft.predict(n=len(val), past_covariates=past_combined, future_covariates=future_combined, series=train, show_warnings=False)
    pred_val_tft_real = scaler_target.inverse_transform(pred_val_tft)
    
    val_real = scaler_target.inverse_transform(val)
    
    # Calculate Single-Shot Metrics
    mae_lr = mae(val_real, pred_val_lr_real)
    rmse_lr = rmse(val_real, pred_val_lr_real)
    mae_lgbm = mae(val_real, pred_val_lgbm_real)
    rmse_lgbm = rmse(val_real, pred_val_lgbm_real)
    mae_tft = mae(val_real, pred_val_tft_real)
    rmse_tft = rmse(val_real, pred_val_tft_real)
    
    import glob
    
    # ---------------- Plot 1: Forecast ----------------
    print("Generating Forecast Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot only the last 3 days of train for context + validation
    history_len = 24 * 3
    train_real = scaler_target.inverse_transform(train[-history_len:])
    
    val_start = val_real.time_index[0]
    val_end = val_real.time_index[-1]
    
    # Subplot 1: LR vs TFT
    ax1.plot(train_real.time_index, train_real.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax1.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax1.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax1.plot(pred_val_lr_real.time_index, pred_val_lr_real.values(), label="Linear Regression", color="orange", linestyle="--", linewidth=1.5)
    ax1.plot(pred_val_tft_real.time_index, pred_val_tft_real.values(), label="TFT Forecast", color="blue", linestyle="--", linewidth=2.0)
    ax1.set_title(f"Week-Ahead Forecast: Linear Regression (MAE: {mae_lr:.2f}) vs TFT (MAE: {mae_tft:.2f})", fontsize=14)
    ax1.set_ylabel("Price (CHF/MWh)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: LightGBM vs TFT
    ax2.plot(train_real.time_index, train_real.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax2.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax2.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax2.plot(pred_val_lgbm_real.time_index, pred_val_lgbm_real.values(), label="LightGBM", color="purple", linestyle="--", linewidth=1.5)
    ax2.plot(pred_val_tft_real.time_index, pred_val_tft_real.values(), label="TFT Forecast", color="blue", linestyle="--", linewidth=2.0)
    ax2.set_title(f"Week-Ahead Forecast: LightGBM (MAE: {mae_lgbm:.2f}) vs TFT (MAE: {mae_tft:.2f})", fontsize=14)
    ax2.set_ylabel("Price (CHF/MWh)", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    os.makedirs("reports", exist_ok=True)
    plt.tight_layout()
    plt.savefig("reports/forecast_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Forecast plot saved to reports/forecast_comparison.png")
    
    # ---------------- Plot 2: Learning Curve ----------------
    print("Generating Learning Curve Plot...")
    plt.figure(figsize=(10, 5))
    
    # Find latest metrics.csv
    try:
        log_dirs = glob.glob("reports/logs/tft_logs/version_*")
        if log_dirs:
            # Sort by version number
            latest_log_dir = sorted(log_dirs, key=lambda x: int(x.split('version_')[-1]))[-1]
            metrics_path = os.path.join(latest_log_dir, "metrics.csv")
            
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                if 'train_loss' in metrics_df.columns:
                    train_loss = metrics_df[['epoch', 'train_loss']].dropna().groupby('epoch').mean()
                    plt.plot(train_loss.index, train_loss['train_loss'], label="Train Loss", color="blue", linewidth=2)
                if 'val_loss' in metrics_df.columns:
                    val_loss = metrics_df[['epoch', 'val_loss']].dropna().groupby('epoch').mean()
                    plt.plot(val_loss.index, val_loss['val_loss'], label="Validation Loss", color="orange", linewidth=2)
                plt.title("TFT Learning Curve (Loss per Epoch)", fontsize=14)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Loss", fontsize=12)
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
                
                plt.savefig("reports/learning_curve.png", dpi=300, bbox_inches='tight')
                print("Learning curve saved to reports/learning_curve.png")
            else:
                print("Metrics file not found. Could not generate learning curve.")
        else:
            print("No training logs found. Could not generate learning curve.")
    except Exception as e:
        print(f"Error loading learning curve: {e}")
    finally:
        plt.close()
    
    # ---------------- Plot 3: Rolling Day-Ahead Forecast ----------------
    print("Generating Rolling Day-Ahead Forecast...")
    rolling_forecasts = []
    
    num_days = len(val) // 24
    for i in range(num_days):
        if i == 0:
            current_train = train
        else:
            current_train = train.append(val[:i*24])
            
        pred = tft.predict(n=24, series=current_train, past_covariates=past_combined, future_covariates=future_combined, show_warnings=False)
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
        
    print("Generating Rolling Forecast Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Subplot 1: LR vs TFT (Rolling)
    ax1.plot(train_real.time_index, train_real.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax1.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax1.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax1.plot(pred_val_lr_real.time_index, pred_val_lr_real.values(), label=f"Linear Regression (Week-Ahead MAE: {mae_lr:.2f})", color="orange", linestyle="--", linewidth=1.5)
    for idx, p in enumerate(rolling_forecasts):
        p_real = scaler_target.inverse_transform(p)
        label = f"TFT Rolling Day-Ahead (MAE: {mae_tft_rolling:.2f})" if idx == 0 else None
        ax1.plot(p_real.time_index, p_real.values(), label=label, color="green", linestyle="-", linewidth=2.0)
        if idx > 0:
            ax1.axvline(p_real.time_index[0], color='gray', linestyle=':', alpha=0.5)
    ax1.set_title("Rolling Day-Ahead: Linear Regression vs TFT", fontsize=14)
    ax1.set_ylabel("Price (CHF/MWh)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: GBM vs TFT (Rolling)
    ax2.plot(train_real.time_index, train_real.values(), label="Actual (Train)", color="black", linewidth=1.5)
    ax2.plot(val_real.time_index, val_real.values(), label="Actual (Validation)", color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax2.axvspan(val_start, val_end, color='gray', alpha=0.1, label="Validation Period")
    ax2.plot(pred_val_lgbm_real.time_index, pred_val_lgbm_real.values(), label=f"LightGBM (Week-Ahead MAE: {mae_lgbm:.2f})", color="purple", linestyle="--", linewidth=1.5)
    for idx, p in enumerate(rolling_forecasts):
        p_real = scaler_target.inverse_transform(p)
        label = f"TFT Rolling Day-Ahead (MAE: {mae_tft_rolling:.2f})" if idx == 0 else None
        ax2.plot(p_real.time_index, p_real.values(), label=label, color="green", linestyle="-", linewidth=2.0)
        if idx > 0:
            ax2.axvline(p_real.time_index[0], color='gray', linestyle=':', alpha=0.5)
    ax2.set_title("Rolling Day-Ahead: LightGBM vs TFT", fontsize=14)
    ax2.set_ylabel("Price (CHF/MWh)", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/rolling_forecast_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Rolling forecast plot saved to reports/rolling_forecast_comparison.png")
    
    # Generate Error Comparison Plot
    print("Generating Error Comparison Plot...")
    
    plt.figure(figsize=(10, 6))
    
    # We will plot single-shot (Week-Ahead) errors for all 3, and rolling Day-Ahead for TFT
    models = ['Linear Reg.', 'LightGBM', 'TFT (Week-Ahead)', 'TFT (Day-Ahead Rolling)']
    mae_scores = [mae_lr, mae_lgbm, mae_tft, mae_tft_rolling]
    rmse_scores = [rmse_lr, rmse_lgbm, rmse_tft, rmse_tft_rolling]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mae_scores, width, label='MAE', color='skyblue')
    rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE', color='salmon')
    
    ax.set_ylabel('Error (CHF/MWh)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig("reports/error_comparison.png", dpi=300)
    plt.close('all')
    print("Error comparison plot saved to reports/error_comparison.png")

if __name__ == "__main__":
    generate_comparison_plot()
