import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import NaiveSeasonal, TFTModel
from darts.metrics import mae, rmse
import pandas as pd
import numpy as np

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
    
    # 1. Baseline
    print("Running Baseline...")
    seasonal_model = NaiveSeasonal(K=24)
    seasonal_model.fit(train)
    pred_val_baseline = seasonal_model.predict(n=len(val))
    pred_val_baseline_real = scaler_target.inverse_transform(pred_val_baseline)
    
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
            batch_size=32,
            n_epochs=2,
            add_relative_index=False,
            random_state=42,
            pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}
        )
        tft.fit(series=train, past_covariates=past_train, future_covariates=future_train, val_series=val, val_past_covariates=past_val, val_future_covariates=future_val)
        os.makedirs("models", exist_ok=True)
        tft.save(tft_path)
        
    past_combined = past_train.append(past_val)
    future_combined = future_train.append(future_val)
    pred_val_tft = tft.predict(n=len(val), past_covariates=past_combined, future_covariates=future_combined, series=train)
    pred_val_tft_real = scaler_target.inverse_transform(pred_val_tft)
    
    val_real = scaler_target.inverse_transform(val)
    
    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(14, 6))
    
    # Plot only the last 5 days of train for context + validation
    history_len = 24 * 5
    train_real = scaler_target.inverse_transform(train[-history_len:])
    
    train_real.plot(label="Actual (Train)", color="black", linewidth=1.5)
    val_real.plot(label="Actual (Validation)", color="gray", linestyle="--", linewidth=1.5)
    pred_val_baseline_real.plot(label="Baseline (NaiveSeasonal)", color="orange", alpha=0.8)
    pred_val_tft_real.plot(label="TFT Forecast", color="blue", alpha=0.8)
    
    plt.title("Day-Ahead Electricity Price Forecast Comparison (CH)", fontsize=16)
    plt.ylabel("Price (CHF/MWh)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/forecast_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to reports/forecast_comparison.png")
    
    # Generate Error Comparison Plot
    print("Generating Error Comparison Plot...")
    mae_baseline = mae(val_real, pred_val_baseline_real)
    rmse_baseline = rmse(val_real, pred_val_baseline_real)
    
    mae_tft = mae(val_real, pred_val_tft_real)
    rmse_tft = rmse(val_real, pred_val_tft_real)
    
    labels = ['MAE', 'RMSE']
    baseline_scores = [mae_baseline, rmse_baseline]
    tft_scores = [mae_tft, rmse_tft]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, baseline_scores, width, label='Baseline (Naive Seasonal)', color='orange', alpha=0.8)
    ax.bar(x + width/2, tft_scores, width, label='TFT Model', color='blue', alpha=0.8)
    
    ax.set_ylabel('Error (CHF/MWh)')
    ax.set_title('Forecasting Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(baseline_scores):
        ax.text(i - width/2, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold')
    for i, v in enumerate(tft_scores):
        ax.text(i + width/2, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold')
        
    plt.savefig("reports/error_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Error comparison plot saved to reports/error_comparison.png")

if __name__ == "__main__":
    generate_comparison_plot()
