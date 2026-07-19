import os
import sys
import torch
import warnings

warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")
torch.set_float32_matmul_precision('high')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import NaiveSeasonal
from darts.metrics import mae, rmse

def backtest_models():
    print("Loading data for backtesting...")
    try:
        data = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    train = data["train"]
    val = data["val"]
    scaler_target = data["scaler_target"]
    
    # Let's backtest the NaiveSeasonal model
    model = NaiveSeasonal(K=24)
    model.fit(train)
    
    print("Running historical backtest (Walk-forward validation)...")
    
    # We will backtest on the validation set
    # Using start=0.5 to start halfway through the validation set
    historical_forecasts = model.historical_forecasts(
        series=val,
        start=0.5,
        forecast_horizon=24,
        stride=24, # Move forward by 24 hours at a time
        retrain=True,
        verbose=True
    )
    
    # Inverse transform
    historical_forecasts_real = scaler_target.inverse_transform(historical_forecasts)
    val_real = scaler_target.inverse_transform(val)
    
    # Metrics
    mae_score = mae(val_real, historical_forecasts_real)
    rmse_score = rmse(val_real, historical_forecasts_real)
    
    print(f"\n--- Backtesting Results (Naive Seasonal K=24) ---")
    print(f"Backtest MAE:  {mae_score:.2f} CHF/MWh")
    print(f"Backtest RMSE: {rmse_score:.2f} CHF/MWh")
    
    # Save backtest results
    os.makedirs("reports", exist_ok=True)
    with open("reports/backtest_metrics.txt", "w") as f:
        f.write(f"Backtest: Naive Seasonal (K=24)\n")
        f.write(f"MAE: {mae_score:.2f}\n")
        f.write(f"RMSE: {rmse_score:.2f}\n")

if __name__ == "__main__":
    backtest_models()
