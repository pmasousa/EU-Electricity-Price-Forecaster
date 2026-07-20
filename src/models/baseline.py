import os
import sys
import torch
import warnings

warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")
torch.set_float32_matmul_precision('high')

# Add project root to sys.path if not running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import LinearRegressionModel, LightGBMModel
from darts.metrics import mae, rmse
import os

def train_and_evaluate_baselines():
    print("Loading data for baseline evaluation...")
    try:
        data = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    train = data["train"]
    val = data["val"]
    val_len = len(val)
    
    past_train = data["past_train"]
    past_val = data["past_val"]
    future_train = data["future_train"]
    future_val = data["future_val"]
    
    scaler_target = data["scaler_target"]
    actual_val = scaler_target.inverse_transform(val)
    
    baselines = {
        "Linear Regression": LinearRegressionModel(lags=168, lags_past_covariates=24, lags_future_covariates=[0]),
        "LightGBM": LightGBMModel(lags=168, lags_past_covariates=24, lags_future_covariates=[0])
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.txt", "w") as f:
        print(f"\n--- Baseline Model Evaluation ---")
        
        for name, model in baselines.items():
            print(f"Training {name} baseline...")
            model.fit(series=train, past_covariates=past_train, future_covariates=future_train)
            pred_val_scaled = model.predict(n=val_len, past_covariates=past_train.append(past_val), future_covariates=future_train.append(future_val))
            
            pred_val = scaler_target.inverse_transform(pred_val_scaled)
            mae_score = mae(actual_val, pred_val)
            rmse_score = rmse(actual_val, pred_val)
            
            print(f"Model: {name}")
            print(f"Validation MAE:  {mae_score:.2f} CHF/MWh")
            print(f"Validation RMSE: {rmse_score:.2f} CHF/MWh\n")
            
            f.write(f"{name}\n")
            f.write(f"MAE: {mae_score:.2f}\n")
            f.write(f"RMSE: {rmse_score:.2f}\n\n")

if __name__ == "__main__":
    train_and_evaluate_baselines()
