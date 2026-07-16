import os
import sys

# Add project root to sys.path if not running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import NaiveSeasonal
from darts.metrics import mae, rmse

def train_and_evaluate_baselines():
    print("Loading data for baseline evaluation...")
    try:
        data = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    train = data["train"]
    val = data["val"]
    scaler_target = data["scaler_target"]
    
    # We will use NaiveSeasonal with K=24 (daily seasonality) since the data is hourly
    print("Training NaiveSeasonal baseline (K=24)...")
    seasonal_model = NaiveSeasonal(K=24)
    seasonal_model.fit(train)
    
    # Predict the validation period length
    val_len = len(val)
    pred_val_scaled = seasonal_model.predict(n=val_len)
    
    # Inverse transform to get real values
    pred_val = scaler_target.inverse_transform(pred_val_scaled)
    actual_val = scaler_target.inverse_transform(val)
    
    # Calculate metrics
    mae_score = mae(actual_val, pred_val)
    rmse_score = rmse(actual_val, pred_val)
    
    print(f"\n--- Baseline Model Evaluation ---")
    print(f"Model: Naive Seasonal (K=24)")
    print(f"Validation MAE:  {mae_score:.2f} CHF/MWh")
    print(f"Validation RMSE: {rmse_score:.2f} CHF/MWh")
    
    # Save the results to a file for reporting
    os.makedirs("reports", exist_ok=True)
    with open("reports/baseline_metrics.txt", "w") as f:
        f.write(f"Naive Seasonal (K=24)\n")
        f.write(f"MAE: {mae_score:.2f}\n")
        f.write(f"RMSE: {rmse_score:.2f}\n")
        
if __name__ == "__main__":
    train_and_evaluate_baselines()
