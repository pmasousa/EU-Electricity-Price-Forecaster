import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import TFTModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler

def train_tft():
    print("Loading data for TFT model...")
    try:
        data = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    train = data["train"]
    val = data["val"]
    cov_train = data["cov_train"]
    cov_val = data["cov_val"]
    scaler_target = data["scaler_target"]
    
    # TFT hyperparameters
    input_chunk_length = 24 * 3  # Look back 3 days to fit the short validation set
    output_chunk_length = 24     # Predict next 24 hours
    
    tft = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=32,
        n_epochs=2,  # Keep low for faster demonstration
        add_relative_index=True,
        random_state=42
    )
    
    print("Training TFT model...")
    tft.fit(
        series=train,
        past_covariates=cov_train,
        val_series=val,
        val_past_covariates=cov_val,
        verbose=True
    )
    
    # Evaluate
    print("Evaluating TFT model...")
    cov_combined = cov_train.append(cov_val)
    pred_val_scaled = tft.predict(n=len(val), past_covariates=cov_combined, series=train)
    
    # Inverse transform
    pred_val = scaler_target.inverse_transform(pred_val_scaled)
    actual_val = scaler_target.inverse_transform(val)
    
    # Metrics
    mae_score = mae(actual_val, pred_val)
    rmse_score = rmse(actual_val, pred_val)
    
    print(f"\n--- TFT Model Evaluation ---")
    print(f"Validation MAE:  {mae_score:.2f} CHF/MWh")
    print(f"Validation RMSE: {rmse_score:.2f} CHF/MWh")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    tft.save("models/tft_model.pt")
    print("TFT model saved to models/tft_model.pt")
    
    # Save metrics
    os.makedirs("reports", exist_ok=True)
    with open("reports/tft_metrics.txt", "w") as f:
        f.write(f"TFT Model (epochs=2)\n")
        f.write(f"MAE: {mae_score:.2f}\n")
        f.write(f"RMSE: {rmse_score:.2f}\n")

if __name__ == "__main__":
    train_tft()
