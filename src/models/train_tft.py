import os
import sys
import torch
import warnings

# Suppress PyTorch Lightning pytree and Tensor Core warnings robustly
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")

# Enable Tensor Cores for massive speedup on RTX 50-series
torch.set_float32_matmul_precision('high')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import TFTModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime

class GlobalTimerCallback(Callback):
    def __init__(self):
        self.start_time = None
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        epochs_completed = trainer.current_epoch + 1
        total_epochs = trainer.max_epochs
        
        avg_time_per_epoch = elapsed / epochs_completed
        remaining_epochs = total_epochs - epochs_completed
        eta_seconds = remaining_epochs * avg_time_per_epoch
        
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        elapsed_string = str(datetime.timedelta(seconds=int(elapsed)))
        
        print(f"\n[Global Timer] Epoch {epochs_completed}/{total_epochs} | Elapsed: {elapsed_string} | ETA: {eta_string}")

def train_tft():
    print("Loading data for TFT model...")
    try:
        data = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    train = data["train"]
    val = data["val"]
    past_train = data["past_train"]
    past_val = data["past_val"]
    future_train = data["future_train"]
    future_val = data["future_val"]
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
        batch_size=1024,
        n_epochs=30,
        add_relative_index=False,
        random_state=42,
        pl_trainer_kwargs={
            "logger": CSVLogger("reports/logs", name="tft_logs"),
            "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else "auto",
            "callbacks": [GlobalTimerCallback()]
        }
    )
    
    print("Training TFT model...")
    tft.fit(
        series=train,
        past_covariates=past_train,
        future_covariates=future_train,
        val_series=val,
        val_past_covariates=past_val,
        val_future_covariates=future_val,
        verbose=True
    )
    
    # Evaluate
    print("Evaluating TFT model...")
    past_combined = past_train.append(past_val)
    future_combined = future_train.append(future_val)
    pred_val_scaled = tft.predict(n=len(val), past_covariates=past_combined, future_covariates=future_combined, series=train, show_warnings=False)
    
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
    with open("reports/metrics.txt", "a") as f:
        f.write(f"TFT Model (Single-Shot Week-Ahead)\n")
        f.write(f"MAE: {mae_score:.2f}\n")
        f.write(f"RMSE: {rmse_score:.2f}\n\n")

if __name__ == "__main__":
    train_tft()
