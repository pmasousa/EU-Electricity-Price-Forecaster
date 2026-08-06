import os
import sys
import argparse
import torch
import warnings

# Suppress PyTorch Lightning pytree and Tensor Core warnings robustly
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
warnings.filterwarnings("ignore", module="pytorch_lightning.*")

# Enable Tensor Cores for massive speedup on RTX 50-series
torch.set_float32_matmul_precision('high')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import DEFAULT_COUNTRIES, get_country
from src.data.dataset import load_and_prepare_data
from darts.models import TFTModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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

def train_tft(country: str = "CH", epochs: int = 100):
    """Train a per-country TFT model and save it to ``models/tft_model_{country}.pt``.

    ``epochs`` overrides the default 100 — pass a small value (e.g. 3) for a fast
    smoke test that exercises the full train/eval/save path without waiting.
    """
    get_country(country)  # validate
    print(f"[{country}] Loading data for TFT model (epochs={epochs})...")
    try:
        data = load_and_prepare_data(country=country)
    except Exception as e:
        print(f"[{country}] Error loading data: {e}")
        return

    train = data["train"]
    val = data["val"]
    future_train = data["future_train"]
    future_val = data["future_val"]
    scaler_target = data["scaler_target"]

    # TFT hyperparameters
    input_chunk_length = 24 * 7  # Look back 7 days to learn weekly seasonality
    output_chunk_length = 24     # Predict next 24 hours

    tft = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=8,
        dropout=0.3,
        batch_size=128,
        n_epochs=epochs,
        add_relative_index=True,
        random_state=42,
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={"patience": 4, "factor": 0.5},
        pl_trainer_kwargs={
            "logger": CSVLogger("reports/logs", name=f"tft_logs_{country}"),
            "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else "auto",
            "callbacks": [
                GlobalTimerCallback(),
                EarlyStopping(monitor="val_loss", patience=15, min_delta=0.001, mode="min")
            ]
        }
    )

    print(f"[{country}] Training TFT model...")
    tft.fit(
        series=train,
        future_covariates=future_train,
        val_series=train[-168:].append(val),
        val_future_covariates=future_train[-168:].append(future_val),
        verbose=True
    )

    # Evaluate
    print(f"[{country}] Evaluating TFT model...")
    pred_val_scaled = tft.predict(n=len(val), future_covariates=future_train.append(future_val), series=train, show_warnings=False)

    # Inverse transform
    pred_val = scaler_target.inverse_transform(pred_val_scaled)
    actual_val = scaler_target.inverse_transform(val)

    # Metrics
    mae_score = mae(actual_val, pred_val)
    rmse_score = rmse(actual_val, pred_val)

    print(f"\n--- [{country}] TFT Model Evaluation ---")
    print(f"Validation MAE:  {mae_score:.2f} EUR/MWh")
    print(f"Validation RMSE: {rmse_score:.2f} EUR/MWh")

    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/tft_model_{country}.pt"
    tft.save(model_path)
    print(f"[{country}] TFT model saved to {model_path}")

    # Save metrics (per-country section)
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.txt", "a") as f:
        f.write(f"=== Country: {country} ===\n")
        f.write(f"TFT Model (Single-Shot Week-Ahead)\n")
        f.write(f"MAE: {mae_score:.2f}\n")
        f.write(f"RMSE: {rmse_score:.2f}\n\n")

    print(f"[{country}] Script finished successfully. Cleaning up PyTorch resources...")

    # Clean up to avoid teardown crashes
    # Hard exit prevents Windows C++ heap corruption during PyTorch multi-processing teardown
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-country TFT models.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in config).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default: 100). Use a small value "
                             "(e.g. 3) for a fast smoke test.")
    args = parser.parse_args()

    if args.countries:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
        for c in countries:
            get_country(c)  # validate
    else:
        countries = DEFAULT_COUNTRIES

    for country in countries:
        train_tft(country=country, epochs=args.epochs)
