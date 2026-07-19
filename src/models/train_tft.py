import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import load_and_prepare_data
from darts.models import TFTModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import pandas as pd

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
        batch_size=32,
        n_epochs=30,  # Increased for proper training!
        add_relative_index=False,
        random_state=42,
        pl_trainer_kwargs={"logger": CSVLogger("reports/logs", name="tft_logs")}
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
    
    # Plot learning curve
    print("Plotting learning curve...")
    try:
        metrics_df = pd.read_csv(f"{tft.trainer.logger.experiment.metrics_file_path}")
        plt.figure(figsize=(10, 5))
        if 'train_loss' in metrics_df.columns:
            train_loss = metrics_df[['epoch', 'train_loss']].dropna().groupby('epoch').mean()
            plt.plot(train_loss.index, train_loss['train_loss'], label="Train Loss")
        if 'val_loss' in metrics_df.columns:
            val_loss = metrics_df[['epoch', 'val_loss']].dropna().groupby('epoch').mean()
            plt.plot(val_loss.index, val_loss['val_loss'], label="Validation Loss")
        plt.title("TFT Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("reports/learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not plot learning curve: {e}")
        
    # Evaluate
    print("Evaluating TFT model...")
    past_combined = past_train.append(past_val)
    future_combined = future_train.append(future_val)
    pred_val_scaled = tft.predict(n=len(val), past_covariates=past_combined, future_covariates=future_combined, series=train)
    
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
        f.write(f"TFT Model (epochs={tft.n_epochs})\n")
        f.write(f"MAE: {mae_score:.2f}\n")
        f.write(f"RMSE: {rmse_score:.2f}\n")

if __name__ == "__main__":
    train_tft()
