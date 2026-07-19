from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
import datetime
import pandas as pd
import pickle
import os
from darts import TimeSeries
from darts.models import TFTModel

app = FastAPI(title="Swiss Electricity Price Forecaster")

# Handle CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model and scalers
model = None
scaler_target = None
scaler_past = None
scaler_future = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler_target, scaler_past, scaler_future
    try:
        if os.path.exists("models/tft_model.pt"):
            model = TFTModel.load("models/tft_model.pt")
            
            with open("models/scaler_target.pkl", "rb") as f:
                scaler_target = pickle.load(f)
            with open("models/scaler_past.pkl", "rb") as f:
                scaler_past = pickle.load(f)
            with open("models/scaler_future.pkl", "rb") as f:
                scaler_future = pickle.load(f)
            print("Successfully loaded TFT model and scalers.")
        else:
            print("Warning: TFT model not found. API will fail unless pipeline is run first.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def read_root():
    return {"message": "Swiss Electricity Price Forecaster API is running"}

@app.get("/predict")
def predict_next_day():
    global model, scaler_target, scaler_past, scaler_future
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please wait for training to finish.")
        
    try:
        # Load the latest data
        features_path = "data/processed/features.csv"
        df = pd.read_csv(features_path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # We need 72 hours of history, and 24 hours of future covariates
        # Total rows needed = 96 from the end of the file
        total_len = 96
        df_recent = df.iloc[-total_len:]
        
        # Split into historical context (first 72) and forecast horizon (last 24)
        df_history = df_recent.iloc[:-24]
        df_future = df_recent # all 96 rows for future_covariates
        
        # Target and Past Covariates ONLY use the historical 72 rows
        series = TimeSeries.from_series(df_history['price'])
        
        past_cols = ['load']
        past_covs = TimeSeries.from_dataframe(df_history, value_cols=past_cols)
        
        # Future Covariates uses all 96 rows
        future_cols = [c for c in df.columns if c not in past_cols + ['price']]
        future_covs = TimeSeries.from_dataframe(df_future, value_cols=future_cols)
        
        # Scale
        series_scaled = scaler_target.transform(series)
        past_scaled = scaler_past.transform(past_covs)
        future_scaled = scaler_future.transform(future_covs)
        
        # Predict 24 hours into the future
        pred_scaled = model.predict(
            n=24,
            series=series_scaled,
            past_covariates=past_scaled,
            future_covariates=future_scaled
        )
        
        # Inverse transform
        pred_real = scaler_target.inverse_transform(pred_scaled)
        
        # Format the response
        results = [
            {
                "timestamp": ts.isoformat(),
                "predicted_price_chf_mwh": float(val)
            }
            for ts, val in zip(pred_real.time_index, pred_real.values().flatten())
        ]
        
        return {"forecast": results}
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
