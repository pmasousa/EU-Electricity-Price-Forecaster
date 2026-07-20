from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
import uvicorn
import datetime
import pandas as pd
import pickle
import os
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models.torch import QuantileRegression
import torch
from pytorch_lightning.callbacks import Callback
from contextlib import asynccontextmanager

# Define dummy callback to allow pickle to load the model
class GlobalTimerCallback(Callback):
    pass

# Global variables to hold model and scalers
model = None
scaler_target = None
scaler_past = None
scaler_future = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler_target, scaler_past, scaler_future
    try:
        # PyTorch 2.6 security fix for unpickling weights
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([QuantileRegression])
            
        if os.path.exists("models/tft_model.pt"):
            model = TFTModel.load("models/tft_model.pt", map_location="cpu", weights_only=False)
            if hasattr(model, 'trainer_params'):
                model.trainer_params['accelerator'] = 'cpu'
                model.trainer_params['devices'] = 1
            
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
    yield
    # Clean up (if any) goes here

app = FastAPI(title="Swiss Electricity Price Forecaster", lifespan=lifespan)

# Handle CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return {"message": "Swiss Electricity Price Forecaster API is running"}

def create_cyclic_features(df, col_name, period):
    df[f"{col_name}_sin"] = np.sin(2 * np.pi * df[col_name] / period)
    df[f"{col_name}_cos"] = np.cos(2 * np.pi * df[col_name] / period)
    return df

@app.get("/predict")
def predict_next_day(target_date: Optional[str] = None):
    global model, scaler_target, scaler_past, scaler_future
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please wait for training to finish.")
        
    try:
        # Load the latest data
        features_path = "data/processed/features.csv"
        df = pd.read_csv(features_path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # We need 72 hours of history
        history_len = 72
        forecast_len = 24
        
        df_actual = pd.DataFrame()
        
        if target_date:
            try:
                target_start = pd.to_datetime(target_date).normalize()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
                
            if target_start <= df.index[0] + pd.Timedelta(hours=history_len):
                raise HTTPException(status_code=400, detail="Date too early, not enough historical data.")
                
            history_end = target_start - pd.Timedelta(hours=1)
            df_history = df.loc[:history_end].iloc[-history_len:].copy()
            
            target_end = target_start + pd.Timedelta(hours=forecast_len - 1)
            df_actual = df.loc[target_start:target_end].copy()
            actual_prices = {ts.isoformat(): float(v) for ts, v in df_actual['price'].items()} if not df_actual.empty else {}
        else:
            df_history = df.iloc[-history_len:].copy()
            actual_prices = {}
        
        # Create future 24 hours of covariates
        last_date = df_history.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_len, freq='h')
        
        df_future_24 = pd.DataFrame(index=future_dates)
        
        if target_date and not df_actual.empty and len(df_actual) == forecast_len:
            # Use actual weather if available for historical comparison
            df_future_24['temperature_2m'] = df_actual['temperature_2m'].values
            df_future_24['relative_humidity_2m'] = df_actual['relative_humidity_2m'].values
            df_future_24['wind_speed_10m'] = df_actual['wind_speed_10m'].values
            df_future_24['direct_radiation'] = df_actual['direct_radiation'].values
        else:
            # Naive forecast for weather (copy yesterday's weather)
            yesterday_weather = df_history.iloc[-forecast_len:][['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'direct_radiation']]
            df_future_24['temperature_2m'] = yesterday_weather['temperature_2m'].values
            df_future_24['relative_humidity_2m'] = yesterday_weather['relative_humidity_2m'].values
            df_future_24['wind_speed_10m'] = yesterday_weather['wind_speed_10m'].values
            df_future_24['direct_radiation'] = yesterday_weather['direct_radiation'].values
        
        # Recompute calendar features for future dates
        df_future_24['hour'] = df_future_24.index.hour
        df_future_24['day_of_week'] = df_future_24.index.dayofweek
        df_future_24['day_of_month'] = df_future_24.index.day
        df_future_24['month'] = df_future_24.index.month
        df_future_24['is_weekend'] = df_future_24['day_of_week'].isin([5, 6]).astype(int)
        
        df_future_24 = create_cyclic_features(df_future_24, "hour", 24)
        df_future_24 = create_cyclic_features(df_future_24, "day_of_week", 7)
        df_future_24 = create_cyclic_features(df_future_24, "month", 12)
        
        # Drop raw cyclic inputs to match training
        # Wait, the training just added _sin and _cos but kept the raw cols or dropped them?
        # In build_features.py it didn't drop the raw columns. We keep them.
        
        # Combine past and future for the TFT model (which requires history + forecast horizon for future_covariates)
        df_future_full = pd.concat([df_history.drop(columns=['price', 'load']), df_future_24])
        
        series = TimeSeries.from_series(df_history['price'])
        
        past_cols = ['load']
        past_covs = TimeSeries.from_dataframe(df_history, value_cols=past_cols)
        
        future_cols = [c for c in df_history.columns if c not in past_cols + ['price']]
        future_covs = TimeSeries.from_dataframe(df_future_full, value_cols=future_cols)
        
        # Scale
        series_scaled = scaler_target.transform(series)
        past_scaled = scaler_past.transform(past_covs)
        future_scaled = scaler_future.transform(future_covs)
        
        # Predict 24 hours into the future using probabilistic sampling
        pred_scaled = model.predict(
            n=24,
            series=series_scaled,
            past_covariates=past_scaled,
            future_covariates=future_scaled,
            num_samples=100
        )
        
        # Inverse transform
        pred_real = scaler_target.inverse_transform(pred_scaled)
        
        # Extract quantiles
        quantiles_df = pred_real.quantiles_df((0.1, 0.5, 0.9))
        
        # Format the response
        results = []
        for ts, row in quantiles_df.iterrows():
            ts_iso = ts.isoformat()
            res = {
                "timestamp": ts_iso,
                "predicted_price_chf_mwh": float(row.iloc[1]),  # 0.5 quantile (median)
                "q10": float(row.iloc[0]),                     # 0.1 quantile
                "q90": float(row.iloc[2])                      # 0.9 quantile
            }
            if ts_iso in actual_prices:
                res["actual_price_chf_mwh"] = actual_prices[ts_iso]
            results.append(res)
        
        return {"forecast": results}
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
