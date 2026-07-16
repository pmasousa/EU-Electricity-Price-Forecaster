from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
import datetime

app = FastAPI(title="Swiss Electricity Price Forecaster")

# Handle CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since we need to avoid CORS problems
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Swiss Electricity Price Forecaster API is running"}

@app.get("/predict")
def predict_next_day():
    """
    Simulated endpoint returning 24h predictions.
    In a real scenario, this would load the Darts model, fetch latest data, and run inference.
    """
    # Dummy prediction using a sine wave + noise for demo purposes
    hours = np.arange(24)
    base_price = 100.0
    seasonality = np.sin(2 * np.pi * hours / 24) * 30
    noise = np.random.normal(0, 5, 24)
    predictions = base_price + seasonality + noise
    
    now = datetime.datetime.now()
    start_time = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    
    results = [
        {
            "timestamp": (start_time + datetime.timedelta(hours=int(i))).isoformat(),
            "predicted_price_chf_mwh": float(pred)
        }
        for i, pred in enumerate(predictions)
    ]
    
    return {"forecast": results}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
