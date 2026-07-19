import os
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pickle

def load_and_prepare_data(processed_data_dir: str = "data/processed", test_days: int = 7, val_days: int = 7):
    """
    Loads processed features and converts them to Darts TimeSeries objects.
    Splits into train, validation, and test sets.
    """
    features_path = os.path.join(processed_data_dir, "features.csv")
    if not os.path.exists(features_path):
        print("Features not found. Please run build_features.py first.")
        # Attempt to run it
        import subprocess
        subprocess.run(["python", "src/features/build_features.py"])
        
    df = pd.read_csv(features_path, parse_dates=[0], index_col=0)
    
    df.index = pd.to_datetime(df.index) # type: ignore
    df.index = df.index.tz_localize(None) # type: ignore
    
    # Target
    series = TimeSeries.from_series(df['price'])
    
    # Split Covariates
    past_cols = ['load']
    future_cols = [c for c in df.columns if c not in past_cols + ['price']]
    
    past_covariates = TimeSeries.from_dataframe(df, value_cols=past_cols)
    future_covariates = TimeSeries.from_dataframe(df, value_cols=future_cols)
    
    # Split data
    test_len = test_days * 24
    val_len = val_days * 24
    
    train_val, test = series[:-test_len], series[-test_len:]
    train, val = train_val[:-val_len], train_val[-val_len:]
    
    past_train_val, past_test = past_covariates[:-test_len], past_covariates[-test_len:]
    past_train, past_val = past_train_val[:-val_len], past_train_val[-val_len:]
    
    future_train_val, future_test = future_covariates[:-test_len], future_covariates[-test_len:]
    future_train, future_val = future_train_val[:-val_len], future_train_val[-val_len:]
    
    # Scale data
    scaler_target = Scaler()
    scaler_past = Scaler()
    scaler_future = Scaler()
    
    train_scaled = scaler_target.fit_transform(train)
    val_scaled = scaler_target.transform(val)
    test_scaled = scaler_target.transform(test)
    
    past_train_scaled = scaler_past.fit_transform(past_train)
    past_val_scaled = scaler_past.transform(past_val)
    past_test_scaled = scaler_past.transform(past_test)
    
    future_train_scaled = scaler_future.fit_transform(future_train)
    future_val_scaled = scaler_future.transform(future_val)
    future_test_scaled = scaler_future.transform(future_test)
    
    # Save scalers for inference later
    os.makedirs("models", exist_ok=True)
    with open("models/scaler_target.pkl", "wb") as f:
        pickle.dump(scaler_target, f)
    with open("models/scaler_past.pkl", "wb") as f:
        pickle.dump(scaler_past, f)
    with open("models/scaler_future.pkl", "wb") as f:
        pickle.dump(scaler_future, f)
        
    print(f"Data prepared. Train shape: {len(train_scaled)}, Val shape: {len(val_scaled)}, Test shape: {len(test_scaled)}")
    
    return {
        "train": train_scaled,
        "val": val_scaled,
        "test": test_scaled,
        "past_train": past_train_scaled,
        "past_val": past_val_scaled,
        "past_test": past_test_scaled,
        "future_train": future_train_scaled,
        "future_val": future_val_scaled,
        "future_test": future_test_scaled,
        "scaler_target": scaler_target,
        "scaler_past": scaler_past,
        "scaler_future": scaler_future,
    }

if __name__ == "__main__":
    load_and_prepare_data()
