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
    
    # Convert index to timezone naive for Darts compatibility
    df.index = df.index.tz_localize(None)
    
    # Target
    series = TimeSeries.from_series(df['price'])
    
    # Past Covariates (features)
    covariates_cols = [c for c in df.columns if c != 'price']
    covariates = TimeSeries.from_dataframe(df, value_cols=covariates_cols)
    
    # Split data
    test_len = test_days * 24
    val_len = val_days * 24
    
    train_val, test = series[:-test_len], series[-test_len:]
    train, val = train_val[:-val_len], train_val[-val_len:]
    
    cov_train_val, cov_test = covariates[:-test_len], covariates[-test_len:]
    cov_train, cov_val = cov_train_val[:-val_len], cov_train_val[-val_len:]
    
    # Scale data
    scaler_target = Scaler()
    scaler_covs = Scaler()
    
    train_scaled = scaler_target.fit_transform(train)
    val_scaled = scaler_target.transform(val)
    test_scaled = scaler_target.transform(test)
    
    cov_train_scaled = scaler_covs.fit_transform(cov_train)
    cov_val_scaled = scaler_covs.transform(cov_val)
    cov_test_scaled = scaler_covs.transform(cov_test)
    
    # Save scalers for inference later
    os.makedirs("models", exist_ok=True)
    with open("models/scaler_target.pkl", "wb") as f:
        pickle.dump(scaler_target, f)
    with open("models/scaler_covs.pkl", "wb") as f:
        pickle.dump(scaler_covs, f)
        
    print(f"Data prepared. Train shape: {len(train_scaled)}, Val shape: {len(val_scaled)}, Test shape: {len(test_scaled)}")
    
    return {
        "train": train_scaled,
        "val": val_scaled,
        "test": test_scaled,
        "cov_train": cov_train_scaled,
        "cov_val": cov_val_scaled,
        "cov_test": cov_test_scaled,
        "scaler_target": scaler_target,
        "scaler_covs": scaler_covs,
    }

if __name__ == "__main__":
    load_and_prepare_data()
