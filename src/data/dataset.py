import os
import sys
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pickle

# Allow running as a script: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import get_country


def load_and_prepare_data(
    country: str = "CH",
    processed_data_dir: str = "data/processed",
    test_days: int = 7,
    val_days: int = 7,
):
    """Load processed features for one country and convert them to Darts TimeSeries.

    Reads ``features_{country}.csv``, builds the target + future covariates,
    splits into train/val/test, fits per-country scalers saved as
    ``models/scaler_target_{country}.pkl`` and ``models/scaler_future_{country}.pkl``.
    """
    get_country(country)  # validate
    features_path = os.path.join(processed_data_dir, f"features_{country}.csv")
    if not os.path.exists(features_path):
        print(f"Features for {country} not found. Please run build_features.py first.")
        # Attempt to run it
        import subprocess
        subprocess.run([sys.executable, "src/features/build_features.py"])

    df = pd.read_csv(features_path, parse_dates=[0], index_col=0)

    df.index = pd.to_datetime(df.index)  # type: ignore
    df.index = df.index.tz_localize(None)  # type: ignore

    # Target
    series = TimeSeries.from_series(df['price'])

    # Split Covariates
    past_cols = []
    future_cols = [c for c in df.columns if c not in past_cols + ['price']]

    future_covariates = TimeSeries.from_dataframe(df, value_cols=future_cols)

    # Split data
    test_len = test_days * 24
    val_len = val_days * 24

    train_val, test = series[:-test_len], series[-test_len:]
    train, val = train_val[:-val_len], train_val[-val_len:]

    future_train_val, future_test = future_covariates[:-test_len], future_covariates[-test_len:]
    future_train, future_val = future_train_val[:-val_len], future_train_val[-val_len:]

    # Scale data
    scaler_target = Scaler()
    scaler_future = Scaler()

    train_scaled = scaler_target.fit_transform(train)
    val_scaled = scaler_target.transform(val)
    test_scaled = scaler_target.transform(test)

    future_train_scaled = scaler_future.fit_transform(future_train)
    future_val_scaled = scaler_future.transform(future_val)
    future_test_scaled = scaler_future.transform(future_test)

    os.makedirs("models", exist_ok=True)
    with open(f"models/scaler_target_{country}.pkl", "wb") as f:
        pickle.dump(scaler_target, f)
    with open(f"models/scaler_future_{country}.pkl", "wb") as f:
        pickle.dump(scaler_future, f)

    print(
        f"[{country}] Data prepared. "
        f"Train shape: {len(train_scaled)}, Val shape: {len(val_scaled)}, Test shape: {len(test_scaled)}"
    )

    return {
        "train": train_scaled,
        "val": val_scaled,
        "test": test_scaled,
        "future_train": future_train_scaled,
        "future_val": future_val_scaled,
        "future_test": future_test_scaled,
        "scaler_target": scaler_target,
        "scaler_future": scaler_future,
    }


if __name__ == "__main__":
    from src.config import DEFAULT_COUNTRIES

    for country in DEFAULT_COUNTRIES:
        load_and_prepare_data(country=country)
