import os
import sys

import numpy as np
import pandas as pd

# Allow running as a script: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import get_country


def create_cyclic_features(df, col_name, period):
    df[f"{col_name}_sin"] = np.sin(2 * np.pi * df[col_name] / period)
    df[f"{col_name}_cos"] = np.cos(2 * np.pi * df[col_name] / period)
    return df


def build_features(
    country: str = "CH",
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
):
    """Build the engineered feature table for one country.

    Reads the three per-country raw files (``entsoe_prices_{country}.csv``,
    ``entsoe_load_{country}.csv``, ``weather_{country}.csv``) and writes
    ``data/processed/features_{country}.csv``. The column set is identical for
    every country so the downstream model shape stays constant.
    """
    get_country(country)  # validate
    print(f"Building features for {country}...")

    # Check if files exist
    prices_path = os.path.join(raw_data_dir, f"entsoe_prices_{country}.csv")
    load_path = os.path.join(raw_data_dir, f"entsoe_load_{country}.csv")
    weather_path = os.path.join(raw_data_dir, f"weather_{country}.csv")

    if not all(os.path.exists(p) for p in [prices_path, load_path, weather_path]):
        print(f"Raw data files for {country} not found. Running download scripts...")
        import subprocess
        subprocess.run([sys.executable, "src/data/download_entsoe.py"])
        subprocess.run([sys.executable, "src/data/download_weather.py"])

    # Load data (if generation failed, this will crash, but that's expected)
    prices_df = pd.read_csv(prices_path, parse_dates=[0], index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index, utc=True)

    load_df = pd.read_csv(load_path, parse_dates=[0], index_col=0)
    load_df.index = pd.to_datetime(load_df.index, utc=True)

    weather_df = pd.read_csv(weather_path, parse_dates=['date'], index_col='date')
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)

    # Merge datasets on the hourly index
    df = prices_df.join(load_df, how="inner")
    df = df.join(weather_df, how="inner")

    # Rename columns to ensure consistency
    df.columns = [
        "price", "load", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "direct_radiation",
    ]

    # Missing values (interpolate or forward fill)
    df = df.interpolate(method='linear', limit_direction='both')

    df_index = pd.to_datetime(df.index)
    df['hour'] = df_index.hour  # type: ignore
    df['day_of_week'] = df_index.dayofweek  # type: ignore
    df['day_of_month'] = df_index.day  # type: ignore
    df['month'] = df_index.month  # type: ignore
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Cyclic encoding
    df = create_cyclic_features(df, "hour", 24)
    df = create_cyclic_features(df, "day_of_week", 7)
    df = create_cyclic_features(df, "month", 12)

    # Save to processed
    os.makedirs(processed_data_dir, exist_ok=True)
    output_path = os.path.join(processed_data_dir, f"features_{country}.csv")
    df.to_csv(output_path)

    print(f"Features built and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    import argparse

    from src.config import parse_countries

    parser = argparse.ArgumentParser(description="Build per-country feature tables.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in config).")
    args = parser.parse_args()
    countries = parse_countries(args.countries)

    for country in countries:
        build_features(country=country)
