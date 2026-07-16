import os
import pandas as pd
import numpy as np

def create_cyclic_features(df, col_name, period):
    df[f"{col_name}_sin"] = np.sin(2 * np.pi * df[col_name] / period)
    df[f"{col_name}_cos"] = np.cos(2 * np.pi * df[col_name] / period)
    return df

def build_features(raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
    print("Building features...")
    
    # Check if files exist
    prices_path = os.path.join(raw_data_dir, "entsoe_prices.csv")
    load_path = os.path.join(raw_data_dir, "entsoe_load.csv")
    weather_path = os.path.join(raw_data_dir, "weather_zurich.csv")
    
    if not all(os.path.exists(p) for p in [prices_path, load_path, weather_path]):
        print("Raw data files not found. Please run data downloading scripts first.")
        # Try to run the data downloading scripts if they don't exist
        print("Generating mock data by running scripts...")
        import subprocess
        subprocess.run(["python", "src/data/download_entsoe.py"])
        subprocess.run(["python", "src/data/download_weather.py"])

    # Load data (if generation failed, this will crash, but that's expected)
    prices_df = pd.read_csv(prices_path, parse_dates=[0], index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index, utc=True)
    
    load_df = pd.read_csv(load_path, parse_dates=[0], index_col=0)
    load_df.index = pd.to_datetime(load_df.index, utc=True)
    
    weather_df = pd.read_csv(weather_path, parse_dates=['date'], index_col='date')
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)
    
    # Merge datasets
    df = prices_df.join(load_df, how="inner")
    df = df.join(weather_df, how="inner")
    
    # Rename columns to ensure consistency
    df.columns = ["price", "load", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "direct_radiation"]
    
    # Missing values (interpolate or forward fill)
    df = df.interpolate(method='linear', limit_direction='both')
    
    df_index = pd.to_datetime(df.index)
    df['hour'] = df_index.hour # type: ignore
    df['day_of_week'] = df_index.dayofweek # type: ignore
    df['day_of_month'] = df_index.day # type: ignore
    df['month'] = df_index.month # type: ignore
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclic encoding
    df = create_cyclic_features(df, "hour", 24)
    df = create_cyclic_features(df, "day_of_week", 7)
    df = create_cyclic_features(df, "month", 12)
    
    # Save to processed
    os.makedirs(processed_data_dir, exist_ok=True)
    output_path = os.path.join(processed_data_dir, "features.csv")
    df.to_csv(output_path)
    
    print(f"Features built and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    build_features()
