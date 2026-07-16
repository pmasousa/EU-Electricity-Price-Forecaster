import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient

# Load environment variables
load_dotenv()

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "dummy_key")

def download_entsoe_data(start_date: str, end_date: str, output_dir: str = "data/raw"):
    """
    Downloads ENTSO-E data for Switzerland (CH) and saves to CSV.
    Uses a dummy fallback if no API key is provided.
    """
    if ENTSOE_API_KEY == "dummy_key" or not ENTSOE_API_KEY:
        print("WARNING: ENTSOE_API_KEY not found in .env. Using fallback mock data.")
        return generate_mock_entsoe_data(start_date, end_date, output_dir)
        
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    
    start = pd.Timestamp(start_date, tz='Europe/Zurich') # type: ignore
    end = pd.Timestamp(end_date, tz='Europe/Zurich') # type: ignore
    assert isinstance(start, pd.Timestamp)
    assert isinstance(end, pd.Timestamp)

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Day-ahead Prices
        print(f"Downloading ENTSO-E Day-ahead prices from {start_date} to {end_date}...")
        prices = client.query_day_ahead_prices("CH", start=start, end=end)
        prices_df = prices.to_frame(name="price")
        prices_df.to_csv(os.path.join(output_dir, "entsoe_prices.csv"))
        
        # Load (Actual)
        print(f"Downloading ENTSO-E Load data...")
        load = client.query_load("CH", start=start, end=end)
        load.to_csv(f"{output_dir}/entsoe_load.csv")
        
        print("Successfully downloaded ENTSO-E data.")
    except Exception as e:
        print(f"Error fetching ENTSO-E data: {e}")
        print("Falling back to generating mock data for testing purposes.")
        generate_mock_entsoe_data(start_date, end_date, output_dir)

def generate_mock_entsoe_data(start_date, end_date, output_dir):
    print("Generating mock ENTSO-E data...")
    os.makedirs(output_dir, exist_ok=True)
    
    start = pd.Timestamp(start_date, tz='Europe/Zurich')
    end = pd.Timestamp(end_date, tz='Europe/Zurich')
    date_range = pd.date_range(start, end, freq='h') # hourly frequency
    
    # Prices mock
    prices = pd.Series(
        np.random.normal(loc=100.0, scale=30.0, size=len(date_range)), 
        index=date_range,
        name='price'
    )
    prices.to_csv(f"{output_dir}/entsoe_prices.csv", header=True)
    
    # Load mock
    load = pd.Series(
        np.random.normal(loc=6000.0, scale=1000.0, size=len(date_range)), 
        index=date_range,
        name='Actual Load'
    )
    load.to_csv(f"{output_dir}/entsoe_load.csv", header=True)
    print("Mock ENTSO-E data generated.")

if __name__ == "__main__":
    # Download last 30 days as an example
    end = pd.Timestamp.now(tz='Europe/Zurich')
    start = end - pd.Timedelta(days=30)
    download_entsoe_data(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')) # type: ignore
