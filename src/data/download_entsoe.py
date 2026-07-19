import os
import pandas as pd
import requests
from datetime import datetime, timedelta

def download_entsoe_data(start_date, end_date, output_dir="data/raw"):
    print("Downloading ENTSO-E data from Energy-Charts API...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Format dates to match the API requirements
    start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # 1. Download Prices
    print("Fetching Day-Ahead Prices...")
    prices_url = f"https://api.energy-charts.info/price?bzn=CH&start={start}&end={end}"
    response = requests.get(prices_url)
    if response.status_code == 200:
        data = response.json()
        timestamps = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
        prices = pd.Series(data['price'], index=timestamps, name='price')
        prices.to_csv(os.path.join(output_dir, "entsoe_prices.csv"), header=True)
    else:
        print(f"Error fetching prices: {response.text}")
        generate_mock_entsoe_data(start_date, end_date, output_dir)
        return
        
    # 2. Download Load
    print("Fetching Actual Load...")
    load_url = f"https://api.energy-charts.info/public_power?country=ch&start={start}&end={end}"
    response = requests.get(load_url)
    if response.status_code == 200:
        data = response.json()
        timestamps = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
        load_data = next((pt['data'] for pt in data['production_types'] if pt['name'] == 'Load'), None)
        
        if load_data is not None:
            load = pd.Series(load_data, index=timestamps, name='Actual Load')
            load.to_csv(os.path.join(output_dir, "entsoe_load.csv"), header=True)
        else:
            print("Load data not found in response.")
            generate_mock_entsoe_data(start_date, end_date, output_dir)
            return
    else:
        print(f"Error fetching load: {response.text}")
        generate_mock_entsoe_data(start_date, end_date, output_dir)
        return
        
    print("Successfully downloaded real ENTSO-E data!")

def generate_mock_entsoe_data(start_date, end_date, output_dir):
    print("Generating mock ENTSO-E data as fallback...")
    os.makedirs(output_dir, exist_ok=True)
    
    start = pd.Timestamp(start_date, tz='Europe/Zurich')
    end = pd.Timestamp(end_date, tz='Europe/Zurich')
    date_range = pd.date_range(start, end, freq='h') # hourly frequency
    
    # Prices mock
    prices = pd.Series(
        pd.Series(range(len(date_range))).apply(lambda x: 100 + 20 * (x % 24)), # fake pattern
        index=date_range,
        name='price'
    )
    prices.to_csv(os.path.join(output_dir, "entsoe_prices.csv"), header=True)
    
    # Load mock
    load = pd.Series(
        pd.Series(range(len(date_range))).apply(lambda x: 6000 + 1000 * (x % 24)), 
        index=date_range,
        name='Actual Load'
    )
    load.to_csv(os.path.join(output_dir, "entsoe_load.csv"), header=True)
    print("Mock ENTSO-E data generated.")

if __name__ == "__main__":
    # Download last 365 days of real data
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=365)
    download_entsoe_data(start, end)
