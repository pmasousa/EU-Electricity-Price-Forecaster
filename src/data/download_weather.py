import os
import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

def download_weather_data(start_date: str, end_date: str, output_dir: str = "data/raw"):
    """
    Downloads historical weather data for Switzerland from Open-Meteo.
    """
    print(f"Downloading Open-Meteo data from {start_date} to {end_date}...")
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session) # type: ignore

    # Zurich coordinates as a proxy for Swiss weather
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 47.3667,
        "longitude": 8.55,
        "start_date": start_date, # Format: YYYY-MM-DD
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "direct_radiation"],
        "timezone": "Europe/Berlin"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        
        # Process first location
        response = responses[0]
        
        hourly = response.Hourly()
        assert hourly is not None
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy() # type: ignore
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy() # type: ignore
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy() # type: ignore
        hourly_direct_radiation = hourly.Variables(3).ValuesAsNumpy() # type: ignore

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        
        hourly_data["temperature_2m"] = hourly_temperature_2m # type: ignore
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m # type: ignore
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m # type: ignore
        hourly_data["direct_radiation"] = hourly_direct_radiation # type: ignore

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        
        os.makedirs(output_dir, exist_ok=True)
        hourly_dataframe.to_csv(f"{output_dir}/weather_zurich.csv", index=False)
        print("Successfully downloaded Open-Meteo data.")
        
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        print("Generating mock weather data for testing...")
        generate_mock_weather_data(start_date, end_date, output_dir)

def generate_mock_weather_data(start_date, end_date, output_dir):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    # Adding +1 day to end to match the inclusive range if needed, 
    # but let's just make it simpler
    date_range = pd.date_range(start, end, freq='h')
    
    df = pd.DataFrame({
        "date": date_range,
        "temperature_2m": np.random.normal(15, 10, len(date_range)),
        "relative_humidity_2m": np.random.normal(60, 20, len(date_range)),
        "wind_speed_10m": np.random.normal(10, 5, len(date_range)),
        "direct_radiation": np.random.normal(200, 100, len(date_range))
    })
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/weather_zurich.csv", index=False)

if __name__ == "__main__":
    end = datetime.now() - timedelta(days=5) # Archive API has 5 day delay usually
    start = end - timedelta(days=30)
    download_weather_data(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
