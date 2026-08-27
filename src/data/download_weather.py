import os
import sys
from datetime import datetime, timedelta

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Allow running as a script: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import get_country


def download_weather_data(
    start_date: str, end_date: str, country: str = "CH", output_dir: str = "data/raw"
):
    """Download historical weather data for one country from Open-Meteo.

    Uses the country's representative coordinates/timezone from ``src.config`` and
    writes ``weather_{country}.csv``. No API key required.
    """
    cfg = get_country(country)
    print(
        f"Downloading Open-Meteo data for {cfg['name']} ({country})"
        f" from {start_date} to {end_date}..."
    )

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)  # type: ignore

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "start_date": start_date,  # Format: YYYY-MM-DD
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "direct_radiation"],
        "timezone": cfg["tz"],
    }

    out_path = f"{output_dir}/weather_{country}.csv"

    try:
        responses = openmeteo.weather_api(url, params=params)

        # Process first location
        response = responses[0]

        hourly = response.Hourly()
        assert hourly is not None
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()  # type: ignore
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()  # type: ignore
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()  # type: ignore
        hourly_direct_radiation = hourly.Variables(3).ValuesAsNumpy()  # type: ignore

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        hourly_data["temperature_2m"] = hourly_temperature_2m  # type: ignore
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m  # type: ignore
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m  # type: ignore
        hourly_data["direct_radiation"] = hourly_direct_radiation  # type: ignore

        hourly_dataframe = pd.DataFrame(data=hourly_data)

        os.makedirs(output_dir, exist_ok=True)
        hourly_dataframe.to_csv(out_path, index=False)
        print(f"Successfully downloaded Open-Meteo data for {country}.")

    except Exception as e:
        print(f"Error downloading weather data for {country}: {e}")
        print("Generating mock weather data for testing...")
        generate_mock_weather_data(start_date, end_date, country, output_dir)


def generate_mock_weather_data(
    start_date, end_date, country: str = "CH", output_dir: str = "data/raw"
):
    get_country(country)  # validate
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start, end, freq='h')

    df = pd.DataFrame({
        "date": date_range,
        "temperature_2m": np.random.normal(15, 10, len(date_range)),
        "relative_humidity_2m": np.random.normal(60, 20, len(date_range)),
        "wind_speed_10m": np.random.normal(10, 5, len(date_range)),
        "direct_radiation": np.random.normal(200, 100, len(date_range))
    })

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/weather_{country}.csv", index=False)


if __name__ == "__main__":
    import argparse

    from src.config import parse_countries

    parser = argparse.ArgumentParser(description="Download Open-Meteo weather data per country.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in config).")
    parser.add_argument("--days", type=int, default=365 * 3,
                        help="Number of days of history to download (default: 3 years). "
                             "Use a small value (e.g. 90) for a fast test run.")
    args = parser.parse_args()
    countries = parse_countries(args.countries)

    end = datetime.now() - timedelta(days=5)  # Archive API has ~5 day delay
    start = end - timedelta(days=args.days)
    for country in countries:
        download_weather_data(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), country=country)
