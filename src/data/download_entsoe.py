import os
import sys
import pandas as pd
import requests

# Allow running as a script: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import COUNTRIES, DEFAULT_COUNTRIES, get_country

# Energy-Charts aggregates EPEX SPOT / ENTSO-E transparency data. No API key.
PRICE_URL = "https://api.energy-charts.info/price"
LOAD_URL = "https://api.energy-charts.info/public_power"


def download_entsoe_data(start_date, end_date, country: str = "CH", output_dir: str = "data/raw"):
    """Download day-ahead prices and actual load for one country from Energy-Charts.

    Prices are returned in EUR/MWh and are resampled to an hourly grid (some zones
    publish at 15-minute resolution). Output files are suffixed with the country
    code: ``entsoe_prices_{country}.csv`` and ``entsoe_load_{country}.csv``.
    Falls back to mock data if the API call fails.
    """
    cfg = get_country(country)
    print(f"Downloading Energy-Charts data for {cfg['name']} ({country})...")
    os.makedirs(output_dir, exist_ok=True)

    # Format dates to match the API requirements
    start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    prices_path = os.path.join(output_dir, f"entsoe_prices_{country}.csv")
    load_path = os.path.join(output_dir, f"entsoe_load_{country}.csv")

    # 1. Download Prices
    print("Fetching Day-Ahead Prices...")
    prices_url = f"{PRICE_URL}?bzn={cfg['bzn']}&start={start}&end={end}"
    response = requests.get(prices_url)
    if response.status_code == 200:
        data = response.json()
        timestamps = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
        prices = pd.Series(data['price'], index=timestamps, name='price')
        # Normalize to hourly: some zones (ES, PT) report 15-min resolution.
        prices = _to_hourly(prices)
        prices.to_csv(prices_path, header=True)
    else:
        print(f"Error fetching prices: {response.text}")
        generate_mock_entsoe_data(start_date, end_date, country, output_dir)
        return

    # 2. Download Load
    print("Fetching Actual Load...")
    load_url = f"{LOAD_URL}?country={cfg['country']}&start={start}&end={end}"
    response = requests.get(load_url)
    if response.status_code == 200:
        data = response.json()
        timestamps = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
        load_data = next((pt['data'] for pt in data['production_types'] if pt['name'] == 'Load'), None)

        if load_data is not None:
            load = pd.Series(load_data, index=timestamps, name='Actual Load')
            load = _to_hourly(load)
            load.to_csv(load_path, header=True)
        else:
            print("Load data not found in response.")
            generate_mock_entsoe_data(start_date, end_date, country, output_dir)
            return
    else:
        print(f"Error fetching load: {response.text}")
        generate_mock_entsoe_data(start_date, end_date, country, output_dir)
        return

    print(f"Successfully downloaded real Energy-Charts data for {country}!")


def _to_hourly(series: pd.Series) -> pd.Series:
    """Resample an arbitrary-resolution series to hourly.

    Energy-Charts returns hourly load for all zones, but day-ahead prices are
    15-minute for some bidding zones (e.g. ES, PT) and hourly for others (e.g.
    CH). Down-sampling via mean keeps the price column on a common hourly grid so
    it aligns with the hourly weather/load series during feature merging.
    """
    # If already hourly, this is a no-op.
    resampled = series.resample('h').mean()
    # Forward-fill the rare gaps created when averaging an incomplete hour.
    return resampled.ffill()


def generate_mock_entsoe_data(start_date, end_date, country: str = "CH", output_dir: str = "data/raw"):
    cfg = get_country(country)
    print(f"Generating mock Energy-Charts data as fallback for {country}...")
    os.makedirs(output_dir, exist_ok=True)

    start = pd.Timestamp(start_date, tz=cfg['tz'])
    end = pd.Timestamp(end_date, tz=cfg['tz'])
    date_range = pd.date_range(start, end, freq='h')  # hourly frequency

    # Prices mock
    prices = pd.Series(
        pd.Series(range(len(date_range))).apply(lambda x: 100 + 20 * (x % 24)),  # fake pattern
        index=date_range,
        name='price'
    )
    prices.to_csv(os.path.join(output_dir, f"entsoe_prices_{country}.csv"), header=True)

    # Load mock
    load = pd.Series(
        pd.Series(range(len(date_range))).apply(lambda x: 6000 + 1000 * (x % 24)),
        index=date_range,
        name='Actual Load'
    )
    load.to_csv(os.path.join(output_dir, f"entsoe_load_{country}.csv"), header=True)
    print(f"Mock Energy-Charts data generated for {country}.")


if __name__ == "__main__":
    import argparse
    from src.config import parse_countries

    parser = argparse.ArgumentParser(description="Download Energy-Charts data per country.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in config).")
    parser.add_argument("--days", type=int, default=365 * 3,
                        help="Number of days of history to download (default: 3 years). "
                             "Use a small value (e.g. 90) for a fast test run.")
    args = parser.parse_args()
    countries = parse_countries(args.countries)

    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=args.days)
    for country in countries:
        download_entsoe_data(start, end, country=country)
