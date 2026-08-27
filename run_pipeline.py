import argparse
import os
import subprocess
import sys

# Allow running as a script from project root: make ``src`` importable.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.config import parse_countries


def run_script(script_path, extra_args=None):
    print(f"\n{'='*50}")
    print(f"Running {script_path} {' '.join(extra_args or [])}...")
    print(f"{'='*50}")

    # We use sys.executable to ensure it runs with the same python interpreter (e.g. the uv env)
    cmd = [sys.executable, script_path] + (extra_args or [])
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Successfully finished {script_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-country forecasting pipeline.")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Script path to start the pipeline from.")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated country codes (default: all in src/config.py). "
                             "Example: --countries CH,PT,ES")
    parser.add_argument("--days", type=int, default=365 * 3,
                        help="Days of history to download (default: 3 years). "
                             "Use a small value (e.g. 90) for a fast smoke test.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max TFT training epochs (default: 100). "
                             "Use a small value (e.g. 3) for a fast smoke test.")
    args = parser.parse_args()

    # Ensure we are in the project root
    if not os.path.exists("src"):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)

    countries = parse_countries(args.countries)
    countries_arg = ",".join(countries)

    # Download steps need --days; the benchmark harness needs --epochs.
    # The harness (src/evaluation/backtest.py) is the single source of truth for
    # model comparison: shared splits, honest covariates, one walk-forward
    # protocol for every model. build_serving then assembles models/serving_{CC}
    # from the benchmarked artifacts so `api`/`ui` have something to load.
    scripts = [
        ("src/data/download_entsoe.py", ["--countries", countries_arg, "--days", str(args.days)]),
        ("src/data/download_weather.py", ["--countries", countries_arg, "--days", str(args.days)]),
        ("src/features/build_features.py", ["--countries", countries_arg]),
        ("src/evaluation/backtest.py",
         ["--countries", countries_arg, "--epochs", str(args.epochs)]),
        ("src/models/build_serving.py", ["--country", countries_arg]),
    ]

    # Resolve --start-from into the script list.
    script_names = [s[0] for s in scripts]
    if args.start_from:
        if args.start_from in script_names:
            start_index = script_names.index(args.start_from)
            scripts = scripts[start_index:]
            print(f"Resuming pipeline from {args.start_from}...")
        else:
            print(f"Error: {args.start_from} is not a valid script in the pipeline. "
                  f"Valid scripts: {script_names}")
            sys.exit(1)

    print(f"Starting Electricity Price Forecaster Pipeline for countries: {countries} "
          f"(days={args.days}, epochs={args.epochs})")
    for script_path, extra in scripts:
        run_script(script_path, extra_args=extra)

    print("Pipeline complete. Reports and plots are in the 'reports' directory.")
