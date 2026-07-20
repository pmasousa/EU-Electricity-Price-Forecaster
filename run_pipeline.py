import subprocess
import argparse
import sys
import os

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Running {script_path}...")
    print(f"{'='*50}")
    
    # We use sys.executable to ensure it runs with the same python interpreter (e.g., the uv env)
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Successfully finished {script_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forecasting pipeline.")
    parser.add_argument("--start-from", type=str, default=None, help="Script path to start the pipeline from.")
    args = parser.parse_args()

    # Ensure we are in the project root
    if not os.path.exists("src"):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)
        
    scripts = [
        "src/data/download_entsoe.py",
        "src/data/download_weather.py",
        "src/features/build_features.py",
        "src/models/baseline.py",
        "src/models/train_tft.py",
        "src/models/backtest.py",
        "src/models/plot_comparison.py"
    ]
    
    if args.start_from:
        if args.start_from in scripts:
            start_index = scripts.index(args.start_from)
            scripts = scripts[start_index:]
            print(f"Resuming pipeline from {args.start_from}...")
        else:
            print(f"Error: {args.start_from} is not a valid script in the pipeline.")
            sys.exit(1)
    
    print("Starting Swiss Electricity Price Forecaster Pipeline...")
    for script in scripts:
        run_script(script)
    
    print("Entire pipeline completed successfully! You can find reports and plots in the 'reports' directory.")
