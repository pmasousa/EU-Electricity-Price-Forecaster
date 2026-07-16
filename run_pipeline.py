import subprocess
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
    
    print("Starting Swiss Electricity Price Forecaster Pipeline...")
    for script in scripts:
        run_script(script)
    
    print("Entire pipeline completed successfully! You can find reports and plots in the 'reports' directory.")
