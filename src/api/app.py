import json
import os
import sys

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Allow running as a module: make ``src`` importable from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import COUNTRIES, DEFAULT_COUNTRIES, get_country

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Distinct colors per country for overlay plots.
COUNTRY_COLORS = {
    "CH": "#d62728",  # red
    "PT": "#2ca02c",  # green
    "ES": "#1f77b4",  # blue
}


def _color_for(country: str) -> str:
    return COUNTRY_COLORS.get(country, None) or "#7f7f7f"


def fetch_predictions(country, target_date="", show_ci=False):
    """Fetch a single-country forecast and render a plot + table."""
    try:
        params = {"country": country}
        if target_date:
            params["target_date"] = target_date
        response = requests.get(f"{API_URL}/predict", params=params)
        if response.status_code == 200:
            payload = response.json()
            data = payload.get("forecast", [])
            if not data:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No data returned", ha='center')
                return fig, pd.DataFrame({"error": ["No data returned from API"]})

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            df_display = df.copy()
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

            display_rename = {
                "timestamp": "Time",
                "predicted_price_eur_mwh": "Predicted Price (EUR/MWh)",
            }
            if "actual_price_eur_mwh" in df_display.columns:
                display_rename["actual_price_eur_mwh"] = "Actual Price (EUR/MWh)"
            if "q10" in df_display.columns:
                display_rename["q10"] = "Lower Bound (10%)"
                display_rename["q90"] = "Upper Bound (90%)"

            df_display = df_display.rename(columns=display_rename)
            for col in df_display.columns:
                if col != "Time":
                    df_display[col] = df_display[col].round(2)

            fig, ax = plt.subplots(figsize=(10, 5))

            if show_ci and "q10" in df.columns and "q90" in df.columns:
                ax.fill_between(df['timestamp'], df['q10'], df['q90'],
                                color=_color_for(country), alpha=0.15, label='10th-90th CI')

            ax.plot(df['timestamp'], df['predicted_price_eur_mwh'],
                    color=_color_for(country), linewidth=2, label='Predicted Price')

            if "actual_price_eur_mwh" in df.columns:
                ax.plot(df['timestamp'], df['actual_price_eur_mwh'],
                        color='black', linestyle='--', linewidth=2, label='Actual Price')

            country_name = get_country(country)["name"]
            ax.set_title(f"Price Forecast Comparison — {country_name} ({country})"
                         f"{f' for {target_date}' if target_date else ''}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (EUR/MWh)")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            return fig, df_display
        else:
            error_msg = response.json().get('detail', 'Unknown API Error')
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"API Error: {error_msg}", ha='center')
            return fig, pd.DataFrame({"error": [f"API Error {response.status_code}: {error_msg}"]})
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Connection error", ha='center')
        return fig, pd.DataFrame(
            {"error": [f"Connection error: {e} - Is FastAPI backend running?"]}
        )


def fetch_comparison(countries_csv, target_date=""):
    """Fetch forecasts for all requested countries and overlay them on one plot."""
    try:
        params = {}
        if countries_csv:
            params["countries"] = countries_csv
        if target_date:
            params["target_date"] = target_date
        response = requests.get(f"{API_URL}/compare", params=params)
        if response.status_code != 200:
            err = response.json().get('detail', 'Unknown API Error')
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"API Error: {err}", ha='center')
            return fig, pd.DataFrame({"error": [f"API Error {response.status_code}: {err}"]})

        payload = response.json()
        forecasts = payload.get("forecasts", [])
        if not forecasts:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data returned", ha='center')
            return fig, pd.DataFrame({"error": ["No data returned from API"]})

        fig, ax = plt.subplots(figsize=(12, 6))
        table_rows = []
        timestamps = None

        for entry in forecasts:
            country = entry["country"]
            data = entry["forecast"]
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if timestamps is None:
                timestamps = df['timestamp']

            ax.plot(df['timestamp'], df['predicted_price_eur_mwh'],
                    color=_color_for(country), linewidth=2,
                    label=f"{country} — {entry['country_name']}")

            if "q10" in df.columns and "q90" in df.columns:
                ax.fill_between(df['timestamp'], df['q10'], df['q90'],
                                color=_color_for(country), alpha=0.08)

            # Build a flat row table: timestamp + per-country predicted price column.
            col_name = f"{country} (EUR/MWh)"
            if not table_rows:
                table_rows = pd.DataFrame({"Time": df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')})
            table_rows[col_name] = df['predicted_price_eur_mwh'].round(2)

        title_date = f" for {target_date}" if target_date else " — Next 24h"
        ax.set_title(f"Cross-Country Price Forecast{title_date}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        skipped = payload.get("skipped", [])
        if skipped:
            note = "; ".join(f"{s['country']} ({s['reason']})" for s in skipped)
            print(f"Skipped countries in comparison: {note}")

        return fig, table_rows
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Connection error", ha='center')
        return fig, pd.DataFrame(
            {"error": [f"Connection error: {e} - Is FastAPI backend running?"]}
        )


def fetch_metrics():
    """Fetch the per-country metrics table from /metrics."""
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code != 200:
            return pd.DataFrame({"error": [f"API Error {response.status_code}"]})
        records = response.json().get("metrics", [])
        if not records:
            return pd.DataFrame({"info": ["No metrics found. Run the pipeline first."]})
        df = pd.DataFrame(records)
        # Present currency consistently; values are already EUR/MWh.
        df["mae_eur_mwh"] = df["mae"]
        df["rmse_eur_mwh"] = df["rmse"]
        return df[["country", "model", "mae_eur_mwh", "rmse_eur_mwh"]].round(2)
    except Exception as e:
        return pd.DataFrame({"error": [f"Connection error: {e}"]})


def fetch_summary(countries_csv):
    """Fetch the per-country price-level summary from /summary."""
    try:
        params = {"countries": countries_csv} if countries_csv else {}
        response = requests.get(f"{API_URL}/summary", params=params)
        if response.status_code != 200:
            return pd.DataFrame({"error": [f"API Error {response.status_code}"]})
        rows = response.json().get("summary", [])
        if not rows:
            return pd.DataFrame({"info": ["No summary available."]})
        df = pd.DataFrame(rows)
        numeric_cols = ["mean", "median", "min", "max", "peak_price"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = df[c].round(2)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [f"Connection error: {e}"]})


try:
    # Serving bounds from the CH serving bundle (built by build_serving.py);
    # fall back to the raw data end when no bundle exists yet.
    with open("models/serving_CH/config.json") as _f:
        _cfg = json.load(_f)
    last_train_date = pd.to_datetime(_cfg["fit_through"]).strftime('%Y-%m-%d')
    last_context_date = pd.to_datetime(_cfg["data_through"]).strftime('%Y-%m-%d')
except Exception:
    try:
        df_meta = pd.read_csv("data/processed/features_CH.csv", usecols=[0])
        last_train_date = "Unknown"
        last_context_date = pd.to_datetime(df_meta.iloc[-1, 0]).strftime('%Y-%m-%d')
    except Exception:
        last_train_date = "Unknown"
        last_context_date = "Unknown"

# Create Gradio interface
with gr.Blocks(title="Electricity Price Forecaster", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Multi-Country Electricity Price Forecaster")
    gr.Markdown(
        "Forecast and compare day-ahead electricity prices across countries using a Deep "
        "Temporal Fusion Transformer model."
    )
    gr.Markdown(
        f"*(Note: model weights trained on data up to **{last_train_date}**. "
        f"Latest context data available up to **{last_context_date}**)*"
    )

    # ---------------- Single-country forecast tab ----------------
    with gr.Tab("Single Country"):
        with gr.Row():
            country_dd = gr.Dropdown(
                choices=[f"{c} — {COUNTRIES[c]['name']}" for c in DEFAULT_COUNTRIES],
                value=f"{DEFAULT_COUNTRIES[0]} — {COUNTRIES[DEFAULT_COUNTRIES[0]]['name']}",
                label="Country",
            )
            show_ci_cb = gr.Checkbox(label="Show 10th-90th Percentile CI", value=True)
        with gr.Row():
            predict_btn = gr.Button("Get Tomorrow's Forecast", variant="primary")
            target_date_input = gr.Textbox(label="Past Date for Comparison (YYYY-MM-DD)",
                                           placeholder="e.g. 2026-06-01")
            compare_btn = gr.Button("Compare Past Date", variant="secondary")
        with gr.Row():
            with gr.Column(scale=2):
                plot_output = gr.Plot()
            with gr.Column(scale=1):
                table_output = gr.Dataframe(interactive=False)

        def _selected_country_code(label):
            return label.split(" — ")[0].strip().upper()

        predict_btn.click(
            fn=lambda c, show_ci: fetch_predictions(_selected_country_code(c), "", show_ci),
            inputs=[country_dd, show_ci_cb], outputs=[plot_output, table_output],
        )
        compare_btn.click(
            fn=lambda c, d, show_ci: fetch_predictions(_selected_country_code(c), d, show_ci),
            inputs=[country_dd, target_date_input, show_ci_cb], outputs=[plot_output, table_output],
        )

    # ---------------- Cross-country comparison tab ----------------
    with gr.Tab("Compare Countries"):
        with gr.Row():
            compare_countries_box = gr.Textbox(
                label="Countries (comma-separated)", value=",".join(DEFAULT_COUNTRIES),
                placeholder="e.g. PT,ES,CH",
            )
            compare_target_date = gr.Textbox(label="Past Date (YYYY-MM-DD, optional)",
                                             placeholder="e.g. 2026-06-01")
        compare_all_btn = gr.Button("Compare Forecasts", variant="primary")
        with gr.Row():
            with gr.Column(scale=2):
                compare_plot = gr.Plot()
            with gr.Column(scale=1):
                compare_table = gr.Dataframe(interactive=False)
        compare_all_btn.click(
            fn=fetch_comparison,
            inputs=[compare_countries_box, compare_target_date],
            outputs=[compare_plot, compare_table],
        )

    # ---------------- Metrics & summary tab ----------------
    with gr.Tab("Metrics & Summary"):
        with gr.Row():
            metrics_btn = gr.Button("Load Per-Country Metrics", variant="primary")
            summary_btn = gr.Button("Load Price Summary", variant="secondary")
            summary_countries_box = gr.Textbox(
                label="Countries for Summary", value=",".join(DEFAULT_COUNTRIES),
            )
        metrics_table = gr.Dataframe(interactive=False, label="Model Metrics (EUR/MWh)")
        summary_table = gr.Dataframe(interactive=False, label="Price-Level Summary (EUR/MWh)")
        metrics_btn.click(fn=fetch_metrics, inputs=[], outputs=[metrics_table])
        summary_btn.click(fn=fetch_summary, inputs=[summary_countries_box], outputs=[summary_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
