import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

def fetch_predictions(target_date="", show_ci=False):
    try:
        params = {}
        if target_date:
            params["target_date"] = target_date
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            data = response.json().get("forecast", [])
            if not data:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No data returned", ha='center')
                return fig, pd.DataFrame({"error": ["No data returned from API"]})
                
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Format dataframe for display
            df_display = df.copy()
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            display_rename = {
                "timestamp": "Time",
                "predicted_price_chf_mwh": "Predicted Price (CHF/MWh)"
            }
            if "actual_price_chf_mwh" in df_display.columns:
                display_rename["actual_price_chf_mwh"] = "Actual Price (CHF/MWh)"
            if "q10" in df_display.columns:
                display_rename["q10"] = "Lower Bound (10%)"
                display_rename["q90"] = "Upper Bound (90%)"
            
            df_display = df_display.rename(columns=display_rename)
            for col in df_display.columns:
                if col != "Time":
                    df_display[col] = df_display[col].round(2)
            
            # Create matplotlib plot for shaded areas
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot CI
            if show_ci and "q10" in df.columns and "q90" in df.columns:
                ax.fill_between(df['timestamp'], df['q10'], df['q90'], color='blue', alpha=0.15, label='10th-90th CI')
                
            # Plot Predicted
            ax.plot(df['timestamp'], df['predicted_price_chf_mwh'], color='blue', linewidth=2, label='Predicted Price')
            
            # Plot Actual
            if "actual_price_chf_mwh" in df.columns:
                ax.plot(df['timestamp'], df['actual_price_chf_mwh'], color='red', linestyle='--', linewidth=2, label='Actual Price')
                
            ax.set_title(f"Price Forecast Comparison for {target_date}" if target_date else "Next 24h Electricity Price Forecast (CH)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (CHF/MWh)")
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
        ax.text(0.5, 0.5, f"Connection error", ha='center')
        return fig, pd.DataFrame({"error": [f"Connection error: {str(e)} - Is FastAPI backend running?"]})

try:
    # Quick read of the last date from the dataset to show training bounds
    df_meta = pd.read_csv("data/processed/features.csv", usecols=[0])
    last_data_date = pd.to_datetime(df_meta.iloc[-1, 0])
    
    # 14 days were held out for validation (7) and testing (7)
    last_train_date = (last_data_date - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
    last_context_date = last_data_date.strftime('%Y-%m-%d')
except Exception:
    last_train_date = "Unknown"
    last_context_date = "Unknown"

# Create Gradio interface
with gr.Blocks(title="Electricity Price Forecaster", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Swiss Day-Ahead Electricity Price Forecaster")
    gr.Markdown("Click the button below to fetch the 24-hour forecast from the Deep Temporal Fusion Transformer model.")
    gr.Markdown(f"*(Note: Model weights trained on data up to **{last_train_date}**. Latest context data available up to **{last_context_date}**)*")
    
    with gr.Row():
        predict_btn = gr.Button("Get Tomorrow's Forecast", variant="primary")
        target_date_input = gr.Textbox(label="Past Date for Comparison (YYYY-MM-DD)", placeholder="e.g. 2026-06-01")
        compare_btn = gr.Button("Compare Past Date", variant="secondary")
        show_ci_cb = gr.Checkbox(label="Show 10th-90th Percentile Confidence Interval", value=True)
    
    with gr.Row():
        with gr.Column(scale=2):
            plot_output = gr.Plot()
        with gr.Column(scale=1):
            table_output = gr.Dataframe(interactive=False)
            
    # For default predict, pass empty string so it predicts tomorrow
    predict_btn.click(fn=lambda show_ci: fetch_predictions("", show_ci), inputs=[show_ci_cb], outputs=[plot_output, table_output])
    compare_btn.click(fn=fetch_predictions, inputs=[target_date_input, show_ci_cb], outputs=[plot_output, table_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
