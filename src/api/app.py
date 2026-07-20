import gradio as gr
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

def fetch_predictions(target_date=""):
    try:
        params = {}
        if target_date:
            params["target_date"] = target_date
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            data = response.json().get("forecast", [])
            if not data:
                return gr.LinePlot(visible=False), pd.DataFrame({"error": ["No data returned from API"]})
                
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
            
            df_display = df_display.rename(columns=display_rename)
            for col in df_display.columns:
                if col != "Time":
                    df_display[col] = df_display[col].round(2)
            
            # Melt data for plotting multiple lines if actuals exist
            if "actual_price_chf_mwh" in df.columns:
                df_melted = df.melt(id_vars=["timestamp"], value_vars=["predicted_price_chf_mwh", "actual_price_chf_mwh"], var_name="Type", value_name="Price")
                df_melted["Type"] = df_melted["Type"].map({"predicted_price_chf_mwh": "Predicted", "actual_price_chf_mwh": "Actual"})
                y_col = "Price"
                color_col = "Type"
                title = f"Price Forecast Comparison for {target_date}"
            else:
                df_melted = df.copy()
                y_col = "predicted_price_chf_mwh"
                color_col = None
                title = "Next 24h Electricity Price Forecast (CH)"
                
            # Create interactive line plot
            plot = gr.LinePlot(
                value=df_melted,
                x="timestamp",
                y=y_col,
                color=color_col,
                title=title,
                tooltip=["timestamp", y_col],
                x_title="Time",
                y_title="Price (CHF/MWh)"
            )
            
            return plot, df_display
        else:
            error_msg = response.json().get('detail', 'Unknown API Error')
            return gr.LinePlot(visible=False), pd.DataFrame({"error": [f"API Error {response.status_code}: {error_msg}"]})
    except Exception as e:
        return gr.LinePlot(visible=False), pd.DataFrame({"error": [f"Connection error: {str(e)} - Is FastAPI backend running?"]})

# Create Gradio interface
with gr.Blocks(title="Electricity Price Forecaster", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Swiss Day-Ahead Electricity Price Forecaster")
    gr.Markdown("Click the button below to fetch the 24-hour forecast from the Deep Temporal Fusion Transformer model.")
    
    with gr.Row():
        predict_btn = gr.Button("Get Tomorrow's Forecast", variant="primary")
        target_date_input = gr.Textbox(label="Past Date for Comparison (YYYY-MM-DD)", placeholder="e.g. 2026-06-01")
        compare_btn = gr.Button("Compare Past Date", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=2):
            plot_output = gr.LinePlot()
        with gr.Column(scale=1):
            table_output = gr.Dataframe(interactive=False)
            
    # For default predict, pass empty string so it predicts tomorrow
    predict_btn.click(fn=lambda: fetch_predictions(""), outputs=[plot_output, table_output])
    compare_btn.click(fn=fetch_predictions, inputs=[target_date_input], outputs=[plot_output, table_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
