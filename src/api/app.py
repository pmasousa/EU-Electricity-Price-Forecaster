import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

def fetch_predictions():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            data = response.json()["forecast"]
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['timestamp'], df['predicted_price_chf_mwh'], marker='o', linestyle='-', color='b')
            ax.set_title("Day-Ahead Electricity Price Forecast (CH)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (CHF/MWh)")
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig, df
        else:
            return None, pd.DataFrame({"error": ["API Error"]})
    except Exception as e:
        return None, pd.DataFrame({"error": [f"Connection error: {str(e)} - Is FastAPI running?"]})

# Create Gradio interface
with gr.Blocks(title="Electricity Price Forecaster") as demo:
    gr.Markdown("# Swiss Day-Ahead Electricity Price Forecaster")
    gr.Markdown("Click the button below to fetch the 24-hour forecast from the backend API.")
    
    with gr.Row():
        predict_btn = gr.Button("Get Forecast", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=2):
            plot_output = gr.Plot()
        with gr.Column(scale=1):
            table_output = gr.Dataframe()
            
    predict_btn.click(fn=fetch_predictions, outputs=[plot_output, table_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
