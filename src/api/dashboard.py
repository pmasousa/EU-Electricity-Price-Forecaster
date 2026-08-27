"""Streamlit dashboard for the EU Electricity Price Forecaster API.

Reactive by construction: every widget change reruns this script, so forecasts
load on open and update immediately — no buttons, no event wiring. Serves the
TFT (quantile bands) plus the benchmarked Linear Regression and LightGBM
refits via the API's ``model`` parameter.

Run: streamlit run src/api/dashboard.py   (API on API_URL, default :8000)
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import COUNTRIES, DEFAULT_COUNTRIES, get_country

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

COUNTRY_COLORS = {"PT": "#d62728", "ES": "#1f77b4", "CH": "#2ca02c"}
MODEL_LABELS = {
    "tft": "TFT — quantile bands",
    "lr": "Linear Regression",
    "lgbm": "LightGBM",
}

# Line style per overlaid model; the TFT keeps the country color, the
# classical models get fixed hues so they read across countries.
MODEL_LINE = {
    "lr": {"color": "#e67e22", "dash": "dash", "width": 2.1},
    "lgbm": {"color": "#8e44ad", "dash": "dot", "width": 2.1},
}

DARK_CSS = """
<style>
.stApp, [data-testid="stSidebar"] {background: #0e1512 !important; color: #e6edf3 !important;}
[data-testid="stSidebar"] {border-color: #27352d !important;}
.stPlotlyChart, [data-testid="stDataFrame"] {border-radius: 8px;}
</style>
"""


@st.cache_data(ttl=300, show_spinner=False)
def _get(path, params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_forecast(country, model, target_date=""):
    params = {"country": country, "model": model}
    if target_date:
        params["target_date"] = target_date
    return _get("/predict", params).get("forecast", [])


def fetch_comparison(codes, model):
    return _get("/compare", {"countries": ",".join(codes), "model": model})


def fetch_metrics():
    return _get("/metrics", {}).get("metrics", [])


# ---------------- page ----------------
st.set_page_config(
    page_title="⚡ EU Electricity Price Forecaster",
    page_icon="⚡",
    layout="wide",
)

with st.sidebar:
    dark = st.toggle("🌙 Dark mode", value=False)
    country_label = st.selectbox(
        "Country",
        [f"{c} — {COUNTRIES[c]['name']}" for c in DEFAULT_COUNTRIES],
        index=0,
    )
    country = country_label.split(" — ")[0].strip()
    models = st.multiselect(
        "Models (click the legend to toggle lines)",
        list(MODEL_LABELS), default=["tft"], format_func=MODEL_LABELS.get,
    )
    if not models:
        st.info("Pick at least one model.")
        st.stop()
    show_ci = "tft" in models and st.checkbox("Show q10–q90 band (TFT)", value=True)
    past_date = st.text_input("Past date to compare (YYYY-MM-DD)", placeholder="2026-06-01")
    if past_date and not past_date.strip():
        past_date = ""

if dark:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

template = "plotly_dark" if dark else "plotly_white"
plot_bg = "rgba(0,0,0,0)"

try:
    with st.spinner("Loading forecast…"):
        frames = {m: fetch_forecast(country, m, past_date.strip()) for m in models}
except Exception as e:
    st.error(f"Backend unreachable ({e}). Is the FastAPI server running on {API_URL}?")
    st.stop()

st.title("⚡ EU Electricity Price Forecaster")
st.caption(
    f"Day-ahead electricity prices · {get_country(country)['name']} · "
    + " · ".join(MODEL_LABELS[m] for m in models)
)

first = pd.DataFrame(frames[models[0]])
if first.empty:
    st.warning("No forecast data returned.")
    st.stop()
first["timestamp"] = pd.to_datetime(first["timestamp"])
prices = first["predicted_price_eur_mwh"]
peak_hour = first.loc[prices.idxmax(), "timestamp"].strftime("%H:%M")
st.markdown(
    f"**Peak {prices.max():.1f} EUR/MWh** at {peak_hour} ({MODEL_LABELS[models[0]]}) · "
    f"min {prices.min():.1f} · mean {prices.mean():.1f}"
)

color = COUNTRY_COLORS.get(country, "#7f7f7f")
fig = go.Figure()
if show_ci and {"q10", "q90"} <= set(first.columns):
    fig.add_trace(go.Scatter(
        x=first["timestamp"], y=first["q90"], mode="lines",
        line={"width": 0}, hoverinfo="skip", name="q90",
    ))
    fig.add_trace(go.Scatter(
        x=first["timestamp"], y=first["q10"], mode="lines",
        line={"width": 0}, fill="tonexty", fillcolor=f"rgba({int(color[1:3], 16)},"
        f"{int(color[3:5], 16)},{int(color[5:7], 16)},0.16)",
        hoverinfo="skip", name="q10–q90 band",
    ))

for m in models:
    d = first if m == models[0] else pd.DataFrame(frames[m])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    style = {"color": color, "width": 2.6, "dash": "solid"} if m == "tft" else MODEL_LINE[m]
    fig.add_trace(go.Scatter(
        x=d["timestamp"], y=d["predicted_price_eur_mwh"],
        mode="lines+markers" if m == "tft" else "lines",
        line=style, marker={"size": 5} if m == "tft" else None,
        name=MODEL_LABELS[m],
        hovertemplate="%{x|%H:%M} — %{y:.1f} EUR/MWh<extra></extra>",
    ))

if "actual_price_eur_mwh" in first.columns:
    fig.add_trace(go.Scatter(
        x=first["timestamp"], y=first["actual_price_eur_mwh"], mode="lines",
        line={"color": "black", "dash": "dash", "width": 1.8},
        name="actual",
        hovertemplate="%{x|%H:%M} — %{y:.1f} EUR/MWh<extra></extra>",
    ))
fig.update_layout(
    template=template,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
    height=440,
    plot_bgcolor=plot_bg,
    paper_bgcolor=plot_bg,
    yaxis_title="EUR / MWh",
    legend={"orientation": "h", "y": 1.02},
    xaxis={"dtick": 3600000 * 3, "tickformat": "%H:%M"},
)
st.plotly_chart(fig, use_container_width=True)

display = pd.DataFrame({"Time": first["timestamp"].dt.strftime("%Y-%m-%d %H:%M")})
for m in models:
    d = first if m == models[0] else pd.DataFrame(frames[m])
    display[MODEL_LABELS[m]] = d["predicted_price_eur_mwh"].round(2).tolist()
for q in ("q10", "q90"):
    if show_ci and q in first.columns:
        display[f"TFT {q}"] = first[q].round(2).tolist()
if "actual_price_eur_mwh" in first.columns:
    display["Actual (EUR/MWh)"] = first["actual_price_eur_mwh"].round(2).tolist()
st.dataframe(display, use_container_width=True, height=320)

# ---------------- compare + benchmark tabs ----------------
tab_compare, tab_bench = st.tabs(["Compare countries", "Benchmark"])

with tab_compare:
    cmp_codes = st.multiselect(
        "Countries", list(COUNTRIES), default=list(DEFAULT_COUNTRIES)
    )
    cmp_model = st.selectbox(
        "Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get, key="cmp_model"
    )
    if cmp_codes:
        try:
            payload = fetch_comparison(cmp_codes, cmp_model)
            fig2 = go.Figure()
            table = None
            for entry in payload.get("forecasts", []):
                c = entry["country"]
                d = pd.DataFrame(entry["forecast"])
                d["timestamp"] = pd.to_datetime(d["timestamp"])
                col = COUNTRY_COLORS.get(c, "#7f7f7f")
                if {"q10", "q90"} <= set(d.columns) and cmp_model == "tft":
                    fig2.add_trace(go.Scatter(
                        x=d["timestamp"], y=d["q90"], mode="lines",
                        line={"width": 0}, hoverinfo="skip", name=f"{c} q90",
                    ))
                    fig2.add_trace(go.Scatter(
                        x=d["timestamp"], y=d["q10"], mode="lines", line={"width": 0},
                        fill="tonexty",
                        fillcolor=f"rgba({int(col[1:3], 16)},{int(col[3:5], 16)},"
                        f"{int(col[5:7], 16)},0.10)",
                        hoverinfo="skip", name=f"{c} band",
                    ))
                fig2.add_trace(go.Scatter(
                    x=d["timestamp"], y=d["predicted_price_eur_mwh"], mode="lines",
                    line={"color": col, "width": 2.4}, name=entry["country_name"],
                    hovertemplate="%{x|%H:%M} — %{y:.1f} EUR/MWh<extra></extra>",
                ))
                if table is None:
                    table = pd.DataFrame({"Time": d["timestamp"].dt.strftime("%H:%M")})
                table[c] = d["predicted_price_eur_mwh"].round(2)
            fig2.update_layout(
                template=template, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                height=420, plot_bgcolor=plot_bg, paper_bgcolor=plot_bg,
                yaxis_title="EUR / MWh",
                legend={"orientation": "h", "y": 1.02},
                xaxis={"dtick": 3600000 * 3, "tickformat": "%H:%M"},
            )
            st.plotly_chart(fig2, use_container_width=True)
            if table is not None:
                st.dataframe(table, use_container_width=True, height=300)
            skipped = payload.get("skipped", [])
            if skipped:
                st.info("; ".join(f"{s['country']}: {s['reason']}" for s in skipped))
        except Exception as e:
            st.error(f"Comparison failed: {e}")

with tab_bench:
    try:
        records = fetch_metrics()
    except Exception as e:
        st.error(f"Metrics unavailable: {e}")
        records = []
    if records:
        b = pd.DataFrame(records)
        b["served"] = b["served"].map(lambda v: "✓" if v else "")
        b = b.rename(columns={
            "country": "Country", "model": "Model", "served": "Served",
            "mae": "MAE (EUR/MWh)", "rmse": "RMSE (EUR/MWh)", "rmae": "rMAE",
        })
        keep = [c for c in ("Country", "Model", "MAE (EUR/MWh)",
                            "RMSE (EUR/MWh)", "rmae", "Served") if c in b.columns]
        b = b[keep].sort_values(["Country", "Served"], ascending=[True, False])
        st.caption("Walk-forward benchmark (8-week holdout, EUR/MWh) — ✓ marks the served model.")
        st.dataframe(b, use_container_width=True, height=420)
    else:
        st.info("No benchmark tables found — run the pipeline first.")
