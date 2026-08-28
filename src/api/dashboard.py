"""Streamlit dashboard for the EU Electricity Price Forecaster API.

Reactive by construction: every widget change reruns this script, so forecasts
load on open and update immediately — no buttons, no event wiring. Serves the
TFT (quantile bands) plus the benchmarked Linear Regression and LightGBM
refits via the API's ``model`` parameter, overlays countries on the time
window they share, and scores the served models on recent out-of-sample days.

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

HOVER = "%{x|%a %H:%M} — %{y:.1f} EUR/MWh<extra></extra>"

DARK_CSS = """
<style>
/* base surfaces */
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
  background: #0e1512 !important; color: #e6edf3 !important;}
[data-testid="stSidebar"] {border-color: #27352d !important;}
.stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp summary, .stApp td, .stApp th {color: #e6edf3 !important;}
[data-testid="stCaptionContainer"] {color: #9fb3a8 !important;}
/* input + dropdown controls */
[data-testid="stMultiSelect"] .react-aria-ComboBox > div,
[data-testid="stSelectbox"] .react-aria-ComboBox > div,
[data-testid="stTextInput"] input {
  background-color: #16211c !important; color: #e6edf3 !important;
  border-color: #2c3a32 !important;}
[data-testid="stMultiSelect"] input {color: #e6edf3 !important;}
[data-testid="stMultiSelectTagsContainer"] > div {
  background-color: #1d2a23 !important; color: #cfe5d8 !important;
  border-color: #2c3a32 !important;}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background-color: #16211c !important; color: #e6edf3 !important;}
[role="option"] {color: #e6edf3 !important;}
[role="option"]:hover {background-color: #1f2e26 !important;}
/* segmented control (day picker) */
.stButtonGroup button {
  background-color: #16211c !important; color: #9fb3a8 !important;
  border-color: #2c3a32 !important;}
.stButtonGroup button[data-selected="true"],
.stButtonGroup button[aria-pressed="true"] {color: #2e9e5b !important;}
/* tabs */
[data-testid="stTabs"] [role="tab"] {color: #9fb3a8 !important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color: #e6edf3 !important;}
/* expanders + alerts */
[data-testid="stExpander"], [data-testid="stExpanderDetails"] {
  background-color: #131c17 !important; border-color: #27352d !important;}
[data-testid="stAlert"] {background-color: #16211c !important; color: #e6edf3 !important;}
/* dataframes: glide grid palette is inline CSS vars -> override with !important */
.stDataFrameGlideDataEditor {
  --gdg-bg-cell: #131c17 !important;
  --gdg-bg-cell-medium: #131c17 !important;
  --gdg-bg-header: #1a2620 !important;
  --gdg-bg-header-hovered: #223129 !important;
  --gdg-bg-header-has-focus: #223129 !important;
  --gdg-bg-group-header: #1a2620 !important;
  --gdg-bg-group-header-hovered: #223129 !important;
  --gdg-bg-bubble: #1d2a23 !important;
  --gdg-bg-bubble-selected: #1d2a23 !important;
  --gdg-bg-icon-header: rgba(230, 237, 243, 0.6) !important;
  --gdg-text-dark: #e6edf3 !important;
  --gdg-text-medium: rgba(230, 237, 243, 0.85) !important;
  --gdg-text-light: rgba(230, 237, 243, 0.5) !important;
  --gdg-text-header: rgba(230, 237, 243, 0.6) !important;
  --gdg-text-header-selected: #ffffff !important;
  --gdg-text-group-header: rgba(230, 237, 243, 0.6) !important;
  --gdg-text-bubble: rgba(230, 237, 243, 0.6) !important;
  --gdg-border-color: rgba(230, 237, 243, 0.12) !important;
  --gdg-horizontal-border-color: rgba(230, 237, 243, 0.12) !important;}
/* polish */
.stPlotlyChart, [data-testid="stDataFrame"] {border-radius: 8px;}
::-webkit-scrollbar {width: 10px; height: 10px;}
::-webkit-scrollbar-track {background: #0e1512;}
::-webkit-scrollbar-thumb {background: #27352d; border-radius: 5px;}
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


def to_frame(rows):
    d = pd.DataFrame(rows)
    if not d.empty:
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d


def fetch_frames(country, models, target_date=""):
    """Fetch one frame per model; a failing model degrades to a warning
    instead of killing the page (e.g. classical refits still warming up)."""
    frames, failed = {}, []
    for m in models:
        try:
            frames[m] = to_frame(fetch_forecast(country, m, target_date))
        except Exception as e:
            failed.append((m, e))
    return frames, failed


def warn_failures(failed):
    if failed:
        st.warning(
            "Could not load " + ", ".join(MODEL_LABELS[m] for m, _ in failed)
            + f" — showing the rest ({failed[0][1]})."
        )


def trim_common(frames):
    """Restrict every frame to the time window ALL of them share.

    Country/model horizons can sit hours or a day apart when their data
    downloads ran at different times — plotting raw horizons draws the lines
    side by side instead of overlaid. Frames with no overlap are dropped
    (reported by the caller), never silently misplotted."""
    lo = max(d["timestamp"].min() for d in frames.values())
    hi = min(d["timestamp"].max() for d in frames.values())
    kept, dropped = {}, []
    for key, d in frames.items():
        t = d[(d["timestamp"] >= lo) & (d["timestamp"] <= hi)]
        if t.empty:
            dropped.append(key)
        else:
            kept[key] = t
    return kept, dropped


def align_frames(frames):
    """Trim to the shared window and warn about anything dropped."""
    frames = {k: v for k, v in frames.items() if not v.empty}
    if len(frames) > 1:
        frames, dropped = trim_common(frames)
        if dropped:
            st.warning(
                "No time overlap for " + ", ".join(str(k) for k in dropped)
                + " — their data horizon sits outside the shared window "
                  "(refresh that country's data downloads)."
            )
    return frames


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


def apply_layout(fig, height=440):
    fig.update_layout(
        template=template,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=height,
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        yaxis_title="EUR / MWh",
        legend={"orientation": "h", "y": 1.02},
        xaxis={"dtick": 3600000 * 3, "tickformat": "%H:%M"},
    )


def add_band(fig, d, color, label=""):
    if {"q10", "q90"} <= set(d.columns):
        fill = (f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},"
                f"{int(color[5:7], 16)},0.16)")
        fig.add_trace(go.Scatter(
            x=d["timestamp"], y=d["q90"], mode="lines",
            line={"width": 0}, hoverinfo="skip", name=f"q90 {label}".strip(),
        ))
        fig.add_trace(go.Scatter(
            x=d["timestamp"], y=d["q10"], mode="lines", line={"width": 0},
            fill="tonexty", fillcolor=fill, hoverinfo="skip",
            name=f"q10–q90 band {label}".strip(),
        ))


def add_actual(fig, d):
    if "actual_price_eur_mwh" in d.columns:
        a = d.dropna(subset=["actual_price_eur_mwh"])
        if not a.empty:
            fig.add_trace(go.Scatter(
                x=a["timestamp"], y=a["actual_price_eur_mwh"], mode="lines",
                line={"color": "black", "dash": "dash", "width": 1.8},
                name="actual", hovertemplate=HOVER,
            ))


st.title("⚡ EU Electricity Price Forecaster")

# ---------------- forecast tab ----------------
with st.spinner("Loading forecast…"):
    frames, failed = fetch_frames(country, models, past_date.strip())
warn_failures(failed)
frames = align_frames(frames)
if not frames:
    st.error(f"Backend unreachable. Is the FastAPI server running on {API_URL}?")
    st.stop()
models = [m for m in models if m in frames]

st.caption(
    f"Day-ahead electricity prices · {get_country(country)['name']} · "
    + " · ".join(MODEL_LABELS[m] for m in models)
)

first = frames[models[0]]
prices = first["predicted_price_eur_mwh"]
peak_hour = first.loc[prices.idxmax(), "timestamp"].strftime("%H:%M")
st.markdown(
    f"**Peak {prices.max():.1f} EUR/MWh** at {peak_hour} ({MODEL_LABELS[models[0]]}) · "
    f"min {prices.min():.1f} · mean {prices.mean():.1f}"
)

color = COUNTRY_COLORS.get(country, "#7f7f7f")
fig = go.Figure()
if show_ci:
    add_band(fig, first, color)
for m in models:
    style = {"color": color, "width": 2.6, "dash": "solid"} if m == "tft" else MODEL_LINE[m]
    fig.add_trace(go.Scatter(
        x=frames[m]["timestamp"], y=frames[m]["predicted_price_eur_mwh"],
        mode="lines+markers" if m == "tft" else "lines",
        line=style, marker={"size": 5} if m == "tft" else None,
        name=MODEL_LABELS[m], hovertemplate=HOVER,
    ))
add_actual(fig, first)
apply_layout(fig)
st.plotly_chart(fig, width="stretch")

base_idx = first["timestamp"]
display = pd.DataFrame({"Time": base_idx.dt.strftime("%Y-%m-%d %H:%M")})
for m in models:
    display[MODEL_LABELS[m]] = (
        frames[m].set_index("timestamp")["predicted_price_eur_mwh"]
        .reindex(base_idx).round(2).tolist()
    )
for q in ("q10", "q90"):
    if show_ci and q in first.columns:
        display[f"TFT {q}"] = first[q].round(2).tolist()
if "actual_price_eur_mwh" in first.columns:
    display["Actual (EUR/MWh)"] = first["actual_price_eur_mwh"].round(2).tolist()
with st.expander("Hourly forecast table", expanded=False):
    st.dataframe(display, width="stretch", height=320)

# ---------------- compare + backtest + benchmark tabs ----------------
tab_compare, tab_backtest, tab_bench = st.tabs(
    ["Compare countries", "Backtest days", "Benchmark"]
)

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
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            payload = None
        if payload:
            cmp_frames = {
                e["country"]: to_frame(e["forecast"])
                for e in payload.get("forecasts", [])
            }
            cmp_frames = align_frames(cmp_frames)
            skipped = list(payload.get("skipped", []))
            for key, d in cmp_frames.items():
                if d.empty:
                    skipped.append({"country": key, "reason": "no shared time window"})
            cmp_frames = {k: v for k, v in cmp_frames.items() if not v.empty}
            if skipped:
                st.info("; ".join(f"{s['country']}: {s['reason']}" for s in skipped))
            if cmp_frames:
                fig2 = go.Figure()
                table, table_idx = None, None
                for c, d in cmp_frames.items():
                    col = COUNTRY_COLORS.get(c, "#7f7f7f")
                    if cmp_model == "tft":
                        add_band(fig2, d, col, label=c)
                    fig2.add_trace(go.Scatter(
                        x=d["timestamp"], y=d["predicted_price_eur_mwh"],
                        mode="lines", line={"color": col, "width": 2.4},
                        name=COUNTRIES[c]["name"], hovertemplate=HOVER,
                    ))
                    if table is None:
                        table_idx = d["timestamp"]
                        table = pd.DataFrame(
                            {"Time": table_idx.dt.strftime("%H:%M")}
                        )
                    table[COUNTRIES[c]["name"]] = (
                        d.set_index("timestamp")["predicted_price_eur_mwh"]
                        .reindex(table_idx).round(2).tolist()
                    )
                apply_layout(fig2, height=420)
                st.plotly_chart(fig2, width="stretch")
                if table is not None:
                    with st.expander("Hourly comparison table", expanded=False):
                        st.dataframe(table, width="stretch", height=300)

with tab_backtest:
    st.caption(
        "Forecast vs. actual on the 5 most recent complete days. Every served "
        "model was trained before these dates — this is an honest "
        "out-of-sample error read, not a training fit."
    )
    try:
        days = _get("/days", {"country": country, "n": 5}).get("days", [])
    except Exception as e:
        days = []
        st.warning(f"Could not load backtest days ({e}).")
    if days:
        day = st.segmented_control(
            "Day", days, default=days[-1],
            format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b"),
        ) or days[-1]
        bt_sel = st.multiselect(
            "Models", list(MODEL_LABELS), default=list(MODEL_LABELS),
            format_func=MODEL_LABELS.get, key="bt_models",
        )
        if not bt_sel:
            st.info("Pick at least one model.")
        else:
            bt_show_ci = "tft" in bt_sel and st.checkbox(
                "Show q10–q90 band (TFT)", value=True, key="bt_show_ci"
            )
            with st.spinner(f"Scoring {day}…"):
                bt_frames, bt_failed = fetch_frames(country, bt_sel, day)
            warn_failures(bt_failed)
            bt_frames = align_frames(bt_frames)
            if bt_frames:
                bt_models = [m for m in bt_sel if m in bt_frames]
                bt_first = bt_frames[bt_models[0]]
                fig3 = go.Figure()
                if bt_show_ci and "tft" in bt_frames:
                    add_band(fig3, bt_frames["tft"], color)
                for m in bt_models:
                    style = ({"color": color, "width": 2.6, "dash": "solid"}
                             if m == "tft" else MODEL_LINE[m])
                    fig3.add_trace(go.Scatter(
                        x=bt_frames[m]["timestamp"],
                        y=bt_frames[m]["predicted_price_eur_mwh"],
                        mode="lines" if m != "tft" else "lines+markers",
                        line=style, marker={"size": 5} if m == "tft" else None,
                        name=MODEL_LABELS[m], hovertemplate=HOVER,
                    ))
                add_actual(fig3, bt_first)
                apply_layout(fig3)
                st.plotly_chart(fig3, width="stretch")

                rows = []
                for m in bt_models:
                    d = bt_frames[m]
                    if "actual_price_eur_mwh" not in d.columns:
                        continue
                    a = d.dropna(subset=["actual_price_eur_mwh"])
                    if a.empty:
                        continue
                    err = a["predicted_price_eur_mwh"] - a["actual_price_eur_mwh"]
                    rows.append({
                        "Model": MODEL_LABELS[m],
                        "MAE (EUR/MWh)": err.abs().mean(),
                        "RMSE (EUR/MWh)": (err ** 2).mean() ** 0.5,
                        "Bias (EUR/MWh)": err.mean(),
                        "Hours scored": len(a),
                    })
                if rows:
                    metrics = (
                        pd.DataFrame(rows).sort_values("MAE (EUR/MWh)")
                        .round(2).reset_index(drop=True)
                    )
                    metrics.loc[0, "Model"] += "  🏆"
                    st.caption(f"Day-ahead error on {day} — lower is better:")
                    st.dataframe(metrics, width="stretch", hide_index=True)
                else:
                    st.info("No actuals recorded for this day yet.")
    else:
        st.info("No complete actual-price days available yet — run the pipeline.")

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
        st.dataframe(b, width="stretch", height=420)
    else:
        st.info("No benchmark tables found — run the pipeline first.")
