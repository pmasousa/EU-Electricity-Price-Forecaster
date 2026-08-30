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

COUNTRY_COLORS = {"PT": "#c0392b", "ES": "#2e6fa3", "CH": "#3d8a5c"}
MODEL_LABELS = {
    "tft": "TFT",
    "lr": "Linear Regression",
    "lgbm": "LightGBM",
}

# Line style per overlaid model; the TFT keeps the country color, the
# classical models get fixed hues so they read across countries.
MODEL_LINE = {
    "lr": {"color": "#d98e2b", "dash": "dash", "width": 2.0},
    "lgbm": {"color": "#7d5ba6", "dash": "dot", "width": 2.0},
}

HOVER = "%{x|%a %d %b, %H:%M} — %{y:.1f} EUR/MWh<extra></extra>"

# --- base design system (always on): Inter, hidden chrome, cards, stats ---
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"], button, input, select, label {
  font-family: 'Inter', 'Source Sans Pro', system-ui, sans-serif !important;
}
/* hide streamlit chrome: toolbar, running-man, decoration gradient */
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stLogoSpacer"] {
  visibility: hidden !important; height: 0 !important;}
[data-testid="stHeader"] {height: 0.6rem !important; background: transparent !important;}
/* page title block */
.app-head {display: flex; align-items: center; gap: 14px; margin: 6px 0 2px;}
.app-mark {
  width: 40px; height: 40px; border-radius: 10px; flex: 0 0 40px;
  background: linear-gradient(135deg, #1f8a4c, #166b3a);
  color: #fff; font-size: 20px; display: flex; align-items: center;
  justify-content: center; box-shadow: 0 2px 6px rgba(22, 107, 58, 0.25);}
.app-title {font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em;
  color: #17251d; line-height: 1.15;}
.app-sub {font-size: 0.82rem; color: #5f6b64; font-weight: 500;
  letter-spacing: 0.01em;}
/* stat strip */
.stat-strip {display: flex; gap: 12px; margin: 10px 0 6px; flex-wrap: wrap;}
.stat {flex: 1; min-width: 130px; background: #f6faf7; border: 1px solid #e2ebe5;
  border-radius: 10px; padding: 10px 14px;}
.stat .k {font-size: 0.68rem; font-weight: 600; letter-spacing: 0.09em;
  text-transform: uppercase; color: #6b7a70; margin-bottom: 2px;}
.stat .v {font-size: 1.25rem; font-weight: 700; color: #17251d;
  font-variant-numeric: tabular-nums;}
.stat .u {font-size: 0.72rem; color: #8a978d; font-weight: 500;}
.stat.hero {background: #eef7f1; border-color: #cfe5d8;}
.stat.hero .v {color: #166b3a;}
/* sidebar structure */
[data-testid="stSidebar"] .eyebrow {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #8a978d; margin: 18px 0 2px;}
[data-testid="stSidebar"] .eyebrow:first-child {margin-top: 4px;}
[data-testid="stSidebarUserContent"] {padding-top: 0.5rem;}
/* chart cards */
.stPlotlyChart {background: #ffffff; border: 1px solid #e6ece8;
  border-radius: 12px; padding: 10px 6px 2px 2px;}
/* tables sit in cards too */
[data-testid="stExpander"] {border: 1px solid #e6ece8 !important;
  border-radius: 10px !important; overflow: hidden;}
/* footer */
.app-foot {border-top: 1px solid #e6ece8; margin-top: 26px; padding-top: 10px;
  color: #8a978d; font-size: 0.75rem; display: flex; justify-content: space-between;}
.app-foot b {color: #5f6b64; font-weight: 600;}
"""

DARK_CSS = """
<style>
/* base surfaces */
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
  background: #0b120f !important; color: #e6edf3 !important;}
[data-testid="stSidebar"] {background: #101915 !important;
  border-color: #22302a !important;}
[data-testid="stSidebar"] * {color: #c7d4cb !important;}
[data-testid="stSidebar"] .eyebrow {color: #6f8177 !important;}
.stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp summary, .stApp td, .stApp th {color: #e6edf3 !important;}
[data-testid="stCaptionContainer"] {color: #9fb3a8 !important;}
/* title block inverts */
.app-title {color: #eef4f0 !important;}
.app-sub {color: #9fb3a8 !important;}
.app-mark {box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);}
/* stat strip dark */
.stat {background: #121c17 !important; border-color: #22302a !important;}
.stat .k {color: #6f8177 !important;}
.stat .v {color: #eef4f0 !important;}
.stat .u {color: #8fa398 !important;}
.stat.hero {background: #14231b !important; border-color: #2c4436 !important;}
.stat.hero .v {color: #4ecb82 !important;}
/* input + dropdown controls */
[data-testid="stMultiSelect"] .react-aria-ComboBox > div,
[data-testid="stSelectbox"] .react-aria-ComboBox > div,
[data-testid="stTextInput"] input {
  background-color: #16211c !important; color: #e6edf3 !important;
  border-color: #2c3a32 !important;}
[data-testid="stMultiSelect"] input,
[data-testid="stSelectbox"] input {color: #e6edf3 !important;}
[data-testid="stSelectbox"] input {background-color: transparent !important;}
.stApp input::placeholder {color: #9fb3a8 !important;}
[data-testid="stTextInputRootElement"] {
  background-color: #16211c !important; border-color: #2c3a32 !important;}
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
.stButtonGroup button[aria-pressed="true"] {color: #4ecb82 !important;}
/* tabs */
[data-testid="stTabs"] [role="tab"] {color: #9fb3a8 !important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color: #eef4f0 !important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background-color: #4ecb82 !important;}
/* expanders + alerts */
[data-testid="stExpander"], [data-testid="stExpanderDetails"] {
  background-color: #121c17 !important; border-color: #22302a !important;}
[data-testid="stExpander"] summary {
  background-color: #16211c !important; color: #e6edf3 !important;}
[data-testid="stAlert"] {background-color: #16211c !important; color: #e6edf3 !important;}
/* header buttons */
[data-testid="stDeploymentButton"], [data-testid="stMainMenu"] {
  background-color: #16211c !important; color: #9fb3a8 !important;
  border: 1px solid #2c3a32 !important;}
/* tables: st.table renders real DOM cells (unlike st.dataframe's canvas
   grid, whose theme comes from JS and cannot be reached by CSS at all) */
.stApp table {border-collapse: collapse !important;}
.stApp table th, .stApp table td {
  background-color: #121c17 !important; color: #e6edf3 !important;
  border-color: #22302a !important;}
.stApp table thead th, .stApp table th[scope="row"] {
  background-color: #16211c !important; color: #9fb3a8 !important;}
.stApp table td {text-align: right !important;}
/* chart card dark */
.stPlotlyChart {background: #101915 !important; border-color: #22302a !important;}
/* polish */
::-webkit-scrollbar {width: 10px; height: 10px;}
::-webkit-scrollbar-track {background: #0b120f;}
::-webkit-scrollbar-thumb {background: #22302a; border-radius: 5px;}
/* footer dark */
.app-foot {border-color: #22302a !important; color: #6f8177 !important;}
.app-foot b {color: #9fb3a8 !important;}
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


def fmt_table(df):
    """Render floats at 2dp (st.table otherwise pads to 132.3300)."""
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda v: f"{v:.2f}")
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
    dark = st.toggle("Dark mode", value=False)
    st.markdown('<div class="eyebrow">Market</div>', unsafe_allow_html=True)
    country_label = st.selectbox(
        "Country",
        [f"{c} — {COUNTRIES[c]['name']}" for c in DEFAULT_COUNTRIES],
        index=0,
        label_visibility="collapsed",
    )
    country = country_label.split(" — ")[0].strip()
    st.markdown('<div class="eyebrow">Models</div>', unsafe_allow_html=True)
    models = st.multiselect(
        "Overlay models (click the legend to toggle lines)",
        list(MODEL_LABELS), default=["tft", "lr", "lgbm"],
        format_func=MODEL_LABELS.get, label_visibility="collapsed",
    )
    if not models:
        st.info("Pick at least one model.")
        st.stop()
    st.markdown('<div class="eyebrow">Display</div>', unsafe_allow_html=True)
    show_ci = "tft" in models and st.checkbox("Quantile band (TFT q10–q90)", value=True)
    past_date = st.text_input(
        "Replay a past day", placeholder="YYYY-MM-DD",
        label_visibility="collapsed",
    )
    if past_date and not past_date.strip():
        past_date = ""

if dark:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown(BASE_CSS, unsafe_allow_html=True)


def apply_layout(fig, height=440):
    grid = "#27352d" if dark else "#e7ede9"
    tick = "#8fa398" if dark else "#6b7a70"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        font={"family": "Inter, 'Source Sans Pro', sans-serif", "size": 12,
              "color": "#c7d4cb" if dark else "#33413a"},
        margin={"l": 14, "r": 18, "t": 26, "b": 14},
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis={"title": {"text": "EUR / MWh", "font": {"size": 11}},
               "gridcolor": grid, "gridwidth": 1, "zeroline": False,
               "tickfont": {"color": tick, "size": 11}},
        xaxis={"gridcolor": "rgba(0,0,0,0)", "dtick": 3600000 * 3,
               "tickformat": "%H:%M", "tickfont": {"color": tick, "size": 11}},
        legend={"orientation": "h", "y": 1.06, "bgcolor": "rgba(0,0,0,0)",
                "font": {"size": 11.5, "color": "#c7d4cb" if dark else "#33413a"}},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#16211c" if dark else "#ffffff",
                    "bordercolor": "#2c3a32" if dark else "#d9e5dd",
                    "font": {"family": "Inter, sans-serif", "size": 12,
                             "color": "#e6edf3" if dark else "#17251d"}},
    )


def stats_strip(d):
    """Four-metric strip replacing the old floating bold text line."""
    p = d["predicted_price_eur_mwh"]
    ts = d["timestamp"]
    peak_i = int(p.idxmax())
    cells = [
        ("Mean", f"{p.mean():.1f}", "EUR/MWh", False),
        ("Peak", f"{p.max():.1f}", ts.loc[peak_i].strftime("%H:%M"), True),
        ("Min", f"{p.min():.1f}", "EUR/MWh", False),
        ("Spread", f"{p.max() - p.min():.1f}", "EUR/MWh", False),
    ]
    html = '<div class="stat-strip">' + "".join(
        f'<div class="stat{" hero" if hero else ""}">'
        f'<div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="u">{u}</div></div>'
        for k, v, u, hero in cells
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def add_band(fig, d, color, label=""):
    if {"q10", "q90"} <= set(d.columns):
        fill = (f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},"
                f"{int(color[5:7], 16)},0.16)")
        # the q90 trace exists only to close the fill — keep it out of the
        # legend so the band shows a single entry (dupes were colliding)
        fig.add_trace(go.Scatter(
            x=d["timestamp"], y=d["q90"], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
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
                line={"color": "#e6edf3" if dark else "black",
                      "dash": "dash", "width": 1.8},
                name="actual", hovertemplate=HOVER,
            ))


st.markdown(
    '<div class="app-head"><div class="app-mark">⚡</div><div>'
    '<div class="app-title">EU Electricity Price Forecaster</div>'
    '<div class="app-sub">Day-ahead hourly prices · Portugal · Spain · Switzerland'
    " · probabilistic TFT + classical baselines</div></div></div>",
    unsafe_allow_html=True,
)

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
    f"{get_country(country)['name']} · "
    + " · ".join(MODEL_LABELS[m] for m in models)
    + (f" · replaying {past_date.strip()}" if past_date.strip() else "")
)

first = frames[models[0]]
stats_strip(first)

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
    st.table(fmt_table(display).set_index("Time"))

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
                        st.table(fmt_table(table).set_index("Time"))

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
                    st.table(fmt_table(metrics).set_index("Model"))
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
        for col in ("MAE (EUR/MWh)", "RMSE (EUR/MWh)", "rmae"):
            if col in b.columns:
                b[col] = pd.to_numeric(b[col], errors="coerce").round(2)
        st.caption("Walk-forward benchmark (8-week holdout, EUR/MWh) — ✓ marks the served model.")
        st.table(fmt_table(b).set_index("Model"))
    else:
        st.info("No benchmark tables found — run the pipeline first.")

st.markdown(
    '<div class="app-foot"><span><b>Data</b> Energy-Charts (EPEX SPOT / ENTSO-E) '
    "· Open-Meteo</span><span><b>Models</b> per-country TFT, Linear Regression, "
    "LightGBM — walk-forward day-ahead protocol</span></div>",
    unsafe_allow_html=True,
)
