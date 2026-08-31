# Release notes — v1.0.0

**Title:** `v1.0.0 — Day-ahead electricity price forecasting, end to end`

---

Probabilistic day-ahead electricity price forecasting for Portugal, Spain and Switzerland — data pipelines, walk-forward benchmarking, model serving and a live dashboard, all config-driven per country.

## What's inside

- **Walk-forward benchmark** — one shared harness scores every model on identical splits and covariates (load lagged 24h/168h — no realized-load leakage at gate closure), across all 24 forecast hours on ~3 years of hourly data per market
- **Models** — Temporal Fusion Transformer with quantile bands (pinball loss + coverage), Linear Regression, LightGBM, and naive persistence baselines for context
- **Multi-model serving** — FastAPI with a `model` parameter (`tft | lr | lgbm`), self-contained per-country serving bundles, retroactive date queries
- **Interactive dashboard** — Streamlit + Plotly: model overlays, three-country comparison, an out-of-sample day replay (`t-4 … Today | Tomorrow`) with per-model error tables, dark mode
- **Docker** — `train` (GPU) and `serve` profiles, verified end-to-end
- **Quality gates** — contract tests (quantile monotonicity, API schema, scalers-fit-on-train-only), ruff, CI on push/PR

## Headline results (8-week holdout, EUR/MWh)

| Market | Best point model | rMAE | Probabilistic winner |
|---|---|---|---|
| PT | Linear Regression | 0.90 | quantile LightGBM |
| ES | Linear Regression | 0.88 | quantile LightGBM |
| CH | LightGBM | 0.78 | **TFT (pinball 6.49)** |

Full protocol, tables and analysis in the [README](https://github.com/pmasousa/EU-Electricity-Price-Forecaster#-benchmark--day-ahead-walk-forward).

**Try it:** static demo at [pmasousa.github.io/EU-Electricity-Price-Forecaster](https://pmasousa.github.io/EU-Electricity-Price-Forecaster/) · run locally with `docker compose --profile serve up api ui`
