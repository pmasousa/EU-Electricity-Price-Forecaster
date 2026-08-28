# Static demo — EU Electricity Price Forecaster

This branch hosts a **frozen, backend-free demo** of the dashboard for GitHub
Pages. It exists only on this branch — `main` carries the real project.

What it is:

- `index.html` — a static replica of the Streamlit dashboard (Plotly from
  CDN; no framework, no build step).
- `demo_data.json` — a one-time snapshot from the real serving API: the
  current 24h forecast and the five most recent out-of-sample days, for
  every country and model (TFT / Linear Regression / LightGBM), plus the
  walk-forward benchmark table.

The page makes **no requests** except loading `demo_data.json`. No model
inference, no arbitrary dates — the countries, models and days selectable
in the UI are exactly the ones frozen in the snapshot.

## Regenerate the snapshot

```bash
python -m src.api.main        # terminal 1 — serving API on :8000 (repo root)
python generate_snapshot.py   # terminal 2 — rewrites ./demo_data.json
git add demo_data.json && git commit -m "refresh demo snapshot" && git push
```

## Enable the page

Repo **Settings → Pages → Deploy from a branch → `gh-pages` / root**.
Note: GitHub Pages on a *private* repo requires GitHub Pro; on a public
repo it is free.

Data: day-ahead prices via Energy-Charts (EPEX SPOT / ENTSO-E
transparency), weather via Open-Meteo. Forecast values were produced by
the models trained in the main repo.
