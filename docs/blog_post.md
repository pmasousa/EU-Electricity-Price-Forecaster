# My electricity price benchmark lied to me three times

I have a repo that forecasts day-ahead electricity prices for Portugal, Spain
and Switzerland with a Temporal Fusion Transformer. For a while it also had a
benchmark table that said the TFT was ~7x worse than linear regression, and
LightGBM lost to everything. I believed that table. It was wrong three
different ways, and each way is worth knowing about, because none of them were
exotic — they were defaults, shortcuts and quiet API behaviors.

## The table I started with

| Model | Backtest MAE | Backtest RMSE |
|---|---|---|
| Linear Regression | 9.18 | 12.28 |
| LightGBM | 37.59 | 74.27 |
| TFT (rolling day-ahead) | 68.65 | 103.49 |

When both nonlinear models lose to a line, the comparison is broken, not the
models. That sentence was in my own notes for weeks before I acted on it.

## Lie #1: the model behind the number wasn't the model

The published TFT numbers came from a 2-epoch smoke-run artifact. The
baselines next to it had no epoch concept — they were fully trained. The fix
wasn't clever: immutable per-run folders (`reports/runs/<variant>/<timestamp>/`)
that keep the model, the table, the forecasts, the loss curve and the exact
data snapshot together. A number now always points at the artifact that
produced it. The 2-epoch model still existed on disk next to real ones; it
just had no label separating it from them.

## Lie #2: everyone got tomorrow's load for free

The feature table included actual grid load as a future covariate at forecast
time (`lags_future_covariates=[0]` in darts). Day-ahead prices are cleared
before the day happens — you don't know tomorrow's realized load any more than
you know tomorrow's price. Linear regression ate it happily: regressing price
on the actual load curve is nearly cheating, and the 9.18 was partly that.
The harness now lags load by 24h and 168h. Weather at forecast time stays,
disclosed as a day-ahead forecast proxy.

## Lie #3: "day-ahead MAE" measured one hour of the day

Darts' `historical_forecasts` defaults to `last_points_only=True`. With a 24h
horizon and daily stride, that silently scores only the 24th hour of each
forecast — always the same hour of the day, the calmest one. My naive
baselines were computed over all 1,344 holdout hours and the models over 56.
Same table, different denominators. One flag fixed it; a paired t-test on
equal-footing forecasts then showed the "LightGBM beats LR" ordering was noise
(t = 0.34 on 1,344 points).

## The harness rules that fell out

- One code path for every model: same data, same splits, same covariates,
  same walk-forward, `retrain=False`.
- Naive baselines in the table (persistence t-24, weekly t-168). If your model
  doesn't beat yesterday's curve, say so with a number.
- rMAE (error relative to persistence) instead of raw MAE for cross-market
  comparison, and pinball loss + coverage for the quantile bands.
- Every run archived: model, table, forecasts, loss curve, data snapshot.

## What the honest numbers say

After the fixes, on an 8-week holdout, ~3 years of hourly data per market:

| Model | PT rMAE | ES rMAE | CH rMAE |
|---|---|---|---|
| Linear Regression | **0.90** | **0.88** | 0.79 |
| LightGBM | 0.97 | 1.02 | **0.78** |
| TFT | 1.27 | 1.41 | 1.13 |

The classical models win the point forecast everywhere. That's not an insult —
a 186-feature autoregression is the LEAR-family benchmark this literature has
used for years, and it's hard to beat on calendar + weather + lagged load.
The TFT takes the probabilistic score on CH (pinball 6.49 vs 6.96) and loses
it on the Iberian markets. LightGBM drops below persistence on ES.
(Absolute MAE for the rMAE winners: PT 18.00, ES 17.36, CH 16.69 EUR/MWh —
the table is relative because raw MAE isn't comparable across markets with
different price levels.)

The most useful number in the repo is the ugly one: both model families let
31–50% of actual hours breach their q90 band, against a 10% calibration
target. The holdout sits in the 2026 price surge; nothing in this feature set
tracks a regime that steep. That's the v1.1 roadmap — neighboring-zone prices,
day-ahead load forecasts — with evidence instead of a hunch.

## Takeaways

1. A benchmark is code. It has bugs, defaults and stale artifacts, and it will
   testify against you with full confidence.
2. Leakage doesn't look like leakage. "Actual load" is a fine column — at the
   wrong lag it's the answer key.
3. Check what your library's defaults actually measure before you quote the
   output. `last_points_only` is one flag away from a 24x smaller evaluation.
4. Archive runs so numbers have provenance. Immutable folders are cheap;
   post-hoc archaeology isn't.

Repo: [EU-Electricity-Price-Forecaster](https://github.com/pmasousa/EU-Electricity-Price-Forecaster) —
CI, tests, the harness and every table referenced above are in it. The served
stack: per-country models behind a FastAPI API (`?model=tft|lr|lgbm`), a
Streamlit dashboard with forecast overlays, a three-country comparison, a
backtest view of recent out-of-sample days, and the walk-forward benchmark
table — all runnable under docker compose.
