"""Build the frozen data snapshot for the static demo page (gh-pages).

Queries the local serving API once and writes demo_data.json. The demo page
ships with that file and makes no other requests: no model inference, no
date queries — countries, models and the 5 backtest days selectable in the
page are exactly the ones frozen here.

Regenerate after retraining or with fresher data:
    python -m src.api.main          # terminal 1 — the API on :8000
    python generate_snapshot.py     # terminal 2 — writes ./demo_data.json
Then commit demo_data.json to this branch.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone

API = "http://127.0.0.1:8000"
COUNTRIES = ["PT", "ES", "CH"]
MODELS = ["tft", "lr", "lgbm"]
N_DAYS = 5
OUT = sys.argv[1] if len(sys.argv) > 1 else "demo_data.json"


def get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=300) as r:
        return json.load(r)


def main():
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": (
            "Static demo — forecasts pre-computed by the repo's serving API "
            "and frozen. No model runs on this page. Prices: Energy-Charts "
            "(EPEX SPOT / ENTSO-E transparency). Weather: Open-Meteo."
        ),
        "countries": {},
        "metrics": get("/metrics")["metrics"],
    }
    for c in COUNTRIES:
        days = get(f"/days?country={c}&n={N_DAYS}")["days"]
        entry = {"forecast": {}, "backtest": {d: {} for d in days}}
        for m in MODELS:
            print(f"{c} {m}: forecast ...", flush=True)
            entry["forecast"][m] = get(
                f"/predict?country={c}&model={m}")["forecast"]
            for d in days:
                print(f"{c} {m} {d} ...", flush=True)
                entry["backtest"][d][m] = get(
                    f"/predict?country={c}&model={m}&target_date={d}")["forecast"]
        data["countries"][c] = entry

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = len(json.dumps(data)) // 1024
    print(f"wrote {OUT}: {size_kb} KB | "
          + " | ".join(f"{c}: {len(v['backtest'])} days" for c, v in data["countries"].items()))


if __name__ == "__main__":
    main()
