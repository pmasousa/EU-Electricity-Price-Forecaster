"""Schema contract for GET /predict — mirrors the ACTUAL response built by
``src/api/main.py::_forecast_one``:

    {
        "country": str,                 # e.g. "CH"
        "country_name": str,            # e.g. "Switzerland" (from src.config)
        "forecast": [                   # exactly 24 hourly records
            {
                "timestamp": str,       # ISO timestamp, hourly steps
                "predicted_price_eur_mwh": float,   # the q50 median
                "q10": float,
                "q90": float,
                # "actual_price_eur_mwh": float  <- only when target_date is
                # given AND actuals exist for that day; absent on a plain
                # next-day forecast.
            },
            ...
        ]
    }

The quantiles come from a probabilistic TFT predict (num_samples=100),
inverse-transformed, then ``pred_real.quantile(0.1/0.5/0.9)`` — so q10 <= q50
<= q90 must hold per point.

Startup notes: the app loads country models in a FastAPI lifespan. Constructing
``TestClient(app)`` inside a ``with`` block triggers it. Missing artifacts are
caught by the module-level skipif BEFORE the client is constructed. The API
loads on CPU (``map_location="cpu"`` + trainer_params patch), so this test
never touches the GPU. No uvicorn server is started.

KNOWN ENVIRONMENT ISSUE (see ``_install_checkpoint_compat_shims``): the CH
artifacts were produced under different library versions and do NOT load
cleanly with the installed darts 0.45.0 + pytorch_lightning 2.5.2. The shims
below are process-local (test-only, no repo files touched) and let the
contract test exercise the REAL endpoint against the REAL model instead of
skipping. If the shims ever stop being enough, the client fixture skips with
the underlying load error surfaced.
"""

from numbers import Real
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

REPO_ROOT = Path(__file__).resolve().parents[1]

# What /predict needs to answer 200 for CH: the three model artifacts loaded at
# startup plus the processed features file it reads at request time.
REQUIRED_ARTIFACTS = (
    REPO_ROOT / "models" / "serving_CH" / "tft_model.pt",
    REPO_ROOT / "models" / "serving_CH" / "scaler_target.pkl",
    REPO_ROOT / "models" / "serving_CH" / "scaler_cov.pkl",
    REPO_ROOT / "models" / "serving_CH" / "config.json",
    REPO_ROOT / "data" / "processed" / "features_CH.csv",
)

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED_ARTIFACTS),
    reason="CH serving bundle missing (models/serving_CH/* or "
           "data/processed/features_CH.csv) — run build_serving.py first",
)


def _is_real_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _install_checkpoint_compat_shims():
    """Test-process-only compatibility layer for the saved TFT checkpoint.

    models/tft_model_CH.pt cannot be deserialized as-is with the installed
    darts 0.45.0 / pytorch_lightning 2.5.2 (torch 2.14.0.dev) for three
    independent reasons, verified against this repo:

    1. The pickled trainer state references
       ``pytorch_lightning.callbacks.early_stopping.EarlyStoppingReason``,
       which PL 2.5.x removed -> AttributeError during torch.load.
    2. The pickled model references ``darts.models.components.tft_submodels``,
       which moved to ``darts.models.forecasting.tft_submodels`` in darts
       0.45 -> ModuleNotFoundError.
    3. ``src/api/main.py`` calls ``TFTModel.load(..., weights_only=False)``;
       darts 0.45 forwards **kwargs into PL's ``load_from_checkpoint``, where
       ``weights_only`` is merged into the constructor kwargs and rejected by
       ``PLForecastingModule.__init__`` (it would also collide with the
       hardcoded ``weights_only=False`` in darts' own ``torch.load`` call)
       -> TypeError.

    Each shim below is additive and only bridges the version drift; nothing on
    disk and nothing under src/ is modified. The proper fix is to retrain (or
    re-save) the model under the installed library versions.
    """
    import enum
    import sys

    # (1) restore the removed PL enum with the members the pickle asks for.
    import pytorch_lightning.callbacks.early_stopping as es

    if not hasattr(es, "EarlyStoppingReason"):

        class EarlyStoppingReason(enum.IntEnum):
            NOT_STOPPED = 0
            PATIENCE_EXCEEDED = 1
            MIN_EPOCHS_REACHED = 2
            VAL_LOSS_NOT_FINITE = 3

        es.EarlyStoppingReason = EarlyStoppingReason

    # (2) alias the old darts module path to its new location.
    import darts.models.components as darts_components
    import darts.models.forecasting.tft_submodels as tft_submodels_new

    sys.modules.setdefault(
        "darts.models.components.tft_submodels", tft_submodels_new
    )
    if not hasattr(darts_components, "tft_submodels"):
        darts_components.tft_submodels = tft_submodels_new

    # (3) strip ``weights_only`` from the checkpoint-load kwargs.
    from darts.models.forecasting.torch_forecasting_model import (
        TorchForecastingModel,
    )

    if not getattr(TorchForecastingModel, "_weights_only_stripped", False):
        _orig_load_from_checkpoint = TorchForecastingModel._load_from_checkpoint

        def _load_from_checkpoint_no_weights_only(self, *args, **kwargs):
            kwargs.pop("weights_only", None)
            return _orig_load_from_checkpoint(self, *args, **kwargs)

        _load_from_checkpoint_no_weights_only._weights_only_stripped = True
        TorchForecastingModel._load_from_checkpoint = (
            _load_from_checkpoint_no_weights_only
        )


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    _install_checkpoint_compat_shims()

    from src.api.main import MODELS, app

    with TestClient(app) as test_client:  # context manager -> lifespan runs
        if "CH" not in MODELS:
            pytest.skip(
                "CH model artifacts exist but failed to load even with the "
                "compatibility shims — the /predict schema contract cannot be "
                "tested. Retrain the model under the installed library "
                "versions (see _install_checkpoint_compat_shims docstring)."
            )
        yield test_client


@pytest.fixture(scope="module")
def predict_payload(client):
    resp = client.get("/predict", params={"country": "CH"})
    assert resp.status_code == 200, f"/predict failed: {resp.status_code} {resp.text}"
    return resp.json()


def test_predict_top_level_schema(predict_payload):
    payload = predict_payload
    assert set(payload.keys()) == {"country", "country_name", "forecast"}
    assert payload["country"] == "CH"
    assert payload["country_name"] == "Switzerland"
    assert isinstance(payload["country_name"], str)


def test_predict_forecast_records(predict_payload):
    forecast = predict_payload["forecast"]
    assert isinstance(forecast, list)
    assert len(forecast) == 24

    timestamps = []
    for record in forecast:
        # Exactly the documented fields on a plain (no target_date) forecast:
        # the actual price key must not appear because no target_date was given.
        assert set(record.keys()) == {
            "timestamp",
            "predicted_price_eur_mwh",
            "q10",
            "q90",
        }

        assert isinstance(record["timestamp"], str)
        timestamps.append(pd.Timestamp(record["timestamp"]))

        for field in ("predicted_price_eur_mwh", "q10", "q90"):
            assert _is_real_number(record[field]), (
                f"{field} is not a number: {record[field]!r}"
            )

        # Quantile bands must be ordered per point; the median price is q50.
        q10, q50, q90 = record["q10"], record["predicted_price_eur_mwh"], record["q90"]
        assert q10 <= q50, f"{record['timestamp']}: q10 ({q10}) > q50 ({q50})"
        assert q50 <= q90, f"{record['timestamp']}: q50 ({q50}) > q90 ({q90})"

    # 24 consecutive hourly timestamps, no gaps.
    ts = pd.DatetimeIndex(timestamps)
    assert len(set(ts)) == 24
    diffs = pd.Series(ts).diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_predict_with_actuals_includes_actual_field(client):
    """With a target_date inside the data, records gain actual_price_eur_mwh."""
    import numpy as np

    features = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "features_CH.csv",
        parse_dates=[0],
        index_col=0,
    )
    features.index = pd.to_datetime(features.index).tz_localize(None)
    if features.index[-1] - features.index[0] < pd.Timedelta(days=10):
        pytest.skip("features_CH.csv too short for a historical target_date")
    # /predict rejects target dates whose 168h history window doesn't carry
    # valid load lags (target_start must be strictly after data_start + 336h);
    # data may start at any hour of the day, so advance past the gate day by day.
    candidate = (features.index[0] + pd.Timedelta(hours=337)).normalize()
    gate = features.index[0] + pd.Timedelta(hours=336)
    while candidate <= gate:
        candidate += pd.Timedelta(days=1)
    # The 24h actual window must also fit inside the data.
    if candidate + pd.Timedelta(hours=23) > features.index[-1]:
        pytest.skip("features_CH.csv too short for a historical target_date")
    target_date = str(candidate.date())

    resp = client.get("/predict", params={"country": "CH", "target_date": target_date})
    assert resp.status_code == 200, f"{resp.status_code} {resp.text}"
    forecast = resp.json()["forecast"]
    assert len(forecast) == 24
    with_actual = [r for r in forecast if "actual_price_eur_mwh" in r]
    assert with_actual, "expected actual_price_eur_mwh for a fully covered target_date"
    for record in with_actual:
        assert _is_real_number(record["actual_price_eur_mwh"])
        assert np.isfinite(record["actual_price_eur_mwh"])
