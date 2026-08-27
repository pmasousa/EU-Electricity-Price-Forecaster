"""Country / bidding-zone registry for the multi-country electricity price forecaster.

Single source of truth for the countries this project can forecast. Adding a new
country is a one-line change in ``COUNTRIES`` below — every layer (download,
features, dataset, training, API, UI) reads from here.

Data sources (no API keys required):
    - Day-ahead prices & actual load: Energy-Charts API (api.energy-charts.info),
      which aggregates EPEX SPOT / ENTSO-E transparency data. Verified working
      for CH, PT, ES with an identical JSON schema.
    - Weather: Open-Meteo Archive API (archive-api.open-meteo.com).

Notes:
    - Prices are returned in EUR/MWh for every zone (including Switzerland), so
      all currency labels in the project use EUR/MWh.
    - Some bidding zones (e.g. ES, PT) publish day-ahead prices at 15-minute
      resolution while others (e.g. CH) are hourly. The download layer resamples
      everything to a common hourly grid so downstream features stay aligned.
"""

# Each entry maps a short country code (used in file names and API params) to its
# metadata: human-readable name, Energy-Charts bidding-zone + country codes, and
# the representative weather station coordinates + timezone.
COUNTRIES = {
    "CH": {
        "name": "Switzerland",
        "bzn": "CH",
        "country": "ch",
        "lat": 47.3667,
        "lon": 8.55,
        "tz": "Europe/Zurich",
    },
    "PT": {
        "name": "Portugal",
        "bzn": "PT",
        "country": "pt",
        "lat": 38.7223,
        "lon": -9.1393,
        "tz": "Europe/Lisbon",
    },
    "ES": {
        "name": "Spain",
        "bzn": "ES",
        "country": "es",
        "lat": 40.4168,
        "lon": -3.7038,
        "tz": "Europe/Madrid",
    },
}

# Countries processed by default when running the full pipeline or API startup.
DEFAULT_COUNTRIES = ["CH", "PT", "ES"]

# Country served when /predict is called without an explicit country (used
# only if that country's serving bundle is loaded; otherwise the first
# loaded country wins).
DEFAULT_COUNTRY = "PT"


def get_country(code: str) -> dict:
    """Return the metadata dict for a country code, raising a clear error if unknown."""
    if code not in COUNTRIES:
        raise ValueError(
            f"Unknown country code '{code}'. Known codes: {sorted(COUNTRIES)}"
        )
    return COUNTRIES[code]


def parse_countries(spec: str | None = None) -> list[str]:
    """Parse a comma-separated country list (e.g. 'CH,PT,ES') into validated codes.

    Returns DEFAULT_COUNTRIES when ``spec`` is empty/None. Unknown codes raise.
    """
    if not spec:
        return list(DEFAULT_COUNTRIES)
    codes = [c.strip().upper() for c in spec.split(",") if c.strip()]
    for c in codes:
        get_country(c)  # validates
    return codes
