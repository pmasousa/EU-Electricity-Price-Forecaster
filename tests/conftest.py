"""Shared pytest setup for the electricity price forecaster test suite.

- Makes the repo root importable so ``import src.*`` works from any CWD.
- Pins the CWD to the repo root for every test: the application code
  (``src/api/main.py``, ``src/data/dataset.py``) resolves model artifacts,
  processed data and reports through *relative* paths (``models/``,
  ``data/processed/``, ``reports/``), so the tests must run from the repo
  root regardless of where pytest was invoked.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch):
    """Run every test with the repo root as the working directory."""
    monkeypatch.chdir(REPO_ROOT)
