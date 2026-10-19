"""Cached resources and data loading shared by every page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src import database
from src.database import EmployeeDB, open_database
from src.predict import AttritionPredictor
from src.utils import MODEL_PATH, risk_level
from ui.recommendations import lever_effects


@st.cache_resource(show_spinner=False)
def get_db() -> EmployeeDB:
    # Read DB_PATH from the module at call time so tests / env overrides can redirect it.
    return open_database(database.DB_PATH)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> AttritionPredictor:
    return AttritionPredictor(MODEL_PATH)


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def data_stamp() -> tuple[int, int]:
    """Changes whenever the database file or the model file changes (cache key)."""
    return _mtime_ns(Path(database.DB_PATH)), _mtime_ns(MODEL_PATH)


def last_updated() -> datetime | None:
    """When the employee database was last written (the file's modification time)."""
    ns = _mtime_ns(Path(database.DB_PATH))
    return datetime.fromtimestamp(ns / 1e9) if ns else None


@st.cache_data(show_spinner=False, max_entries=4)
def load_workforce(stamp: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All employees with RiskScore / RiskLevel, plus per-lever risk change for active staff.

    ``stamp`` is only the cache key: the data is re-read when the database or model file changes.
    """
    db, predictor = get_db(), get_predictor()
    employees = db.list()
    if employees.empty:
        scored = employees.assign(RiskScore=pd.Series(dtype=float), RiskLevel=pd.Series(dtype=str))
        return scored, pd.DataFrame()
    scored = employees.copy()
    scored["RiskScore"] = predictor.predict_frame(scored)
    scored["RiskLevel"] = scored["RiskScore"].map(risk_level)
    active = scored[scored["Attrition"] == "No"]
    effects = lever_effects(predictor, active) if len(active) else pd.DataFrame()
    return scored, effects


def flash(message: str, kind: str = "success") -> None:
    """Queue a toast that survives the st.rerun() after a dialog closes."""
    st.session_state["flash"] = (kind, message)


def refresh_data() -> None:
    load_workforce.clear()
