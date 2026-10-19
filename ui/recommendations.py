"""Model-estimated interventions.

Each *lever* is a what-if change to one employee (pay +15 %, no overtime, ...).
The trained model re-scores the changed profile, and the drop in predicted risk
is the lever's estimated effect. These are the model's learned associations,
not causal guarantees, and the UI says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.insights import describe_value
from src.predict import AttritionPredictor
from src.utils import FEATURE_COLUMNS

MIN_USEFUL_DROP = 0.02  # ignore levers that move the score by less than 2 points


@dataclass(frozen=True)
class Lever:
    key: str
    title: str          # shown as "Intervention suggested: <title>"
    action: str         # what changes in the profile
    icon: str
    applies: Callable[[pd.DataFrame], pd.Series]
    apply: Callable[[pd.DataFrame], pd.DataFrame]


def _with(frame: pd.DataFrame, **columns) -> pd.DataFrame:
    changed = frame.copy()
    for column, values in columns.items():
        changed[column] = values
    return changed


LEVERS: list[Lever] = [
    Lever("pay", "Compensation review", "Raise monthly pay by 15%", "hand-coins",
          lambda f: f["MonthlyIncome"] < 9_000_000,
          lambda f: _with(f, MonthlyIncome=(f["MonthlyIncome"] * 1.15).round(-2))),
    Lever("overtime", "Workload rebalance", "Remove regular overtime", "timer",
          lambda f: f["OverTime"] == "Yes",
          lambda f: _with(f, OverTime="No")),
    Lever("engagement", "Manager check-in", "Lift job satisfaction by 1 point (growth-plan conversation)", "heart-handshake",
          lambda f: f["JobSatisfaction"] < 5,
          lambda f: _with(f, JobSatisfaction=f["JobSatisfaction"] + 1)),
    Lever("balance", "Flexible schedule", "Lift work-life balance by 1 point", "scale",
          lambda f: f["WorkLifeBalance"] < 5,
          lambda f: _with(f, WorkLifeBalance=f["WorkLifeBalance"] + 1)),
    Lever("remote", "Hybrid work option", "Halve the commute with remote days", "house-wifi",
          lambda f: f["DistanceFromHome"] >= 10,
          lambda f: _with(f, DistanceFromHome=(f["DistanceFromHome"] / 2).round(0))),
]
LEVER_BY_KEY = {lever.key: lever for lever in LEVERS}


def lever_effects(predictor: AttritionPredictor, frame: pd.DataFrame) -> pd.DataFrame:
    """Change in predicted risk (new minus current, so negative is better) per lever.

    One column per lever, NaN where the lever does not apply to that person.
    """
    features = frame[FEATURE_COLUMNS].reset_index(drop=True)
    current = predictor.predict_frame(features)
    effects = pd.DataFrame(np.nan, index=range(len(features)), columns=[lever.key for lever in LEVERS])
    for lever in LEVERS:
        mask = lever.applies(features).to_numpy()
        if not mask.any():
            continue
        changed = lever.apply(features[mask])
        effects.loc[mask, lever.key] = predictor.predict_frame(changed) - current[mask]
    effects.index = frame.index
    return effects


def ranked_levers(effects_row: pd.Series, min_drop: float = MIN_USEFUL_DROP) -> list[tuple[Lever, float]]:
    """Levers that help this person, biggest drop first: [(lever, delta)] with delta below zero."""
    useful = effects_row.dropna()
    useful = useful[useful <= -min_drop].sort_values()
    return [(LEVER_BY_KEY[key], float(delta)) for key, delta in useful.items()]


def driver_labels(row: pd.Series, impacts: pd.Series, k: int = 3, min_impact: float = 0.15) -> list[str]:
    """Readable names of the features pushing this person's risk up the most."""
    positive = impacts[impacts > min_impact].sort_values(ascending=False).head(k)
    return [describe_value(feature, row[feature]) for feature in positive.index]


def apply_lever_to_profile(lever: Lever, profile: dict) -> dict:
    """Apply a lever to a single profile dict (used by the what-if lab's Apply button)."""
    frame = pd.DataFrame([profile])
    return lever.apply(frame).iloc[0].to_dict()
