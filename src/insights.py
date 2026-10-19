"""Model-agnostic explanations: why does an employee get this risk score?

Method ("what-if baseline"): for each input feature, replace the employee's
value with the typical value of the workforce (median / most common) and see
how much the predicted risk drops. Impact is measured in log-odds (not raw
probability) so it stays informative even when a score is close to 0% or 100%. A positive impact means
the feature pushes this employee's risk UP relative to a typical colleague; a
negative impact means it pulls risk down. It works for any scikit-learn
pipeline, unlike coefficient- or tree-specific explanations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.predict import AttritionPredictor
from src.utils import FEATURE_COLUMNS

DEFAULT_BASELINE = {
    "Age": 35, "Department": "Engineering", "MonthlyIncome": 60_000, "YearsAtCompany": 3,
    "JobSatisfaction": 3, "WorkLifeBalance": 3, "PerformanceRating": 3,
    "OverTime": "No", "DistanceFromHome": 10,
}

FEATURE_LABELS = {
    "Age": "Age", "Department": "Department", "MonthlyIncome": "Income",
    "YearsAtCompany": "Tenure", "JobSatisfaction": "Job satisfaction",
    "WorkLifeBalance": "Work-life balance", "PerformanceRating": "Performance",
    "OverTime": "Overtime", "DistanceFromHome": "Commute",
}


def population_baseline(frame: pd.DataFrame, min_rows: int = 10) -> dict:
    """Typical employee: median of numeric columns, most common category."""
    if len(frame) < min_rows:
        return dict(DEFAULT_BASELINE)
    baseline = {}
    for column in FEATURE_COLUMNS:
        series = frame[column]
        baseline[column] = (float(series.median()) if pd.api.types.is_numeric_dtype(series)
                            else series.mode().iloc[0])
    for column in ["JobSatisfaction", "WorkLifeBalance", "PerformanceRating"]:
        baseline[column] = int(round(baseline[column]))
    return baseline


def describe_value(feature: str, value) -> str:
    """Human-readable 'Feature value' text, e.g. 'Income ₹22,000'."""
    label = FEATURE_LABELS[feature]
    if feature == "MonthlyIncome":
        return f"{label} ₹{float(value):,.0f}"
    if feature == "DistanceFromHome":
        return f"{label} {float(value):g} km"
    if feature == "YearsAtCompany":
        return f"{label} {float(value):g} yr"
    if feature == "OverTime":
        return "Works overtime" if value == "Yes" else "No overtime"
    if feature in {"JobSatisfaction", "WorkLifeBalance", "PerformanceRating"}:
        return f"{label} {int(value)}/5"
    return f"{label} {value}" if feature != "Age" else f"{label} {int(value)}"


def feature_impacts(predictor: AttritionPredictor, frame: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Impact of each feature on each row's risk (same index as ``frame``).

    impact = logit(P(actual)) - logit(P(feature replaced by the typical value)).
    """
    frame = frame[FEATURE_COLUMNS].copy()
    base_logit = _logit(predictor.predict_frame(frame))
    impacts = pd.DataFrame(index=frame.index, columns=FEATURE_COLUMNS, dtype=float)
    for feature in FEATURE_COLUMNS:
        counterfactual = frame.copy()
        counterfactual[feature] = baseline[feature]
        impacts[feature] = base_logit - _logit(predictor.predict_frame(counterfactual))
    return impacts


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def top_drivers(row: pd.Series, impacts: pd.Series, k: int = 3, min_impact: float = 0.15) -> str:
    """Comma-separated description of the features pushing risk up the most."""
    positive = impacts[impacts > min_impact].sort_values(ascending=False).head(k)
    if positive.empty:
        return "No standout risk factors"
    return ", ".join(describe_value(feature, row[feature]) for feature in positive.index)
