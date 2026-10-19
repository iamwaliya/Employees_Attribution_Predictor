"""Shared constants, paths, logging and small helpers used across the project."""

from __future__ import annotations

import logging
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root so scripts work from anywhere)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"

DATA_PATH = DATA_DIR / "employee_data.csv"
DB_PATH = DATA_DIR / "employees.db"
TRAINING_EXPORT_PATH = DATA_DIR / "training_export.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

ID_COLUMN = "EmployeeID"
TARGET = "Attrition"

DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]

# Ordinal 1-5 scores are kept numeric: the order carries the signal and a
# linear/tree model can use it directly without exploding the feature space.
SCORE_FEATURES = ["JobSatisfaction", "WorkLifeBalance", "PerformanceRating"]
NUMERIC_FEATURES = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "DistanceFromHome",
    *SCORE_FEATURES,
]
CATEGORICAL_FEATURES = ["Department"]
BINARY_FEATURES = ["OverTime"]  # Yes / No

FEATURE_COLUMNS = ["Age", "Department", "MonthlyIncome", "YearsAtCompany",
                   *SCORE_FEATURES, "OverTime", "DistanceFromHome"]

# Plausible ranges used by data cleaning and by user-input validation.
AGE_RANGE = (18, 70)
SCORE_RANGE = (1, 5)
MAX_MONTHLY_INCOME = 5_000_000   # ₹
MAX_DISTANCE_KM = 500
MIN_WORKING_AGE = 14             # tenure cannot exceed (age - MIN_WORKING_AGE)


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if missing and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def risk_level(probability: float) -> str:
    """Translate an attrition probability into a human-friendly risk band."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"
