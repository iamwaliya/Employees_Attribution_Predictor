"""Data loading, cleaning, encoding and scaling.

Design notes
------------
* ``clean_data`` only does *row-level, statistics-free* fixes (trim text, mark
  impossible values as missing). Anything that learns from the data (median
  imputation, scaling, one-hot categories) lives inside the sklearn
  ``ColumnTransformer`` returned by ``build_preprocessor`` so it is fitted on
  the training split only. That is what prevents train/test data leakage.
* The same fitted transformer is saved inside the model pipeline, so the
  prediction interface applies exactly the same transformations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.utils import (
    AGE_RANGE,
    CATEGORICAL_FEATURES,
    DEPARTMENTS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    MAX_DISTANCE_KM,
    MAX_MONTHLY_INCOME,
    MIN_WORKING_AGE,
    RANDOM_STATE,
    SCORE_FEATURES,
    SCORE_RANGE,
    TARGET,
    get_logger,
)

logger = get_logger(__name__)

# MonthlyIncome is right-skewed, so it gets a log transform before scaling.
SKEWED_FEATURES = ["MonthlyIncome"]
PLAIN_NUMERIC_FEATURES = ["Age", "YearsAtCompany", "DistanceFromHome", *SCORE_FEATURES]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_data(path: str | Path) -> pd.DataFrame:
    """Read the employee CSV, raising clear errors for the common failure modes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Generate it with: python -m src.generate_data"
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset file '{path}' is not a valid CSV: {exc}") from exc

    required = {*FEATURE_COLUMNS, TARGET}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    return df


# --------------------------------------------------------------------------- #
# Cleaning (no statistics are learned here, so it is safe before the split)
# --------------------------------------------------------------------------- #
def _normalise_department(value: object) -> object:
    """Map 'sales', ' SALES ' ... onto the canonical department names."""
    if not isinstance(value, str):
        return np.nan
    lookup = {name.lower(): name for name in DEPARTMENTS}
    return lookup.get(value.strip().lower(), np.nan)


def _normalise_yes_no(value: object) -> object:
    if not isinstance(value, str):
        return np.nan
    cleaned = value.strip().lower()
    if cleaned in {"yes", "y", "true", "1"}:
        return "Yes"
    if cleaned in {"no", "n", "false", "0"}:
        return "No"
    return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent text and turn impossible values into NaN.

    Impossible values are set to NaN (not dropped) so the imputer inside the
    pipeline can fill them using training-set statistics only.
    """
    df = df.copy()
    report: dict[str, int] = {}

    # Drop exact duplicate employees and rows without a usable target.
    before = len(df)
    if ID_COLUMN in df.columns:
        df = df.drop_duplicates(subset=ID_COLUMN)
    df["Attrition"] = df[TARGET].map(_normalise_yes_no)
    df = df.dropna(subset=[TARGET])
    report["rows_dropped"] = before - len(df)

    df["Department"] = df["Department"].map(_normalise_department)
    df["OverTime"] = df["OverTime"].map(_normalise_yes_no)

    numeric_columns = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome", *SCORE_FEATURES]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_masks = {
        "Age": ~df["Age"].between(*AGE_RANGE),
        "MonthlyIncome": ~df["MonthlyIncome"].between(1, MAX_MONTHLY_INCOME),
        "DistanceFromHome": ~df["DistanceFromHome"].between(0, MAX_DISTANCE_KM),
        "YearsAtCompany": df["YearsAtCompany"] < 0,
    }
    for column in SCORE_FEATURES:
        invalid_masks[column] = ~df[column].between(*SCORE_RANGE)

    for column, mask in invalid_masks.items():
        # ``between`` is False for NaN, so exclude rows that were already missing.
        mask = mask & df[column].notna()
        report[f"invalid_{column}"] = int(mask.sum())
        df.loc[mask, column] = np.nan

    # Tenure longer than working life is inconsistent: blank it out.
    inconsistent = df["YearsAtCompany"] > (df["Age"] - MIN_WORKING_AGE)
    report["inconsistent_tenure"] = int(inconsistent.sum())
    df.loc[inconsistent, "YearsAtCompany"] = np.nan

    df[TARGET] = df[TARGET].astype(str)
    summary = {k: v for k, v in report.items() if v}
    if summary:
        logger.info("Cleaning fixed/flagged: %s", summary)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features X and a 0/1 target y (1 = employee left)."""
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split so train and test keep the same attrition rate."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# --------------------------------------------------------------------------- #
# Encoding + scaling (fitted on training data only, inside the model pipeline)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Create the imputation, encoding and scaling transformer.

    * numeric features      -> median imputation -> standard scaling
    * MonthlyIncome         -> median imputation -> log1p -> standard scaling
    * Department (nominal)  -> most-frequent imputation -> one-hot encoding
    * OverTime (Yes/No)     -> most-frequent imputation -> 0/1 ordinal encoding

    Unknown categories at prediction time are ignored by the one-hot encoder
    instead of raising an error.
    """
    plain_numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    skewed_numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    department = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", categories=[DEPARTMENTS])),
    ])
    overtime = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[["No", "Yes"]])),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", plain_numeric, PLAIN_NUMERIC_FEATURES),
            ("income", skewed_numeric, SKEWED_FEATURES),
            ("department", department, CATEGORICAL_FEATURES),
            ("overtime", overtime, ["OverTime"]),
        ],
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Names of the columns produced by a fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())
