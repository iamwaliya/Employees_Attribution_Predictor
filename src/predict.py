"""Prediction logic plus a console interface.

* ``validate_employee_input`` checks and normalises raw user input and reports
  *all* problems at once (so the user can fix them in a single pass).
* ``AttritionPredictor`` loads the saved model bundle with clear errors and
  returns the predicted label, probability and risk band.
* ``main`` is the console interface (interactive prompts or command-line flags).

Examples:
    python -m src.predict
    python -m src.predict --age 29 --department Sales --monthly-income 42000 \
        --years-at-company 1 --job-satisfaction 2 --work-life-balance 2 \
        --performance-rating 3 --overtime Yes --distance-from-home 25
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.utils import (
    AGE_RANGE,
    DEPARTMENTS,
    FEATURE_COLUMNS,
    MAX_DISTANCE_KM,
    MAX_MONTHLY_INCOME,
    MIN_WORKING_AGE,
    MODEL_PATH,
    SCORE_RANGE,
    risk_level,
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ModelLoadError(RuntimeError):
    """The saved model is missing, corrupted or otherwise unusable."""


class InputValidationError(ValueError):
    """One or more user inputs are invalid. ``errors`` maps field -> message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{field}: {message}" for field, message in errors.items()))


class PredictionError(RuntimeError):
    """The model failed to score a (valid) input."""


# --------------------------------------------------------------------------- #
# Field parsers: each returns a clean value or raises ValueError(message)
# --------------------------------------------------------------------------- #
def _to_number(value: Any) -> float:
    """Convert to a finite float; reject blanks, booleans, NaN and infinity."""
    if value is None or isinstance(value, bool):
        raise ValueError("a number is required.")
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("₹", "")
        if not cleaned:
            raise ValueError("a value is required.")
        try:
            number = float(cleaned)
        except ValueError:
            raise ValueError(f"'{value}' is not a number.") from None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{value}' is not a number.") from None
    if math.isnan(number) or math.isinf(number):
        raise ValueError("must be a finite number.")
    return number


def _to_integer(value: Any) -> int:
    number = _to_number(value)
    if number != int(number):
        raise ValueError(f"must be a whole number (got {number:g}).")
    return int(number)


def parse_age(value: Any) -> int:
    age = _to_integer(value)
    low, high = AGE_RANGE
    if not low <= age <= high:
        raise ValueError(f"must be between {low} and {high}.")
    return age


def parse_department(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"choose one of: {', '.join(DEPARTMENTS)}.")
    lookup = {name.lower(): name for name in DEPARTMENTS}
    match = lookup.get(value.strip().lower())
    if match is None:
        raise ValueError(f"'{value.strip()}' is not a valid department. Choose one of: "
                         f"{', '.join(DEPARTMENTS)}.")
    return match


def parse_monthly_income(value: Any) -> float:
    income = _to_number(value)
    if income <= 0:
        raise ValueError("must be greater than 0.")
    if income > MAX_MONTHLY_INCOME:
        raise ValueError(f"is unrealistically high (maximum ₹{MAX_MONTHLY_INCOME:,}).")
    return income


def parse_years_at_company(value: Any) -> float:
    years = _to_number(value)
    if years < 0:
        raise ValueError("cannot be negative.")
    if years > 60:
        raise ValueError("must be 60 or less.")
    return years


def parse_score(value: Any) -> int:
    score = _to_integer(value)
    low, high = SCORE_RANGE
    if not low <= score <= high:
        raise ValueError(f"must be a whole number from {low} to {high}.")
    return score


def parse_overtime(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"yes", "y", "true", "1"}:
            return "Yes"
        if cleaned in {"no", "n", "false", "0"}:
            return "No"
    raise ValueError("must be Yes or No.")


def parse_distance(value: Any) -> float:
    distance = _to_number(value)
    if distance < 0:
        raise ValueError("cannot be negative.")
    if distance > MAX_DISTANCE_KM:
        raise ValueError(f"must be {MAX_DISTANCE_KM} km or less.")
    return distance


FIELD_PARSERS: dict[str, Callable[[Any], Any]] = {
    "Age": parse_age,
    "Department": parse_department,
    "MonthlyIncome": parse_monthly_income,
    "YearsAtCompany": parse_years_at_company,
    "JobSatisfaction": parse_score,
    "WorkLifeBalance": parse_score,
    "PerformanceRating": parse_score,
    "OverTime": parse_overtime,
    "DistanceFromHome": parse_distance,
}

# Prompt text used by the console interface (same order as the assignment spec).
FIELD_PROMPTS: dict[str, str] = {
    "Age": "Age (years)",
    "Department": f"Department ({' / '.join(DEPARTMENTS)})",
    "MonthlyIncome": "Monthly income (₹)",
    "YearsAtCompany": "Years at company",
    "JobSatisfaction": "Job satisfaction (1-5)",
    "WorkLifeBalance": "Work-life balance (1-5)",
    "PerformanceRating": "Performance rating (1-5)",
    "OverTime": "Works overtime? (Yes/No)",
    "DistanceFromHome": "Distance from home (km)",
}


def _canonical_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def validate_employee_input(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise a raw employee record.

    Keys are matched case-insensitively and ignoring spaces/underscores, so
    ``monthly_income`` and ``MonthlyIncome`` both work. Raises
    ``InputValidationError`` listing every problem found.
    """
    lookup = {_canonical_key(str(key)): value for key, value in raw.items()}
    cleaned: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for field, parser in FIELD_PARSERS.items():
        key = _canonical_key(field)
        if key not in lookup or lookup[key] is None or (
                isinstance(lookup[key], str) and not lookup[key].strip()):
            errors[field] = "is required."
            continue
        try:
            cleaned[field] = parser(lookup[key])
        except ValueError as exc:
            errors[field] = str(exc)

    # Cross-field consistency: tenure cannot exceed the person's working life.
    if "Age" in cleaned and "YearsAtCompany" in cleaned:
        max_tenure = cleaned["Age"] - MIN_WORKING_AGE
        if cleaned["YearsAtCompany"] > max_tenure:
            errors["YearsAtCompany"] = (
                f"({cleaned['YearsAtCompany']:g}) is inconsistent with age {cleaned['Age']}; "
                f"it cannot exceed {max_tenure}.")
            cleaned.pop("YearsAtCompany")

    if errors:
        raise InputValidationError(errors)
    return cleaned


# --------------------------------------------------------------------------- #
# Model loading + prediction
# --------------------------------------------------------------------------- #
def load_model_bundle(path: str | Path = MODEL_PATH) -> dict:
    """Load and sanity-check the saved model bundle."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(
            f"Model file not found at '{path}'. Train it first with: python -m src.train_model")
    if not path.is_file():
        raise ModelLoadError(f"'{path}' is not a file.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # version warnings are handled explicitly below
            bundle = joblib.load(path)
    except Exception as exc:  # unpickling can fail in many ways (corruption, missing modules...)
        raise ModelLoadError(
            f"Could not load model '{path}' ({type(exc).__name__}: {exc}). The file may be "
            f"corrupted or built with incompatible library versions; retrain with "
            f"python -m src.train_model") from exc

    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelLoadError(f"'{path}' is not a valid model bundle (missing 'pipeline').")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ModelLoadError("The saved pipeline cannot produce probabilities.")

    saved_version = bundle.get("sklearn_version")
    if saved_version and saved_version.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        warnings.warn(f"Model was trained with scikit-learn {saved_version} but "
                      f"{sklearn.__version__} is installed; consider retraining.", stacklevel=2)
    return bundle


@dataclass
class PredictionResult:
    prediction: str          # "Yes" (likely to leave) or "No"
    probability: float       # probability of leaving, 0-1
    risk_level: str          # Low / Medium / High
    model_name: str
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


class AttritionPredictor:
    """Loads the trained pipeline once and scores employee records."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.bundle = load_model_bundle(model_path)
        self.pipeline = self.bundle["pipeline"]
        self.model_name = self.bundle.get("model_name", "Unknown model")
        self.threshold = float(self.bundle.get("threshold", 0.5))
        self.feature_columns = list(self.bundle.get("feature_columns", FEATURE_COLUMNS))

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Attrition probability for every row of an already-clean DataFrame."""
        if frame.empty:
            return np.array([], dtype=float)
        try:
            return self.pipeline.predict_proba(frame[self.feature_columns])[:, 1]
        except Exception as exc:
            raise PredictionError(f"The model could not score these records: {exc}") from exc

    def predict(self, raw_input: Mapping[str, Any]) -> PredictionResult:
        """Validate ``raw_input`` and return the attrition prediction."""
        record = validate_employee_input(raw_input)
        frame = pd.DataFrame([record], columns=self.feature_columns)
        try:
            probability = float(self.pipeline.predict_proba(frame)[0, 1])
        except Exception as exc:
            raise PredictionError(f"The model could not score this record: {exc}") from exc
        return PredictionResult(
            prediction="Yes" if probability >= self.threshold else "No",
            probability=probability,
            risk_level=risk_level(probability),
            model_name=self.model_name,
            threshold=self.threshold,
        )

    def score_frame(self, employees: pd.DataFrame) -> pd.Series:
        """Probability of leaving for every row of an already-validated DataFrame.

        Used for the dashboard (rows come from the database, which enforces the
        same rules as ``validate_employee_input``). Returns a Series aligned
        with ``employees.index``.
        """
        if employees.empty:
            return pd.Series(dtype=float)
        try:
            probabilities = self.pipeline.predict_proba(employees[self.feature_columns])[:, 1]
        except Exception as exc:
            raise PredictionError(f"The model could not score the employees: {exc}") from exc
        return pd.Series(probabilities, index=employees.index, name="Risk")


# --------------------------------------------------------------------------- #
# Console interface
# --------------------------------------------------------------------------- #
CLI_FLAGS = {
    "Age": "--age",
    "Department": "--department",
    "MonthlyIncome": "--monthly-income",
    "YearsAtCompany": "--years-at-company",
    "JobSatisfaction": "--job-satisfaction",
    "WorkLifeBalance": "--work-life-balance",
    "PerformanceRating": "--performance-rating",
    "OverTime": "--overtime",
    "DistanceFromHome": "--distance-from-home",
}


def _ask_field(field: str, collected: Mapping[str, Any], input_fn: Callable[[str], str]) -> Any:
    """Prompt until the answer is valid. Raises KeyboardInterrupt on 'q'."""
    while True:
        answer = input_fn(f"  {FIELD_PROMPTS[field]}: ").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            value = FIELD_PARSERS[field](answer)
            if field == "YearsAtCompany" and "Age" in collected \
                    and value > collected["Age"] - MIN_WORKING_AGE:
                raise ValueError(f"cannot exceed {collected['Age'] - MIN_WORKING_AGE} "
                                 f"for someone aged {collected['Age']}.")
            return value
        except ValueError as exc:
            print(f"    ✗ {FIELD_PROMPTS[field]} {exc}  (try again, or 'q' to quit)")


def format_result(result: PredictionResult) -> str:
    verdict = "LIKELY TO LEAVE" if result.prediction == "Yes" else "LIKELY TO STAY"
    return (
        "\n" + "=" * 52 + "\n"
        f"  Predicted attrition : {result.prediction}  ({verdict})\n"
        f"  Probability of leaving: {result.probability:.1%}\n"
        f"  Risk level          : {result.risk_level}\n"
        f"  Model               : {result.model_name} (threshold {result.threshold:.2f})\n"
        + "=" * 52
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict whether an employee is likely to leave. Run without flags for "
                    "interactive prompts.")
    for field, flag in CLI_FLAGS.items():
        parser.add_argument(flag, dest=field, default=None, help=FIELD_PROMPTS[field])
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the saved model.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    return parser


def main(argv: list[str] | None = None, input_fn: Callable[[str], str] = input) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        predictor = AttritionPredictor(args.model)
    except ModelLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    provided = {field: getattr(args, field) for field in CLI_FLAGS if getattr(args, field) is not None}

    # ---- one-shot mode: every field was supplied as a flag ---------------
    if len(provided) == len(CLI_FLAGS):
        try:
            result = predictor.predict(provided)
        except InputValidationError as exc:
            print("Invalid input:", file=sys.stderr)
            for field, message in exc.errors.items():
                print(f"  - {field} {message}", file=sys.stderr)
            return 2
        except PredictionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result.to_dict(), indent=2) if args.json else format_result(result))
        return 0

    # ---- interactive mode -------------------------------------------------
    print("\nEmployee Attrition Predictor  (type 'q' at any prompt to quit)")
    try:
        while True:
            collected: dict[str, Any] = {}
            for field in FIELD_PARSERS:
                if field in provided:
                    try:
                        collected[field] = FIELD_PARSERS[field](provided[field])
                        continue
                    except ValueError as exc:
                        print(f"  ✗ {FIELD_PROMPTS[field]} {exc}")
                collected[field] = _ask_field(field, collected, input_fn)
            provided = {}  # flags only apply to the first employee

            try:
                print(format_result(predictor.predict(collected)))
            except (InputValidationError, PredictionError) as exc:
                print(f"Error: {exc}", file=sys.stderr)

            again = input_fn("\nPredict another employee? (y/n): ").strip().lower()
            if again not in {"y", "yes"}:
                break
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
