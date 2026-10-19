"""Tests for input validation and model-loading error handling."""

import pytest

from src.predict import (
    InputValidationError,
    ModelLoadError,
    load_model_bundle,
    validate_employee_input,
)

VALID = {
    "Age": 30, "Department": "sales", "MonthlyIncome": "45,000", "YearsAtCompany": 2,
    "JobSatisfaction": 3, "WorkLifeBalance": "4", "PerformanceRating": 3,
    "OverTime": "yes", "DistanceFromHome": 12,
}


def test_valid_input_is_normalised():
    cleaned = validate_employee_input(VALID)
    assert cleaned["Department"] == "Sales"
    assert cleaned["OverTime"] == "Yes"
    assert cleaned["MonthlyIncome"] == 45_000
    assert cleaned["WorkLifeBalance"] == 4


def test_key_names_are_flexible():
    raw = {key.lower(): value for key, value in VALID.items()}
    raw["monthly_income"] = raw.pop("monthlyincome")
    assert validate_employee_input(raw)["MonthlyIncome"] == 45_000


@pytest.mark.parametrize("field, bad_value", [
    ("Age", -5), ("Age", 17), ("Age", "abc"), ("Age", 30.5), ("Age", float("nan")),
    ("Department", "Legal"), ("Department", ""),
    ("MonthlyIncome", -1), ("MonthlyIncome", 0), ("MonthlyIncome", "free"),
    ("YearsAtCompany", -1),
    ("JobSatisfaction", 0), ("JobSatisfaction", 6), ("WorkLifeBalance", 2.5),
    ("PerformanceRating", "high"),
    ("OverTime", "maybe"),
    ("DistanceFromHome", -3),
])
def test_invalid_values_are_rejected(field, bad_value):
    with pytest.raises(InputValidationError) as error:
        validate_employee_input({**VALID, field: bad_value})
    assert field in error.value.errors


def test_all_errors_are_reported_together():
    with pytest.raises(InputValidationError) as error:
        validate_employee_input({**VALID, "Age": -1, "Department": "Legal", "MonthlyIncome": -5})
    assert {"Age", "Department", "MonthlyIncome"} <= set(error.value.errors)


def test_missing_fields_are_reported():
    with pytest.raises(InputValidationError) as error:
        validate_employee_input({"Age": 30})
    assert "Department" in error.value.errors


def test_tenure_longer_than_working_life_is_rejected():
    with pytest.raises(InputValidationError) as error:
        validate_employee_input({**VALID, "Age": 20, "YearsAtCompany": 15})
    assert "YearsAtCompany" in error.value.errors


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model_bundle(tmp_path / "nope.pkl")


def test_corrupted_model_file(tmp_path):
    broken = tmp_path / "broken.pkl"
    broken.write_bytes(b"this is not a pickle")
    with pytest.raises(ModelLoadError, match="Could not load"):
        load_model_bundle(broken)
