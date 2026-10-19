"""Tests for the SQLite employee database."""

import sqlite3

import pandas as pd
import pytest

from src.database import EmployeeDB, seed_database, validate_record
from src.insights import feature_impacts, population_baseline, top_drivers
from src.predict import InputValidationError

GOOD = {"Name": "Test Person", "Age": 30, "Department": "Sales", "MonthlyIncome": 45000,
        "YearsAtCompany": 2, "JobSatisfaction": 3, "WorkLifeBalance": 3, "PerformanceRating": 3,
        "OverTime": "No", "DistanceFromHome": 8}


@pytest.fixture()
def db(tmp_path):
    return EmployeeDB(tmp_path / "test.db")


def test_add_get_update_delete(db):
    new_id = db.add(GOOD)
    assert db.count() == 1
    assert db.get(new_id)["Name"] == "Test Person"
    assert db.get(new_id)["Attrition"] == "No"

    db.update(new_id, {**GOOD, "Name": "Renamed", "Attrition": "Yes"})
    row = db.get(new_id)
    assert row["Name"] == "Renamed" and row["Attrition"] == "Yes"

    assert db.delete([new_id]) == 1
    assert db.count() == 0
    assert db.get(new_id) is None


def test_ids_are_never_reused(db):
    first = db.add(GOOD)
    db.delete([first])
    assert db.add(GOOD) > first


def test_invalid_records_are_rejected_and_not_stored(db):
    with pytest.raises(InputValidationError) as error:
        db.add({**GOOD, "Name": "  ", "Age": -4, "Department": "Legal"})
    assert {"Name", "Age", "Department"} <= set(error.value.errors)
    assert db.count() == 0


def test_database_constraints_block_bad_rows_even_without_validation(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db._connect() as conn:
            conn.execute("INSERT INTO employees (Name, Age, Department, MonthlyIncome, YearsAtCompany, "
                         "JobSatisfaction, WorkLifeBalance, PerformanceRating, OverTime, DistanceFromHome) "
                         "VALUES ('X', 5, 'Sales', 1000, 1, 3, 3, 3, 'No', 1)")


def test_update_missing_employee_raises(db):
    with pytest.raises(KeyError):
        db.update(999, GOOD)


def test_validate_record_normalises_values():
    record = validate_record({**GOOD, "Name": "  Asha   Rao ", "Department": " sales ",
                              "OverTime": "YES", "Attrition": "yes"})
    assert record["Name"] == "Asha Rao" and record["Department"] == "Sales"
    assert record["OverTime"] == "Yes" and record["Attrition"] == "Yes"


def test_csv_import_reports_bad_rows_and_inserts_good_ones(db):
    frame = pd.DataFrame([
        {**GOOD, "Name": "Good One"},
        {**GOOD, "Name": "Bad Age", "Age": 3},
        {**GOOD, "Name": "Bad Dept", "Department": "Legal"},
        {**GOOD, "Name": "Another Good", "Attrition": "Yes"},
    ])
    inserted, errors = db.import_dataframe(frame)
    assert inserted == 2 and len(errors) == 2
    assert errors[0].startswith("Row 3") and "Age" in errors[0]
    assert db.count() == 2


def test_csv_import_without_name_or_status_columns(db):
    frame = pd.DataFrame([{k: v for k, v in GOOD.items() if k != "Name"}])
    frame.columns = [c.lower() for c in frame.columns]  # header casing is flexible
    inserted, errors = db.import_dataframe(frame)
    assert inserted == 1 and not errors
    assert db.list().iloc[0]["Name"] == "Unnamed"


def test_csv_import_missing_columns_fails_clearly(db):
    with pytest.raises(InputValidationError, match="missing required columns"):
        db.import_dataframe(pd.DataFrame([{"Age": 30}]))


def test_seed_export_roundtrip_and_training_readiness(db, tmp_path):
    assert db.training_readiness()[0] is False
    seed_database(db, n_rows=400, seed=3)
    assert db.count() == 400
    assert db.training_readiness()[0] is True
    exported = pd.read_csv(db.export_csv(tmp_path / "out.csv"))
    assert len(exported) == 400 and "Attrition" in exported.columns
    assert not exported.isna().any().any()


def test_delete_all_resets_ids(db):
    seed_database(db, n_rows=30, seed=1)
    assert db.delete_all() == 30
    assert db.count() == 0
    assert db.add(GOOD) == 1


def test_driver_explanations_point_at_risk_factors(tmp_path):
    from src.predict import AttritionPredictor
    from src.utils import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("trained model not available")
    predictor = AttritionPredictor(MODEL_PATH)
    db = EmployeeDB(tmp_path / "x.db")
    seed_database(db, n_rows=200, seed=5)
    population = db.list()
    baseline = population_baseline(population)
    risky = pd.DataFrame([{**GOOD, "OverTime": "Yes", "JobSatisfaction": 1, "MonthlyIncome": 20000,
                           "Department": "Sales", "YearsAtCompany": 0}])
    impacts = feature_impacts(predictor, risky, baseline).iloc[0]
    text = top_drivers(risky.iloc[0], impacts)
    assert impacts["OverTime"] > 0 and impacts["JobSatisfaction"] > 0
    assert "overtime" in text.lower()
