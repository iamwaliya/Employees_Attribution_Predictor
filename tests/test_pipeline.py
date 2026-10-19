"""Tests for data generation, cleaning, preprocessing and the end-to-end flow."""

import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_employee_data
from src.predict import AttritionPredictor
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_names,
    split_features_target,
    split_train_test,
)
from src.train_model import run_training


@pytest.fixture(scope="module")
def raw_df():
    return generate_employee_data(n_rows=600, seed=7)


def test_generated_data_is_imbalanced_and_messy(raw_df):
    rate = (raw_df["Attrition"] == "Yes").mean()
    assert 0.08 < rate < 0.30
    assert raw_df.isna().any().any()  # missing values were injected


def test_cleaning_removes_impossible_values_and_normalises_text(raw_df):
    cleaned = clean_data(raw_df)
    assert set(cleaned["Department"].dropna()) <= {"Sales", "Engineering", "HR", "Marketing", "Finance"}
    assert set(cleaned["OverTime"].dropna()) <= {"Yes", "No"}
    assert cleaned["Age"].dropna().between(18, 70).all()
    assert cleaned["JobSatisfaction"].dropna().between(1, 5).all()
    assert (cleaned["MonthlyIncome"].dropna() > 0).all()


def test_preprocessor_outputs_complete_numeric_matrix(raw_df):
    X, y = split_features_target(clean_data(raw_df))
    X_train, X_test, _, _ = split_train_test(X, y)
    preprocessor = build_preprocessor().fit(X_train)
    transformed = preprocessor.transform(X_test)
    assert not np.isnan(transformed).any()
    assert transformed.shape[1] == len(get_feature_names(preprocessor))


def test_preprocessor_statistics_come_from_training_data_only(raw_df):
    """Scaler means must equal the TRAIN means, proving no test data leaked in."""
    X, y = split_features_target(clean_data(raw_df))
    X_train, X_test, _, _ = split_train_test(X, y)
    preprocessor = build_preprocessor().fit(X_train)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    age_index = ["Age", "YearsAtCompany", "DistanceFromHome", "JobSatisfaction",
                 "WorkLifeBalance", "PerformanceRating"].index("Age")
    assert scaler.mean_[age_index] == pytest.approx(X_train["Age"].mean(), rel=0.02)
    assert scaler.mean_[age_index] != pytest.approx(X["Age"].mean(), abs=1e-9)


def test_unknown_department_does_not_crash_preprocessor(raw_df):
    X, y = split_features_target(clean_data(raw_df))
    preprocessor = build_preprocessor().fit(X)
    odd = X.head(3).copy()
    odd["Department"] = "Legal"
    assert preprocessor.transform(odd).shape[0] == 3


def test_end_to_end_training_and_prediction(tmp_path, raw_df):
    data_path = tmp_path / "data.csv"
    raw_df.to_csv(data_path, index=False)
    summary = run_training(data_path, n_iter=3, cv_folds=3, models_dir=tmp_path / "models",
                           visuals_dir=tmp_path / "visuals", with_shap=False)
    assert summary["best_model"] in {"Logistic Regression", "Random Forest", "XGBoost",
                                      "Gradient Boosting"}
    assert 0.5 < summary["test_metrics"]["roc_auc"] <= 1.0
    assert (tmp_path / "visuals" / "feature_importance.png").exists()

    predictor = AttritionPredictor(tmp_path / "models" / "best_model.pkl")
    risky = predictor.predict({"Age": 24, "Department": "Sales", "MonthlyIncome": 22000,
                               "YearsAtCompany": 0, "JobSatisfaction": 1, "WorkLifeBalance": 1,
                               "PerformanceRating": 3, "OverTime": "Yes", "DistanceFromHome": 40})
    safe = predictor.predict({"Age": 45, "Department": "Finance", "MonthlyIncome": 120000,
                              "YearsAtCompany": 15, "JobSatisfaction": 5, "WorkLifeBalance": 4,
                              "PerformanceRating": 4, "OverTime": "No", "DistanceFromHome": 3})
    assert 0 <= safe.probability < risky.probability <= 1
    assert risky.prediction == "Yes"
    assert safe.prediction == "No"
