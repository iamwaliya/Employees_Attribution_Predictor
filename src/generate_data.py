"""Generate a realistic synthetic HR dataset for the attrition project.

Attrition is drawn from a logistic model whose drivers mirror what HR studies
usually find (overtime, low satisfaction, poor work-life balance, low pay,
short tenure, long commute). The intercept is calibrated by bisection so the
overall attrition rate lands near the requested value, which keeps the classes
imbalanced like a real HR dataset.

The raw file also contains a small amount of realistic mess (missing values,
inconsistent casing, a few impossible entries) so the cleaning step in
``preprocessing.py`` has something real to do. Pass ``--clean`` to skip that.

Usage:
    python -m src.generate_data --rows 1500 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_PATH, DEPARTMENTS, RANDOM_STATE, ensure_dir, get_logger

logger = get_logger(__name__)

# Typical monthly salary (₹) by department before age/tenure adjustments.
DEPARTMENT_BASE_INCOME = {
    "Engineering": 72_000,
    "Finance": 66_000,
    "Marketing": 56_000,
    "Sales": 50_000,
    "HR": 46_000,
}
DEPARTMENT_PROBABILITIES = [0.30, 0.30, 0.08, 0.14, 0.18]  # Sales, Eng, HR, Mkt, Fin
# Department-specific shift in attrition log-odds.
DEPARTMENT_ATTRITION_EFFECT = {
    "Sales": 0.45, "HR": 0.20, "Marketing": 0.10, "Engineering": -0.15, "Finance": -0.30,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_clean_frame(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create the employee attributes (without the target)."""
    department = rng.choice(DEPARTMENTS, size=n_rows, p=DEPARTMENT_PROBABILITIES)

    age = np.clip(rng.normal(35, 8, n_rows), 20, 60).round().astype(int)

    # Tenure: skewed towards newer joiners, never longer than the working life.
    years_at_company = np.minimum(rng.gamma(2.0, 3.0, n_rows), age - 20).round().astype(int)
    years_at_company = np.clip(years_at_company, 0, None)

    base_income = np.array([DEPARTMENT_BASE_INCOME[d] for d in department])
    income_noise = rng.lognormal(mean=0.0, sigma=0.22, size=n_rows)
    monthly_income = (
        base_income
        * np.exp(0.018 * (age - 30))
        * (1 + 0.03 * years_at_company)
        * income_noise
    )
    monthly_income = np.clip(monthly_income, 18_000, 400_000)
    monthly_income = (np.round(monthly_income / 100) * 100).astype(int)

    job_satisfaction = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.10, 0.18, 0.30, 0.27, 0.15])
    work_life_balance = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.06, 0.20, 0.47, 0.20, 0.07])
    performance_rating = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.02, 0.06, 0.52, 0.32, 0.08])

    overtime_probability = np.where(np.isin(department, ["Sales", "Engineering"]), 0.32, 0.22)
    overtime = np.where(rng.random(n_rows) < overtime_probability, "Yes", "No")

    distance_from_home = np.clip(rng.lognormal(2.1, 0.75, n_rows), 1, 60).round().astype(int)

    return pd.DataFrame({
        "EmployeeID": np.arange(1001, 1001 + n_rows),
        "Age": age,
        "Department": department,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "PerformanceRating": performance_rating,
        "OverTime": overtime,
        "DistanceFromHome": distance_from_home,
    })


def _attrition_logit(df: pd.DataFrame) -> np.ndarray:
    """Log-odds of leaving, excluding the intercept (calibrated separately)."""
    income_z = (np.log(df["MonthlyIncome"]) - np.log(df["MonthlyIncome"]).mean()) \
        / np.log(df["MonthlyIncome"]).std()
    new_joiner = (df["YearsAtCompany"] <= 1).astype(float)
    dept_effect = df["Department"].map(DEPARTMENT_ATTRITION_EFFECT).to_numpy()

    return (
        1.10 * (df["OverTime"] == "Yes").astype(float)
        - 0.55 * (df["JobSatisfaction"] - 3)
        - 0.40 * (df["WorkLifeBalance"] - 3)
        - 0.55 * income_z
        - 0.04 * (df["Age"] - 35)
        + 0.03 * (df["DistanceFromHome"] - 10)
        + 0.70 * new_joiner
        - 0.05 * df["YearsAtCompany"]
        + dept_effect
    ).to_numpy()


def _calibrate_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find the intercept that gives the desired mean attrition probability."""
    low, high = -15.0, 15.0
    for _ in range(60):
        mid = (low + high) / 2
        if _sigmoid(logit + mid).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic mess: missing values, inconsistent text, impossible entries."""
    df = df.copy()
    n = len(df)

    def random_rows(fraction: float) -> np.ndarray:
        return rng.choice(n, size=max(1, int(n * fraction)), replace=False)

    # Missing values (feature columns only; the target is never blanked).
    for column, fraction in {
        "MonthlyIncome": 0.03, "JobSatisfaction": 0.03, "DistanceFromHome": 0.025,
        "Age": 0.015, "Department": 0.02, "WorkLifeBalance": 0.02,
    }.items():
        df.loc[random_rows(fraction), column] = np.nan

    # Inconsistent casing / whitespace in text columns.
    rows = random_rows(0.02)
    df.loc[rows, "Department"] = df.loc[rows, "Department"].map(
        lambda v: v.upper() if isinstance(v, str) else v)
    rows = random_rows(0.02)
    df.loc[rows, "Department"] = df.loc[rows, "Department"].map(
        lambda v: f" {v.lower()} " if isinstance(v, str) else v)
    rows = random_rows(0.02)
    df.loc[rows, "OverTime"] = df.loc[rows, "OverTime"].str.lower()

    # A handful of impossible values that cleaning must neutralise.
    df = df.astype({"Age": "float64", "JobSatisfaction": "float64",
                    "MonthlyIncome": "float64", "DistanceFromHome": "float64"})
    df.loc[random_rows(0.003), "Age"] = -1
    df.loc[random_rows(0.003), "JobSatisfaction"] = 9
    df.loc[random_rows(0.003), "MonthlyIncome"] = -5000
    df.loc[random_rows(0.003), "DistanceFromHome"] = -3
    return df


def generate_employee_data(
    n_rows: int = 1500,
    attrition_rate: float = 0.17,
    seed: int = RANDOM_STATE,
    add_quality_issues: bool = True,
) -> pd.DataFrame:
    """Return a synthetic employee dataset with an imbalanced Attrition target."""
    rng = np.random.default_rng(seed)
    df = _build_clean_frame(n_rows, rng)

    logit = _attrition_logit(df)
    intercept = _calibrate_intercept(logit, attrition_rate)
    probability = _sigmoid(logit + intercept)
    df["Attrition"] = np.where(rng.random(n_rows) < probability, "Yes", "No")

    if add_quality_issues:
        df = _inject_data_quality_issues(df, rng)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic employee dataset.")
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--attrition-rate", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", default=str(DATA_PATH))
    parser.add_argument("--clean", action="store_true",
                        help="Skip injecting missing values / dirty entries.")
    args = parser.parse_args()

    df = generate_employee_data(args.rows, args.attrition_rate, args.seed,
                                add_quality_issues=not args.clean)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    rate = (df["Attrition"] == "Yes").mean()
    logger.info("Saved %d rows to %s (attrition rate %.1f%%)", len(df), output_path, rate * 100)


if __name__ == "__main__":
    main()
