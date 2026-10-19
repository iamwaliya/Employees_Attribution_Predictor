"""Exploratory data analysis: attrition rates, correlations, top factors, plots.

Run on its own:
    python -m src.eda

Note on leakage: EDA looks at the whole dataset to *describe* it. Nothing
computed here feeds into model training or evaluation (the model pipeline
fits its own imputers/scalers on the training split only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_names,
    load_data,
    split_features_target,
)
from src.utils import DATA_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGET, VISUALS_DIR, ensure_dir, get_logger
from src.visualization import (
    plot_attrition_rate_bar,
    plot_correlation_heatmap,
    plot_correlation_with_target,
    plot_income_boxplot,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def overall_attrition_rate(df: pd.DataFrame) -> float:
    return float((df[TARGET] == "Yes").mean())


def attrition_rate_by(df: pd.DataFrame, column: str, sort_by_rate: bool = True) -> pd.DataFrame:
    """Employees, leavers and attrition rate for each value of ``column``."""
    grouped = (
        df.dropna(subset=[column])
        .assign(left=lambda d: (d[TARGET] == "Yes").astype(int))
        .groupby(column)["left"]
        .agg(employees="count", leavers="sum", attrition_rate="mean")
        .reset_index()
    )
    if sort_by_rate:
        return grouped.sort_values("attrition_rate", ascending=False).reset_index(drop=True)
    return grouped.sort_values(column).reset_index(drop=True)


def _format_rates_table(rates: pd.DataFrame) -> str:
    """Markdown table with whole-number counts and percentage rates."""
    table = rates.copy()
    group_column = table.columns[0]
    if pd.api.types.is_numeric_dtype(table[group_column]) and (table[group_column] % 1 == 0).all():
        table[group_column] = table[group_column].astype(int)
    table[group_column] = table[group_column].astype(str)
    table["employees"] = table["employees"].astype(int).astype(str)
    table["leavers"] = table["leavers"].astype(int).astype(str)
    table["attrition_rate"] = table["attrition_rate"].map("{:.1%}".format)
    return table.to_markdown(index=False)


def numeric_frame_with_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features plus 0/1 flags for OverTime and Attrition."""
    numeric = df[NUMERIC_FEATURES].copy()
    numeric["OverTime"] = (df["OverTime"] == "Yes").astype(float)
    numeric["Attrition"] = (df[TARGET] == "Yes").astype(float)
    return numeric


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlations (missing values are skipped pairwise)."""
    return numeric_frame_with_flags(df).corr()


def correlation_with_attrition(df: pd.DataFrame) -> pd.Series:
    corr = correlation_matrix(df)["Attrition"].drop("Attrition")
    return corr.sort_values(ascending=False)


def rank_attrition_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Rank features by |correlation| and mutual information with attrition.

    Mutual information also catches non-linear effects (e.g. department).
    The two scores are combined by average rank, so neither dominates.
    """
    X, y = split_features_target(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)  # EDA only: not used for evaluation
    names = get_feature_names(preprocessor)
    frame = pd.DataFrame(transformed, columns=names)

    mutual_info = mutual_info_classif(frame, y, random_state=RANDOM_STATE)
    correlation = frame.apply(lambda col: np.corrcoef(col, y)[0, 1])

    ranking = pd.DataFrame({
        "feature": names,
        "correlation": correlation.values,
        "abs_correlation": np.abs(correlation.values),
        "mutual_info": mutual_info,
    })
    ranking["rank_corr"] = ranking["abs_correlation"].rank(ascending=False)
    ranking["rank_mi"] = ranking["mutual_info"].rank(ascending=False)
    ranking["avg_rank"] = ranking[["rank_corr", "rank_mi"]].mean(axis=1)
    return ranking.sort_values("avg_rank").reset_index(drop=True)


def build_insights(df: pd.DataFrame, ranking: pd.DataFrame) -> list[str]:
    """Plain-English findings computed from the data (no hard-coded numbers)."""
    overall = overall_attrition_rate(df)
    insights = [f"Overall attrition rate is {overall:.1%} "
                f"({int((df[TARGET] == 'Yes').sum())} of {len(df)} employees), so the classes are imbalanced."]

    by_department = attrition_rate_by(df, "Department")
    top, bottom = by_department.iloc[0], by_department.iloc[-1]
    insights.append(
        f"{top['Department']} has the highest attrition ({top['attrition_rate']:.1%}); "
        f"{bottom['Department']} has the lowest ({bottom['attrition_rate']:.1%}).")

    by_overtime = attrition_rate_by(df, "OverTime").set_index("OverTime")["attrition_rate"]
    if {"Yes", "No"} <= set(by_overtime.index):
        insights.append(
            f"Employees working overtime leave at {by_overtime['Yes']:.1%} versus "
            f"{by_overtime['No']:.1%} without overtime ({by_overtime['Yes'] / by_overtime['No']:.1f}x higher).")

    by_satisfaction = attrition_rate_by(df, "JobSatisfaction", sort_by_rate=False) \
        .set_index("JobSatisfaction")["attrition_rate"]
    low, high = by_satisfaction.index.min(), by_satisfaction.index.max()
    insights.append(
        f"Attrition falls as job satisfaction rises: {by_satisfaction[low]:.1%} at score {int(low)} "
        f"versus {by_satisfaction[high]:.1%} at score {int(high)}.")

    medians = df.groupby(TARGET)["MonthlyIncome"].median()
    insights.append(
        f"Median monthly income is ₹{medians['Yes']:,.0f} for leavers versus ₹{medians['No']:,.0f} "
        f"for stayers ({medians['Yes'] / medians['No'] - 1:+.0%}).")

    new_joiners = df[df["YearsAtCompany"] <= 1]
    veterans = df[df["YearsAtCompany"] >= 5]
    insights.append(
        f"Employees with up to 1 year at the company leave at "
        f"{overall_attrition_rate(new_joiners):.1%}, compared with "
        f"{overall_attrition_rate(veterans):.1%} for those with 5+ years.")

    top_factors = ", ".join(ranking["feature"].head(5))
    insights.append(f"Top factors by combined correlation / mutual-information rank: {top_factors}.")
    return insights


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, visuals_dir: str | Path = VISUALS_DIR) -> dict:
    """Create the EDA plots and a markdown summary. Returns the main results."""
    visuals_dir = ensure_dir(visuals_dir)
    overall = overall_attrition_rate(df)

    rates_department = attrition_rate_by(df, "Department")
    rates_overtime = attrition_rate_by(df, "OverTime")
    rates_satisfaction = attrition_rate_by(df, "JobSatisfaction", sort_by_rate=False)

    plot_attrition_rate_bar(rates_department, "Department", visuals_dir / "attrition_by_department.png",
                            "Attrition rate by department", overall)
    plot_attrition_rate_bar(rates_overtime, "OverTime", visuals_dir / "attrition_by_overtime.png",
                            "Attrition rate by overtime status", overall, xlabel="Works overtime")
    plot_attrition_rate_bar(rates_satisfaction, "JobSatisfaction",
                            visuals_dir / "attrition_by_job_satisfaction.png",
                            "Attrition rate by job satisfaction", overall,
                            xlabel="Job satisfaction (1 = low, 5 = high)")
    plot_income_boxplot(df, visuals_dir / "income_boxplot.png")

    corr = correlation_matrix(df)
    plot_correlation_heatmap(corr, visuals_dir / "correlation_heatmap.png")
    target_corr = correlation_with_attrition(df)
    plot_correlation_with_target(target_corr, visuals_dir / "correlation_with_attrition.png")

    ranking = rank_attrition_factors(df)
    insights = build_insights(df, ranking)

    summary_path = visuals_dir / "eda_summary.md"
    lines = ["# EDA summary", "", "## Key insights", *[f"- {item}" for item in insights], "",
             "## Attrition by department", _format_rates_table(rates_department), "",
             "## Attrition by overtime", _format_rates_table(rates_overtime), "",
             "## Attrition by job satisfaction", _format_rates_table(rates_satisfaction), "",
             "## Factor ranking (EDA)",
             ranking[["feature", "correlation", "mutual_info", "avg_rank"]].to_markdown(index=False, floatfmt=".3f")]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved EDA summary -> %s", summary_path)

    for item in insights:
        logger.info("INSIGHT: %s", item)
    return {"insights": insights, "ranking": ranking, "correlation": corr,
            "rates_department": rates_department}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--visuals-dir", default=str(VISUALS_DIR))
    args = parser.parse_args()

    df = clean_data(load_data(args.data))
    run_eda(df, args.visuals_dir)


if __name__ == "__main__":
    main()
