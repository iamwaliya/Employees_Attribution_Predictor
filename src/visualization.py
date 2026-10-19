"""All plotting functions. Every function saves a PNG and returns its path.

Style notes: one blue/orange pair for the two outcomes (stays distinguishable
for colour-blind readers), a blue-grey-red diverging scale for correlations,
light gridlines, and direct value labels instead of heavy chart furniture.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers and in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ---- palette ---------------------------------------------------------------
BLUE = "#2a78d6"      # stayed / primary series
ORANGE = "#eb6834"    # left / highlighted series
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1c5cab", "#2a78d6", "#f0efec", "#e34948", "#a82a2a"])

OUTCOME_PALETTE = {"No": BLUE, "Yes": ORANGE}


def apply_style() -> None:
    """Set global matplotlib/seaborn defaults once."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "font.size": 10,
        "legend.frameon": False,
    })


apply_style()


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot -> %s", path)
    return path


def _percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = FuncFormatter(lambda value, _: f"{value:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


# --------------------------------------------------------------------------- #
# EDA plots
# --------------------------------------------------------------------------- #
def plot_attrition_rate_bar(
    rates: pd.DataFrame,
    group_column: str,
    path: str | Path,
    title: str,
    overall_rate: float,
    xlabel: str | None = None,
) -> Path:
    """Bar chart of attrition rate (%) per group, with a dashed overall-rate line.

    ``rates`` is the frame returned by ``eda.attrition_rate_by``.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    group_values = rates[group_column]
    if pd.api.types.is_numeric_dtype(group_values) and (group_values % 1 == 0).all():
        group_values = group_values.astype(int)  # show 1, 2, 3 rather than 1.0, 2.0, 3.0
    labels = group_values.astype(str)
    values = rates["attrition_rate"] * 100

    # Highlight the groups that sit above the company average.
    colours = [ORANGE if v > overall_rate * 100 else BLUE for v in values]
    bars = ax.bar(labels, values, color=colours, width=0.6)
    ax.axhline(overall_rate * 100, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, overall_rate * 100 + 0.4, f"Company average {overall_rate:.1%}",
            ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)

    for bar, value, n in zip(bars, values, rates["employees"]):
        # Filled label background keeps the text readable where it crosses the average line.
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold", zorder=5,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
        ax.text(bar.get_x() + bar.get_width() / 2, 0.4, f"n={n}",
                ha="center", va="bottom", fontsize=8, color="white")

    ax.set_ylim(0, max(values.max() * 1.2, overall_rate * 100 * 1.4))
    ax.set_ylabel("Attrition rate")
    ax.set_xlabel(xlabel or group_column)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _percent_axis(ax)
    return _save(fig, path)


def plot_income_boxplot(df: pd.DataFrame, path: str | Path) -> Path:
    """Box plot of MonthlyIncome for employees who stayed vs. left."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    data = df.dropna(subset=["MonthlyIncome"])
    sns.boxplot(data=data, x="Attrition", y="MonthlyIncome", order=["No", "Yes"],
                hue="Attrition", palette=OUTCOME_PALETTE, legend=False, width=0.5,
                linewidth=1.2, fliersize=3, flierprops={"markerfacecolor": INK_MUTED,
                                                        "markeredgecolor": "none"}, ax=ax)
    medians = data.groupby("Attrition")["MonthlyIncome"].median()
    for position, label in enumerate(["No", "Yes"]):
        # Place the median label just right of the box so it never covers the plot.
        ax.text(position + 0.29, medians[label], f"median ₹{medians[label]:,.0f}", va="center",
                ha="left", fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(-0.5, 1.9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v / 1000:.0f}k"))
    ax.set_xticks([0, 1], ["Stayed", "Left"])
    ax.set_xlabel("")
    ax.set_ylabel("Monthly income")
    ax.set_title("Monthly income: employees who left vs. stayed")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, path: str | Path) -> Path:
    """Annotated correlation heatmap (blue = negative, red = positive)."""
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # lower triangle only
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True,
                fmt=".2f", annot_kws={"size": 8}, linewidths=1.5, linecolor=SURFACE,
                square=True, cbar_kws={"shrink": 0.75, "label": "Pearson correlation"}, ax=ax)
    ax.set_title("Correlation of numerical features")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, path)


def plot_correlation_with_target(target_corr: pd.Series, path: str | Path) -> Path:
    """Bar chart of each feature's correlation with attrition."""
    series = target_corr.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colours = [ORANGE if v > 0 else BLUE for v in series]
    ax.barh(series.index, series.values, color=colours, height=0.6)
    for y, value in enumerate(series.values):
        ax.text(value + (0.004 if value >= 0 else -0.004), y, f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    limit = max(abs(series).max() * 1.25, 0.1)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Correlation with attrition (orange = more likely to leave)")
    ax.set_title("Which features move with attrition?")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Model plots
# --------------------------------------------------------------------------- #
def plot_feature_importance(
    importances: pd.Series,
    path: str | Path,
    model_name: str,
    errors: pd.Series | None = None,
    top_n: int = 12,
    xlabel: str = "Relative importance",
) -> Path:
    """Horizontal bar chart of the trained model's feature importances.

    ``errors`` (optional) draws +/- one standard deviation whiskers, e.g. from
    repeated permutations.
    """
    top = importances.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    xerr = errors.reindex(top.index).to_numpy() if errors is not None else None
    ax.barh(top.index, top.values, xerr=xerr, color=BLUE, height=0.62,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3})
    pad = (top + (errors.reindex(top.index) if errors is not None else 0)).max() * 0.02
    for y, (value, err) in enumerate(zip(top.values, xerr if xerr is not None else [0] * len(top))):
        ax.text(max(value, 0) + err + pad, y, f"{value:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(min(top.min(), 0) * 1.3, (top.max() + (xerr.max() if xerr is not None else 0)) * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top drivers of attrition ({model_name})")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
                    path: str | Path) -> Path:
    """Overlay ROC curves for every tuned model on the held-out test set."""
    series_colours = [BLUE, ORANGE, "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for (name, (fpr, tpr, auc)), colour in zip(curves.items(), series_colours):
        ax.plot(fpr, tpr, color=colour, linewidth=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion_matrix(matrix: np.ndarray, path: str | Path, model_name: str) -> Path:
    """Annotated confusion matrix for the selected model."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=sns.light_palette(BLUE, as_cmap=True),
                cbar=False, linewidths=2, linecolor=SURFACE, annot_kws={"size": 14},
                xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({model_name})")
    ax.grid(False)
    return _save(fig, path)
