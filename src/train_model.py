"""Train, tune, compare and save attrition models.

Pipeline
--------
1. Load + clean the data, split into stratified train / test sets.
2. For each candidate model build ``preprocess -> (SMOTE) -> model`` and tune
   its hyper-parameters with cross-validation on the TRAINING set only.
3. Score every tuned model once on the untouched TEST set.
4. Pick the best model by *cross-validated* ROC-AUC (never by test score, so the
   test set stays an unbiased estimate), save it and the supporting artefacts.

Class imbalance is handled in one of two ways (``--imbalance``):
* ``class_weight`` (default): the loss gives minority-class errors more weight.
* ``smote``: synthetic minority samples are created inside each CV training fold
  (imblearn pipeline), so validation folds never see synthetic data.

Run:
    python -m src.train_model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_names,
    load_data,
    split_features_target,
    split_train_test,
)
from src.utils import (
    DATA_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    VISUALS_DIR,
    ensure_dir,
    get_logger,
)
from src.visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_curves

logger = get_logger(__name__)

DECISION_THRESHOLD = 0.5

try:  # XGBoost is an optional (bonus) dependency
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
@dataclass
class ModelCandidate:
    name: str
    estimator: BaseEstimator
    param_space: dict
    search: str = "random"          # "grid" or "random"
    n_iter: int = 20
    notes: str = field(default="")


def get_candidates(imbalance: str, positive_weight: float, n_iter: int) -> list[ModelCandidate]:
    """Return the models to compare, configured for the chosen imbalance strategy."""
    class_weight = "balanced" if imbalance == "class_weight" else None
    # XGBoost has no class_weight; scale_pos_weight (= negatives / positives) does the same job.
    xgb_weight = positive_weight if imbalance == "class_weight" else 1.0

    candidates = [
        ModelCandidate(
            name="Logistic Regression",
            estimator=LogisticRegression(max_iter=3000, class_weight=class_weight,
                                         random_state=RANDOM_STATE),
            # Inverse regularisation strength (smaller C = stronger L2 penalty).
            param_space={"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
            search="grid",
            notes="Linear baseline; coefficients are directly interpretable.",
        ),
        ModelCandidate(
            name="Random Forest",
            estimator=RandomForestClassifier(class_weight=(
                "balanced_subsample" if class_weight else None),
                random_state=RANDOM_STATE, n_jobs=1),
            param_space={"model__n_estimators": randint(150, 600),
                         "model__max_depth": [4, 6, 8, 12, None],
                         "model__min_samples_leaf": randint(1, 15),
                         "model__max_features": ["sqrt", "log2", 0.5]},
            n_iter=n_iter,
            notes="Bagged trees; captures non-linear effects and interactions.",
        ),
    ]

    if XGBClassifier is not None:
        candidates.append(ModelCandidate(
            name="XGBoost",
            estimator=XGBClassifier(eval_metric="logloss", tree_method="hist",
                                    scale_pos_weight=xgb_weight, random_state=RANDOM_STATE,
                                    n_jobs=1, verbosity=0),
            param_space={"model__n_estimators": randint(100, 500),
                         "model__max_depth": randint(2, 7),
                         "model__learning_rate": loguniform(0.01, 0.3),
                         "model__subsample": uniform(0.6, 0.4),
                         "model__colsample_bytree": uniform(0.6, 0.4),
                         "model__min_child_weight": randint(1, 10),
                         "model__reg_lambda": loguniform(0.1, 10)},
            n_iter=n_iter,
            notes="Gradient-boosted trees; usually the strongest on tabular data.",
        ))
    else:
        logger.warning("xgboost not installed - using HistGradientBoosting instead.")
        candidates.append(ModelCandidate(
            name="Gradient Boosting",
            estimator=HistGradientBoostingClassifier(class_weight=class_weight,
                                                     random_state=RANDOM_STATE),
            param_space={"model__learning_rate": loguniform(0.01, 0.3),
                         "model__max_depth": [2, 3, 4, None],
                         "model__max_leaf_nodes": randint(8, 40),
                         "model__l2_regularization": loguniform(0.01, 10)},
            n_iter=n_iter,
        ))
    return candidates


def build_pipeline(estimator: BaseEstimator, imbalance: str) -> Pipeline:
    """preprocess -> [SMOTE] -> model. The preprocessor is fitted inside each fold."""
    if imbalance == "smote":
        # Imported lazily so the default path has no dependency on imbalanced-learn.
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ])
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


# --------------------------------------------------------------------------- #
# Tuning and evaluation
# --------------------------------------------------------------------------- #
def tune_candidate(candidate: ModelCandidate, X_train: pd.DataFrame, y_train: pd.Series,
                   imbalance: str, cv_folds: int):
    """Hyper-parameter search with stratified CV, optimising ROC-AUC."""
    pipeline = build_pipeline(candidate.estimator, imbalance)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    common = dict(estimator=pipeline, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)

    if candidate.search == "grid":
        search = GridSearchCV(param_grid=candidate.param_space, **common)
    else:
        search = RandomizedSearchCV(param_distributions=candidate.param_space,
                                    n_iter=candidate.n_iter, random_state=RANDOM_STATE, **common)
    search.fit(X_train, y_train)
    return search


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute all required metrics on the held-out test set."""
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "_probabilities": probabilities,
        "_predictions": predictions,
    }


def get_feature_importance(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 30) -> pd.DataFrame:
    """Permutation importance of the trained pipeline.

    Each input feature is shuffled in turn and we measure how much ROC-AUC drops.
    Compared with model-native importances this (a) works for every model type,
    (b) keeps one-hot columns grouped as a single feature (Department), and
    (c) is comparable across features. Returns mean and std over the repeats.

    This describes what the *model* relies on, so it is computed on all rows: the
    test split alone has only ~50 leavers, which makes the ranking noisy. It is
    an explanation only; every reported performance metric still comes from the
    held-out test set.
    """
    result = permutation_importance(pipeline, X, y, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({"importance": result.importances_mean, "std": result.importances_std},
                        index=X.columns).sort_values("importance", ascending=False)


def explain_with_shap(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      path: Path, model_name: str) -> bool:
    """Optional SHAP summary plot. Returns False (without failing) if unavailable."""
    try:
        import matplotlib.pyplot as plt
        import shap

        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        names = get_feature_names(preprocessor)
        test_matrix = pd.DataFrame(preprocessor.transform(X_test), columns=names)

        if hasattr(model, "coef_"):
            background = pd.DataFrame(preprocessor.transform(X_train), columns=names)
            background = shap.sample(background, 100, random_state=RANDOM_STATE)
            explainer = shap.LinearExplainer(model, background)
            values = explainer.shap_values(test_matrix)
        else:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(test_matrix)
            if isinstance(values, list):          # older shap: one array per class
                values = values[1]
            elif getattr(values, "ndim", 2) == 3:  # newer shap: (rows, features, classes)
                values = values[:, :, 1]

        shap.summary_plot(values, test_matrix, show=False, plot_size=(8, 5.5))
        plt.title(f"SHAP summary ({model_name})", loc="left", fontweight="bold")
        plt.tight_layout()
        ensure_dir(path.parent)
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Saved plot -> %s", path)
        return True
    except Exception as exc:  # SHAP is a bonus: never break training because of it
        logger.warning("Skipping SHAP explanation (%s: %s)", type(exc).__name__, exc)
        return False


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def justify_selection(results: pd.DataFrame, best_name: str) -> str:
    """Build a short, data-driven explanation of why the winner was chosen."""
    best = results.loc[best_name]
    others = results.drop(index=best_name)
    runner_up = others["cv_roc_auc"].idxmax()
    margin = best["cv_roc_auc"] - others.loc[runner_up, "cv_roc_auc"]
    return (
        f"{best_name} was selected because it achieved the highest cross-validated ROC-AUC "
        f"({best['cv_roc_auc']:.3f}, ahead of {runner_up} by {margin:+.3f}). ROC-AUC measures how well a model "
        f"ranks leavers above stayers regardless of the decision threshold, which suits an imbalanced "
        f"problem where plain accuracy is misleading. On the untouched test set it reached "
        f"ROC-AUC {best['roc_auc']:.3f}, recall {best['recall']:.3f}, precision {best['precision']:.3f} "
        f"and F1 {best['f1']:.3f}. Selection used cross-validation on the training data only, so the "
        f"test metrics are an unbiased estimate of real-world performance."
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_training(
    data_path: str | Path = DATA_PATH,
    imbalance: str = "class_weight",
    test_size: float = 0.2,
    cv_folds: int = 5,
    n_iter: int = 20,
    models_dir: str | Path = MODELS_DIR,
    visuals_dir: str | Path = VISUALS_DIR,
    with_shap: bool = True,
) -> dict:
    """Run the full training workflow and return a summary dictionary."""
    models_dir, visuals_dir = ensure_dir(models_dir), ensure_dir(visuals_dir)

    df = clean_data(load_data(data_path))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size)
    logger.info("Train: %d rows (%.1f%% attrition) | Test: %d rows (%.1f%% attrition)",
                len(X_train), y_train.mean() * 100, len(X_test), y_test.mean() * 100)

    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = get_candidates(imbalance, positive_weight, n_iter)

    fitted: dict[str, Pipeline] = {}
    rows: list[dict] = []
    best_params: dict[str, dict] = {}
    roc_data: dict[str, tuple] = {}

    for candidate in candidates:
        started = time.perf_counter()
        logger.info("Tuning %s (%s search)...", candidate.name, candidate.search)
        search = tune_candidate(candidate, X_train, y_train, imbalance, cv_folds)
        metrics = evaluate_model(search.best_estimator_, X_test, y_test)

        fpr, tpr, _ = roc_curve(y_test, metrics["_probabilities"])
        roc_data[candidate.name] = (fpr, tpr, metrics["roc_auc"])
        fitted[candidate.name] = search.best_estimator_
        best_params[candidate.name] = {k.replace("model__", ""): _to_python(v)
                                       for k, v in search.best_params_.items()}
        rows.append({
            "model": candidate.name,
            "cv_roc_auc": search.best_score_,
            **{k: v for k, v in metrics.items() if not k.startswith("_")},
            "_predictions": metrics["_predictions"],
            "tuning_seconds": round(time.perf_counter() - started, 1),
        })
        logger.info("  %s: CV AUC %.3f | test AUC %.3f, F1 %.3f, recall %.3f (%.1fs)",
                    candidate.name, search.best_score_, metrics["roc_auc"], metrics["f1"],
                    metrics["recall"], rows[-1]["tuning_seconds"])

    results = pd.DataFrame(rows).set_index("model")
    best_name = results["cv_roc_auc"].idxmax()
    best_pipeline = fitted[best_name]
    justification = justify_selection(results, best_name)
    logger.info(justification)

    # ---- artefacts --------------------------------------------------------
    comparison = results.drop(columns="_predictions")[
        ["cv_roc_auc", "accuracy", "precision", "recall", "f1", "roc_auc", "tuning_seconds"]]
    comparison.round(4).to_csv(models_dir / "model_comparison.csv")

    importance = get_feature_importance(best_pipeline, X, y)
    plot_feature_importance(importance["importance"], visuals_dir / "feature_importance.png",
                            best_name, errors=importance["std"],
                            xlabel="Drop in ROC-AUC when the feature is shuffled")
    plot_roc_curves(roc_data, visuals_dir / "roc_curves.png")
    plot_confusion_matrix(confusion_matrix(y_test, results.loc[best_name, "_predictions"]),
                          visuals_dir / "confusion_matrix.png", best_name)
    if with_shap:
        explain_with_shap(best_pipeline, X_train, X_test, visuals_dir / "shap_summary.png", best_name)

    test_metrics = {k: float(results.loc[best_name, k])
                    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": DECISION_THRESHOLD,
        "test_metrics": test_metrics,
        "imbalance_strategy": imbalance,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    model_path = models_dir / "best_model.pkl"
    joblib.dump(bundle, model_path)
    logger.info("Saved best model (%s) -> %s", best_name, model_path)

    summary = {
        "best_model": best_name,
        "justification": justification,
        "imbalance_strategy": imbalance,
        "test_metrics": test_metrics,
        "comparison": comparison.round(4).reset_index().to_dict(orient="records"),
        "best_params": best_params,
        "feature_importance": importance["importance"].round(4).to_dict(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    (models_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _to_python(value):
    """Convert numpy scalars to plain Python so the summary is JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and compare attrition models.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--imbalance", choices=["class_weight", "smote"], default="class_weight")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument("--n-iter", type=int, default=20, help="Random-search iterations per model.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--visuals-dir", default=str(VISUALS_DIR))
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    try:
        summary = run_training(args.data, args.imbalance, args.test_size, args.cv, args.n_iter,
                               args.models_dir, args.visuals_dir, with_shap=not args.skip_shap)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print("\nModel comparison")
    print(pd.DataFrame(summary["comparison"]).to_string(index=False))
    print(f"\nBest model: {summary['best_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
