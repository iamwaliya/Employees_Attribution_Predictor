"""Model: how well it performs, what it relies on, and retraining on the employees in the database."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.database import EmployeeDB
from src.insights import FEATURE_LABELS
from src.train_model import run_training
from src.utils import DATA_DIR, METRICS_PATH, VISUALS_DIR
from ui import components as ui
from ui import format as fmt
from ui import state
from ui.icons import ibtn

METRICS = [
    ("roc_auc", "ROC-AUC", "chart-line", "How well the model ranks people who left above people who stayed. 0.5 is a coin flip, 1.0 is perfect."),
    ("recall", "Recall", "radar", "Share of people who actually left that the model caught."),
    ("precision", "Precision", "target", "Share of flagged people who actually left."),
    ("f1", "F1 score", "scale", "A single score that balances precision and recall."),
    ("accuracy", "Accuracy", "badge-check", "Share of all predictions that were right. Misleading here: most people stay."),
]
COMPARE_COLUMNS = [("cv_roc_auc", "CV AUC"), ("roc_auc", "Test AUC"), ("accuracy", "Accuracy"),
                   ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]


def load_summary() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metric_cards(summary: dict) -> None:
    metrics = summary["test_metrics"]
    cards = []
    for key, label, icon_name, hint in METRICS:
        value = float(metrics[key])
        bar = ui.pill_bar(value, "var(--accent)", 1.0, height=6, still=True)
        cards.append(ui.kpi(label, icon_name, f"{value:.2f}", f'<div class="kpi-bar">{bar}</div><div class="kpi-hint">{ui.escape(hint)}</div>'))
    for column, markup in zip(st.columns(5, gap="small"), cards):
        with column:
            ui.render(markup)


def _comparison(summary: dict) -> None:
    best = summary["best_model"]
    frame = pd.DataFrame(summary["comparison"])
    header = '<div class="cmp-grid cmp-head"><div>Model</div>' + "".join(f"<div>{label}</div>" for _, label in COMPARE_COLUMNS) + "</div>"
    rows = []
    for r in frame.itertuples(index=False):
        row = r._asdict()
        selected = row["model"] == best
        cells = []
        for key, _ in COMPARE_COLUMNS:
            value = float(row[key])
            top = value >= float(frame[key].max()) - 1e-9
            color = "var(--accent)" if selected else "var(--ink-3)"
            cells.append(f'<div class="cmp-cell{" top" if top else ""}"><b class="num">{value:.3f}</b>'
                         f'{ui.pill_bar(value, color, 1.0, height=4, still=True)}</div>')
        tag = ui.chip("Selected", "accent", "check") if selected else ""
        seconds = row.get("tuning_seconds")
        sub = f'<span class="dim num">tuned in {seconds:g}s</span>' if seconds else ""
        rows.append(f'<div class="cmp-grid cmp-row{" sel" if selected else ""}"><div class="cmp-name"><b>{ui.escape(row["model"])}</b>{tag}{sub}</div>{"".join(cells)}</div>')
    ui.render(ui.card("Model comparison", header + "".join(rows),
                      "Three models tuned with cross-validation. Bold marks the best value in each column."))


def _importance(summary: dict) -> None:
    items = sorted(summary.get("feature_importance", {}).items(), key=lambda kv: -kv[1])
    top = max((v for _, v in items), default=1.0) or 1.0
    rows = []
    for feature, value in items:
        tip = ui.tip(FEATURE_LABELS.get(feature, feature), [("Drop in ROC-AUC when shuffled", f"{value:.3f}")],
                     "Bigger means the model leans on it more")
        rows.append(f'<div class="fi-row has-tip"><span>{FEATURE_LABELS.get(feature, feature)}</span>'
                    f'{ui.pill_bar(max(value, 0), "var(--accent)", top, height=8, still=True)}<b class="num">{value:.3f}</b>'
                    f'<span class="tip">{tip}</span></div>')
    ui.render(ui.card("What the model relies on", f'<div class="fi">{"".join(rows)}</div>',
                      "Permutation importance on the test set"))


def _diagnostics() -> None:
    images = [("roc_curves.png", "ROC curves"), ("confusion_matrix.png", "Confusion matrix"), ("feature_importance.png", "Feature importance")]
    available = [(VISUALS_DIR / f, c) for f, c in images if (VISUALS_DIR / f).exists()]
    if not available:
        return
    with st.expander("Diagnostic plots"):
        st.caption("Static plots saved by the training run. They use a light background in either theme.")
        for column, (path, caption) in zip(st.columns(len(available)), available):
            column.image(str(path), caption=caption)


def _retrain(db: EmployeeDB) -> None:
    ready, message = db.training_readiness()
    with st.container(key="card_retrain"):
        ui.render(ui.card_head("Retrain on your employees",
                               "Set Status to Left for people who actually left, then retrain so predictions reflect your company."))
        ui.render(ui.notice("good" if ready else "warn", "Ready to train" if ready else "Not enough data yet", message))
        if st.button("Retrain model", key=ibtn("cpu", "retrain"), type="primary", disabled=not ready):
            export_path = db.export_csv(DATA_DIR / "employees_export.csv")
            try:
                with st.spinner("Tuning Logistic Regression, Random Forest and XGBoost. This takes about a minute."):
                    result = run_training(export_path, n_iter=10, with_shap=False)
            except Exception as exc:  # surface any training failure instead of a stack trace
                ui.render(ui.notice("crit", "Training failed", str(exc)))
            else:
                state.get_predictor.clear()
                state.refresh_data()
                state.flash(f"Retrained. Best model: {result['best_model']} (test ROC-AUC {result['test_metrics']['roc_auc']:.2f}).")
                st.rerun()


def render(db: EmployeeDB) -> None:
    summary = load_summary()
    if summary:
        bundle = state.get_predictor().bundle
        trained = str(bundle.get("trained_at", "unknown"))[:16].replace("T", " ")
        total = summary["train_rows"] + summary["test_rows"]
        chips = "".join([
            ui.chip(summary["best_model"], "accent", "cpu"),
            ui.chip(summary["imbalance_strategy"].replace("_", " ").capitalize(), "neutral", "layers"),
            ui.chip(f"Trained {trained} UTC", "neutral", "clock"),
            ui.chip(f"{fmt.number(total)} employees", "neutral", "users"),
        ])
        ui.render(f'<div class="chip-row">{chips}</div>')
        _metric_cards(summary)
        ui.render('<p class="toolbar-note">Scores on a held-out test set the model never saw during training or tuning.</p>')
        left, right = st.columns([7, 4], gap="small")
        with left:
            _comparison(summary)
        with right:
            _importance(summary)
        ui.render(ui.notice("info", "Why this model", summary["justification"]))
        _diagnostics()
    else:
        ui.render(ui.notice("warn", "No saved metrics", "Train the model first: python -m src.train_model"))
    _retrain(db)
