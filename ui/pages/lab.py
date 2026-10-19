"""What-if lab: change a profile and watch the predicted risk, drivers and suggested interventions respond."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from src.database import EmployeeDB
from src.insights import describe_value, feature_impacts, population_baseline
from src.predict import AttritionPredictor, InputValidationError, PredictionError
from src.utils import DEPARTMENTS, FEATURE_COLUMNS, risk_level
from ui import components as ui
from ui import format as fmt
from ui.icons import ibtn
from ui.recommendations import LEVERS, MIN_USEFUL_DROP, apply_lever_to_profile

SCORE_HELP = {"JobSatisfaction": "1 very low, 5 very high", "WorkLifeBalance": "1 poor, 5 excellent",
              "PerformanceRating": "1 poor, 5 outstanding"}


# --------------------------------------------------------------------------- #
# Control definitions
# --------------------------------------------------------------------------- #
def _fmt_value(field: str, value) -> str:
    if field == "MonthlyIncome":
        return fmt.inr(value)
    if field == "YearsAtCompany":
        return f"{float(value):g} yr"
    if field == "DistanceFromHome":
        return f"{float(value):g} km"
    if field == "Age":
        return f"{int(value)}"
    return f"{int(value)} / 5"


def _bounds(field: str, start: float) -> tuple[float, float, float]:
    """(min, max, step) wide enough to contain the starting value."""
    if field == "MonthlyIncome":
        return min(10_000.0, float(start)), max(300_000.0, math.ceil(float(start) * 2 / 1000) * 1000), 1000.0
    if field == "Age":
        return 18, 70, 1
    if field == "YearsAtCompany":
        return 0.0, max(40.0, math.ceil(float(start))), 0.5
    if field == "DistanceFromHome":
        return 0.0, max(100.0, math.ceil(float(start))), 1.0
    return 1, 5, 1


GROUPS = [
    ("Role and pay", [("Age", "Age", "user-round"), ("MonthlyIncome", "Monthly income", "banknote"), ("YearsAtCompany", "Time at company", "hourglass")]),
    ("How work feels", [("JobSatisfaction", "Job satisfaction", "heart-handshake"), ("WorkLifeBalance", "Work-life balance", "scale"),
                        ("PerformanceRating", "Performance rating", "target")]),
    ("Work pattern", [("DistanceFromHome", "Commute distance", "map-pin")]),
]


def _key(prefix: str, field: str) -> str:
    return f"{prefix}_{field}"


def _start_profile(db: EmployeeDB, scored: pd.DataFrame, employee_id: int | None) -> dict:
    if employee_id is not None:
        record = db.get(employee_id)
        if record:
            return {c: record[c] for c in FEATURE_COLUMNS}
    return dict(population_baseline(scored))


def _read_profile(prefix: str) -> dict:
    values = {f: st.session_state[_key(prefix, f)] for f in FEATURE_COLUMNS}
    for f in ("Age", "JobSatisfaction", "WorkLifeBalance", "PerformanceRating"):
        values[f] = int(values[f])
    for f in ("MonthlyIncome", "YearsAtCompany", "DistanceFromHome"):
        values[f] = float(values[f])
    return values


def _apply_lever(lever_key: str, prefix: str, start: dict) -> None:
    from ui.recommendations import LEVER_BY_KEY
    changed = apply_lever_to_profile(LEVER_BY_KEY[lever_key], _read_profile(prefix))
    for field in FEATURE_COLUMNS:
        if field in ("Department", "OverTime"):
            st.session_state[_key(prefix, field)] = changed[field]
            continue
        low, high, _ = _bounds(field, start[field])
        value = min(max(changed[field], low), high)
        st.session_state[_key(prefix, field)] = int(value) if field in ("Age", "JobSatisfaction", "WorkLifeBalance", "PerformanceRating") else float(value)


def _reset(ver_key: str) -> None:
    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1


# --------------------------------------------------------------------------- #
# Left: controls
# --------------------------------------------------------------------------- #
def _slider_row(prefix: str, field: str, label: str, icon_name: str, start: dict) -> None:
    key = _key(prefix, field)
    low, high, step = _bounds(field, start[field])
    cast = (lambda v: int(v)) if field in ("Age", "JobSatisfaction", "WorkLifeBalance", "PerformanceRating") else float
    st.session_state.setdefault(key, cast(start[field]))
    current = st.session_state[key]
    changed = abs(float(current) - float(start[field])) > 1e-9
    was = f'<span class="was">was {_fmt_value(field, start[field])}</span>' if changed else ""
    help_text = SCORE_HELP.get(field, "")
    ui.render(f'<div class="ctl-head"><span class="ctl-label">{ui.icon(icon_name, 15)}{label}</span>'
              f'<span class="ctl-val{" changed" if changed else ""}">{was}<b class="num">{_fmt_value(field, current)}</b></span></div>')
    st.slider(label, low if field != "MonthlyIncome" else low, high, key=key, step=step, label_visibility="collapsed",
              format="%d" if step >= 1 else "%.1f", help=help_text or None)


def _controls(prefix: str, start: dict, reset_key: str) -> None:
    with st.container(key="card_labcontrols"):
        a, b = st.columns([5, 2], vertical_alignment="center")
        a.html(ui.card_head("Profile", "Drag a slider and the prediction updates instantly"))
        b.button("Reset", key=ibtn("rotate-ccw", "lab_reset"), on_click=_reset, args=(reset_key,), width="stretch")

        left, right = st.columns(2, gap="small")
        dept_key, ot_key = _key(prefix, "Department"), _key(prefix, "OverTime")
        st.session_state.setdefault(dept_key, start["Department"])
        st.session_state.setdefault(ot_key, start["OverTime"])
        left.selectbox("Department", DEPARTMENTS, key=dept_key)
        right.segmented_control("Works overtime", ["No", "Yes"], key=ot_key, selection_mode="single")
        if st.session_state.get(ot_key) is None:
            st.session_state[ot_key] = start["OverTime"]
        for title, fields in GROUPS:
            ui.render(f'<div class="ctl-group">{title}</div>')
            for field, label, icon_name in fields:
                _slider_row(prefix, field, label, icon_name, start)


# --------------------------------------------------------------------------- #
# Right: results
# --------------------------------------------------------------------------- #
def _tile(label: str, value: str, foot: str = "", tone: str = "") -> str:
    return (f'<div class="tile {tone}"><span class="tile-label">{label}</span><b class="num">{value}</b>'
            f'<span class="tile-foot">{foot}</span></div>')


def _driver_bars(impacts: pd.Series, profile: dict) -> str:
    top = impacts.reindex(impacts.abs().sort_values(ascending=False).index).head(6)
    biggest = max(float(top.abs().max()), 0.2)
    rows = []
    for feature, impact in top.items():
        width = min(abs(float(impact)) / biggest, 1.0) * 100
        up = impact > 0
        odds = math.exp(abs(float(impact)))
        tip = ui.tip(describe_value(feature, profile[feature]),
                     [("Effect on risk", "raises it" if up else "lowers it"), ("Odds of leaving", f"{odds:.1f}x {'higher' if up else 'lower'}")],
                     "Compared with a typical employee")
        side = (f'<i class="drv-fill up" style="width:{width / 2:.1f}%"></i>' if up
                else f'<i class="drv-fill down" style="width:{width / 2:.1f}%"></i>')
        rows.append(f'<div class="drv has-tip"><span class="drv-name">{ui.escape(describe_value(feature, profile[feature]))}</span>'
                    f'<span class="drv-track"><i class="drv-axis"></i>{side}</span><span class="tip">{tip}</span></div>')
    legend = ui.legend([("Lowers risk", "var(--accent)"), ("Raises risk", "var(--crit)")])
    return f'<div class="drivers">{"".join(rows)}</div><div class="drv-legend">{legend}</div>'


def _recommendations(predictor: AttritionPredictor, profile: dict, current: float, prefix: str, start: dict) -> None:
    frame = pd.DataFrame([profile])
    options = []
    for lever in LEVERS:
        if not bool(lever.applies(frame).iloc[0]):
            continue
        new_risk = float(predictor.predict_frame(lever.apply(frame))[0])
        if new_risk - current <= -MIN_USEFUL_DROP:
            options.append((lever, new_risk))
    options.sort(key=lambda pair: pair[1])
    with st.container(key="card_labrecs"):
        ui.render(ui.card_head("Recommended actions", "Model estimates of what would lower this score"))
        if not options:
            kind = "good" if current < 0.3 else "info"
            ui.render(ui.notice(kind, "No intervention stands out" if current >= 0.3 else "No action needed",
                                "None of the standard levers would move this score by 2 points or more." if current >= 0.3
                                else "This profile already has a low predicted risk."))
            return
        for rank, (lever, new_risk) in enumerate(options[:3]):
            label = "Intervention suggested" if rank == 0 else "Also consider"
            with st.container(key=f"rec_{lever.key}"):
                a, b = st.columns([6.5, 1.5], vertical_alignment="center", gap="small")
                a.html(f'<div class="rec{" top" if rank == 0 else ""}"><span class="rec-ic">{ui.icon(lever.icon, 18)}</span>'
                       f'<div><span class="rec-kicker">{label}</span><b>{ui.escape(lever.title)}</b>'
                       f'<span class="rec-action">{ui.escape(lever.action)}</span></div>'
                       f'<div class="rec-eff"><span class="num">{current * 100:.0f}% to {new_risk * 100:.0f}%</span>'
                       f'{ui.delta_chip(new_risk - current)}</div></div>')
                b.button("Apply", key=ibtn("sparkles", f"apply_{lever.key}"), on_click=_apply_lever,
                         args=(lever.key, prefix, start), width="stretch", type="primary" if rank == 0 else "secondary")


def _results(predictor: AttritionPredictor, scored: pd.DataFrame, prefix: str, start: dict, original: float, baseline_label: str) -> None:
    profile = _read_profile(prefix)
    profile["Department"] = st.session_state[_key(prefix, "Department")]
    profile["OverTime"] = st.session_state[_key(prefix, "OverTime")]
    try:
        result = predictor.predict(profile)
    except InputValidationError as exc:
        ui.render(ui.notice("warn", "Check the inputs", "; ".join(f"{k} {v}" for k, v in exc.errors.items())))
        return
    except PredictionError as exc:
        ui.render(ui.notice("crit", "Prediction failed", str(exc)))
        return

    probability = result.probability
    level = result.risk_level
    tone = ui.risk_var(level)
    delta = probability - original
    pay_delta = profile["MonthlyIncome"] - float(start["MonthlyIncome"])
    verdict = "likely to leave" if result.prediction == "Yes" else "likely to stay"

    with st.container(key="labresult"):
        with st.container(key="card_labscore"):
            head = (f'<div class="lab-head"><div><h3 class="card-title">Predicted risk</h3><p class="card-sub">'
                    f'This profile is {verdict}.</p></div>{ui.risk_chip(level, probability)}</div>')
            tiles = "".join([
                _tile("Against " + baseline_label, ui.delta_chip(delta) if abs(delta) >= 0.005 else '<span class="delta flat">no change</span>',
                      f'<span class="num">{original * 100:.0f}%</span> to <span class="num">{probability * 100:.0f}%</span>'),
                _tile("Pay change", f'{"+" if pay_delta > 0 else ""}{fmt.inr(pay_delta)}' if abs(pay_delta) >= 1 else "No change", "per month"),
                _tile("Model cut-off", f'{result.threshold * 100:.0f}%', "flagged as leaver above this"),
            ])
            ui.render(f'{head}{ui.gauge(probability, tone, "probability of leaving", still=True)}<div class="tiles">{tiles}</div>')

        impacts = feature_impacts(predictor, pd.DataFrame([profile]), population_baseline(scored)).iloc[0]
        with st.container(key="card_labdrivers"):
            ui.render(ui.card_head("What drives this score", "Each factor against a typical employee"))
            if float(impacts.abs().max()) < 0.05:
                ui.render('<p class="card-foot">This profile matches the typical employee, so no factor stands out. '
                          'Move a slider to see what pushes the score up or down.</p>')
            ui.render(_driver_bars(impacts, profile))

        _recommendations(predictor, profile, probability, prefix, start)


def render(db: EmployeeDB, predictor: AttritionPredictor, scored: pd.DataFrame) -> None:
    # Arriving from People / Risk radar: preselect that employee.
    if "lab_employee" in st.session_state:
        st.session_state["lab_choice"] = st.session_state.pop("lab_employee")

    ordered = scored.sort_values("Name") if not scored.empty else scored
    names = {int(r.EmployeeID): f"{r.Name}  (ID {r.EmployeeID})" for r in ordered.itertuples()}
    options: list[int | None] = [None, *names]
    if "lab_choice" not in st.session_state:  # first visit: start from the person most at risk
        active = scored[scored["Attrition"] == "No"]
        st.session_state["lab_choice"] = int(active.loc[active["RiskScore"].idxmax(), "EmployeeID"]) if not active.empty else None
    if st.session_state.get("lab_choice") not in options:
        st.session_state["lab_choice"] = None
    with st.container(key="labpick"):
        st.selectbox("Start from", options, key="lab_choice", format_func=lambda v: "Typical employee (new profile)" if v is None else names[v])
    employee_id = st.session_state["lab_choice"]

    reset_key = f"lab_ver_{employee_id}"
    prefix = f"lab{st.session_state.get(reset_key, 0)}_{employee_id or 'new'}"
    start = _start_profile(db, scored, employee_id)
    try:
        original = float(predictor.predict(start).probability)
    except (InputValidationError, PredictionError) as exc:
        st.error(str(exc))
        return
    baseline_label = "saved profile" if employee_id else "typical employee"

    left, right = st.columns([5, 6], gap="small")
    with left:
        _controls(prefix, start, reset_key)
    with right:
        _results(predictor, scored, prefix, start, original, baseline_label)
