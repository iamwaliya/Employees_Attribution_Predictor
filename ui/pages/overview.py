"""Overview: headline numbers, where attrition happens, and who is exposed."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils import DEPARTMENTS
from ui import charts
from ui import components as ui
from ui import format as fmt
from ui.config import USER
from ui.icons import ibtn, icon
from ui.pages.common import empty_workforce, go_to, rate_table
from ui.recommendations import LEVER_BY_KEY, ranked_levers

TENURE_BINS = [0, 1, 3, 5, 10, 200]
TENURE_LABELS = ["Under 1", "1 to 3", "3 to 5", "5 to 10", "10+"]


def _tenure_band(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["YearsAtCompany"], TENURE_BINS, labels=TENURE_LABELS, right=False)


def _kpis(scored: pd.DataFrame, active: pd.DataFrame, rate: float) -> None:
    t = ui.tk()
    left_n = int((scored["Attrition"] == "Yes").sum())
    high = active[active["RiskLevel"] == "High"]
    medium = active[active["RiskLevel"] == "Medium"]
    low = active[active["RiskLevel"] == "Low"]
    by_tenure = rate_table(scored, _tenure_band(scored))
    payroll_active = float(active["MonthlyIncome"].sum()) or 1.0
    payroll_high = float(high["MonthlyIncome"].sum())

    tenure_tip = ui.tip("By tenure (years)", [(r.Band if hasattr(r, "Band") else str(r[0]), f"{fmt.number(r.Employees)} staff")
                                              for r in by_tenure.rename(columns={by_tenure.columns[0]: "Band"}).itertuples()])
    rate_tip = ui.tip("Attrition by tenure (years)", [(str(r[1]), fmt.pct(r.Rate)) for r in by_tenure.itertuples()],
                      "Newer joiners leave most often")
    hist = pd.cut(active["RiskScore"], [i / 10 for i in range(11)], include_lowest=True).value_counts(sort=False).tolist()
    risk_tip = ui.tip("Active staff by risk band", [("Low (under 30%)", fmt.number(len(low))),
                                                    ("Medium (30-60%)", fmt.number(len(medium))),
                                                    ("High (60%+)", fmt.number(len(high)))])
    band_bar = ui.stacked_bar([
        ("Low", len(low), "var(--good)", ui.tip("Low risk", [("Employees", fmt.number(len(low)))])),
        ("Medium", len(medium), "var(--warn)", ui.tip("Medium risk", [("Employees", fmt.number(len(medium)))])),
        ("High", len(high), "var(--crit)", ui.tip("High risk", [("Employees", fmt.number(len(high)))]))], height=6)
    pay_chip = ui.chip(f"{payroll_high / payroll_active * 100:.1f}% of active payroll", "neutral")

    cards = [
        ui.kpi("Headcount", "users", fmt.number(len(scored)),
               f'<span class="num">{fmt.number(len(active))}</span> active, <span class="num">{fmt.number(left_n)}</span> left',
               ui.sparkline(by_tenure["Employees"], t["accent"]), tenure_tip),
        ui.kpi("Attrition rate", "trending-down", fmt.pct(rate),
               f'<span class="num">{fmt.number(left_n)}</span> of <span class="num">{fmt.number(len(scored))}</span> have left',
               ui.sparkline(by_tenure["Rate"], t["series_2"], baseline=rate), rate_tip),
        ui.kpi("High-risk employees", "shield-alert", fmt.number(len(high)),
               f'{fmt.pct(len(high) / max(len(active), 1))} of active staff<div class="kpi-bar">{band_bar}</div>',
               ui.sparkline(hist, t["crit"]), risk_tip),
        ui.kpi("Payroll at risk", "banknote", fmt.inr(payroll_high, compact=True),
               f'per month, {pay_chip}',
               "", ui.tip("High-risk employees", [("Monthly pay", fmt.inr(payroll_high)),
                                                  ("Average each", fmt.inr(payroll_high / max(len(high), 1))),
                                                  ("Share of payroll", fmt.pct(payroll_high / payroll_active))])),
    ]
    for column, markup in zip(st.columns(4, gap="small"), cards):
        with column:
            ui.render(markup)


def _department_card(scored: pd.DataFrame, rate: float) -> None:
    by_dept = rate_table(scored, "Department").sort_values("Rate", ascending=False)
    active_high = scored[(scored["Attrition"] == "No") & (scored["RiskLevel"] == "High")].groupby("Department").size()
    top = max(by_dept["Rate"].max() * 1.2, rate * 1.4, 0.05)
    rows = []
    for r in by_dept.itertuples():
        color = ui.dept_color(r.Department)
        diff = r.Rate - rate
        tooltip = ui.tip(r.Department, [("Employees", fmt.number(r.Employees)), ("Left", fmt.number(r.Left)),
                                        ("Attrition rate", fmt.pct(r.Rate)), ("Against company", fmt.signed_points(diff, 1)),
                                        ("High-risk active", fmt.number(int(active_high.get(r.Department, 0))))])
        chip = (f'<span class="delta {"crit" if diff > 0.002 else "good" if diff < -0.002 else "flat"}">'
                f'{fmt.signed_points(diff, 1) if abs(diff) > 0.0005 else "at average"}</span>')
        rows.append(
            f'<div class="rate-row has-tip"><div class="rr-name"><i class="swatch" style="background:{color}"></i>{r.Department}</div>'
            f'<div class="rr-bar">{ui.pill_bar(r.Rate, color, top, marker=rate, height=10)}</div>'
            f'<div class="rr-val num">{fmt.pct(r.Rate)}</div><div class="rr-diff">{chip}</div>'
            f'<div class="rr-n num">{fmt.number(r.Left)} of {fmt.number(r.Employees)}</div><span class="tip">{tooltip}</span></div>')
    head = ui.card_head("Attrition by department", "Share of each department's employees who have left",
                        ui.legend([(f"Company average {fmt.pct(rate)}", "var(--ink-3)")], shape="tick"))
    worst = by_dept.iloc[0]
    insight = (f'<p class="card-foot"><b>{worst["Department"]}</b> loses the most people: {fmt.pct(worst["Rate"])}, '
               f'{fmt.signed_points(worst["Rate"] - rate, 1).replace("+", "")} above the company average.</p>'
               if worst["Rate"] > rate else "")
    ui.render(f'<section class="card">{head}<div class="rate-list">{"".join(rows)}</div>{insight}</section>')


def _risk_card(active: pd.DataFrame) -> None:
    with st.container(key="card_risk"):
        high = int((active["RiskLevel"] == "High").sum())
        ui.render(ui.card_head("Risk exposure", "Predicted risk of leaving across active employees",
                               ui.legend([("Low", "var(--good)"), ("Medium", "var(--warn)"), ("High", "var(--crit)")])))
        charts.show(charts.risk_distribution(active["RiskScore"]), 210)
        ui.render(f'<p class="card-foot"><span class="num">{fmt.number(high)}</span> active employees score 60% or higher.</p>')


def _curve_card(key: str, title: str, sub: str, frame: pd.DataFrame, by: pd.Series, order: list[str], label: str, rate: float) -> None:
    table = rate_table(frame, by)
    table.columns = ["Band", "Employees", "Left", "Rate"]
    table["Band"] = table["Band"].astype(str)
    with st.container(key=key):
        ui.render(ui.card_head(title, sub, ui.legend([(f"Company {fmt.pct(rate)}", "var(--ink-3)")], shape="tick")))
        charts.show(charts.attrition_curve(table, rate, label, order), 200)


def _overtime_card(scored: pd.DataFrame, rate: float) -> None:
    t = ui.tk()
    groups = rate_table(scored, scored["OverTime"]).set_index("OverTime")
    over, none = groups.loc["Yes"], groups.loc["No"]
    top = max(over["Rate"], none["Rate"]) * 1.15 or 1.0
    ratio = over["Rate"] / none["Rate"] if none["Rate"] else float("nan")
    rows = []
    for label, row, color in (("Works overtime", over, t["series_2"]), ("No overtime", none, t["accent"])):
        tooltip = ui.tip(label, [("Employees", fmt.number(row["Employees"])), ("Left", fmt.number(row["Left"])),
                                 ("Attrition rate", fmt.pct(row["Rate"]))])
        rows.append(f'<div class="ot-row has-tip"><div class="ot-top"><span><i class="swatch" style="background:{color}"></i>{label}</span>'
                    f'<b class="num">{fmt.pct(row["Rate"])}</b></div>{ui.pill_bar(row["Rate"], color, top, height=12, tooltip=tooltip)}</div>')
    headline = (f'<div class="big-ratio"><b class="num">{ratio:.1f}x</b><span>as likely to leave when working overtime</span></div>'
                if ratio == ratio and ratio > 0 else "")
    ui.render(ui.card("Overtime effect", f'{headline}<div class="ot-list">{"".join(rows)}</div>',
                      "Attrition with and without regular overtime"))


def _watchlist(scored: pd.DataFrame, effects: pd.DataFrame) -> None:
    active = scored[scored["Attrition"] == "No"].sort_values("RiskScore", ascending=False).head(5)
    rows = []
    for r in active.itertuples():
        picks = ranked_levers(effects.loc[r.Index]) if r.Index in effects.index else []
        suggestion = (ui.chip(f"{picks[0][0].title}: {fmt.signed_points(picks[0][1])}", "accent", picks[0][0].icon)
                      if picks else ui.chip("No single fix stands out", "muted"))
        rows.append(
            f'<div class="watch-row"><div class="who">{ui.avatar(r.Name, r.Department)}<div><b>{ui.escape(r.Name)}</b>'
            f'<span>{ui.dept_tag(r.Department)}<span class="meta num">{fmt.inr(r.MonthlyIncome)} / month</span></span></div></div>'
            f'<div class="wr-risk">{ui.risk_chip(r.RiskLevel, r.RiskScore)}</div><div class="wr-fix">{suggestion}</div></div>')
    body = f'<div class="watch">{"".join(rows)}</div>'
    with st.container(key="card_watch"):
        left, right = st.columns([5, 1.2], vertical_alignment="center")
        left.html(ui.card_head("Needs attention first", "Highest predicted risk among active employees, with the most effective fix"))
        right.button("Open risk radar", key=ibtn("shield-alert", "to_risk"), on_click=go_to, args=("risk",), width="stretch")
        ui.render(body)


def render(scored: pd.DataFrame, effects: pd.DataFrame) -> None:
    if scored.empty:
        empty_workforce("Add your first employees, or load the 1,500 sample records, to see attrition trends and risk.")
        return
    active = scored[scored["Attrition"] == "No"]
    rate = float((scored["Attrition"] == "Yes").mean())

    _kpis(scored, active, rate)

    left, right = st.columns([7, 5], gap="small")
    with left:
        _department_card(scored, rate)
    with right:
        if active.empty:
            ui.render(ui.empty_state("No active employees", "Everyone in the database is marked as having left.", "user-x"))
        else:
            _risk_card(active)

    c1, c2, c3 = st.columns([5, 5, 4], gap="small")
    with c1:
        _curve_card("card_tenure", "Attrition by tenure", "Years at the company", scored, _tenure_band(scored),
                    TENURE_LABELS, "Tenure (years)", rate)
    with c2:
        _curve_card("card_sat", "Attrition by job satisfaction", "1 is low, 5 is high", scored,
                    scored["JobSatisfaction"].astype(str), ["1", "2", "3", "4", "5"], "Satisfaction", rate)
    with c3:
        _overtime_card(scored, rate)

    if not active.empty:
        _watchlist(scored, effects)
