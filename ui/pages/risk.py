"""Risk radar: active employees ranked by predicted risk, why, and what would help."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.insights import feature_impacts, population_baseline
from src.predict import AttritionPredictor
from src.utils import DEPARTMENTS
from ui import components as ui
from ui import format as fmt
from ui.icons import ibtn
from ui.pages.common import empty_workforce, open_lab
from ui.recommendations import LEVERS, driver_labels, ranked_levers

TOP_CHOICES = {"10": 10, "25": 25, "50": 50, "100": 100}


def _filters(active: pd.DataFrame) -> tuple[list[str], list[str], int]:
    with st.container(key="card_riskfilters"):
        a, b, c = st.columns([3.2, 4.4, 2.4], vertical_alignment="center")
        departments = a.multiselect("Department", DEPARTMENTS, placeholder="All departments", key="risk_depts",
                                    label_visibility="collapsed")
        levels = b.pills("Risk level", ["High", "Medium", "Low"], selection_mode="multi", key="risk_levels",
                         format_func=lambda k: f"{k} risk  {fmt.number(int((active['RiskLevel'] == k).sum()))}",
                         label_visibility="collapsed") or []
        top = c.segmented_control("Show", list(TOP_CHOICES), default="25", key="risk_top", label_visibility="collapsed") or "25"
    return list(departments), list(levels), TOP_CHOICES[top]


def _mix_card(pool: pd.DataFrame, effects: pd.DataFrame) -> None:
    """Which single action helps the most people in the current list."""
    counts = {lever.key: 0 for lever in LEVERS}
    saved = {lever.key: 0.0 for lever in LEVERS}
    for idx in pool.index:
        picks = ranked_levers(effects.loc[idx]) if idx in effects.index else []
        if picks:
            lever, delta = picks[0]
            counts[lever.key] += 1
            saved[lever.key] += -delta
    total = max(sum(counts.values()), 1)
    rows = []
    for lever in sorted(LEVERS, key=lambda lv: -counts[lv.key]):
        n = counts[lever.key]
        if n == 0:
            continue
        average = saved[lever.key] / n
        tip = ui.tip(lever.title, [("People it helps most", fmt.number(n)), ("Average drop", f"{average * 100:.0f} pts")], lever.action)
        rows.append(f'<div class="mix-row has-tip"><div class="mix-top"><span>{ui.icon(lever.icon, 15)}{ui.escape(lever.title)}</span>'
                    f'<b class="num">{fmt.number(n)}</b></div>{ui.pill_bar(n, "var(--accent)", total, height=8)}'
                    f'<span class="tip">{tip}</span></div>')
    body = "".join(rows) or '<p class="card-foot">No single action stands out for this list.</p>'
    ui.render(ui.card("Where to act", f'<div class="mix">{body}</div>',
                      "The single action that lowers each listed person's score the most, counted across the list"))


def render(predictor: AttritionPredictor, scored: pd.DataFrame, effects: pd.DataFrame) -> None:
    active = scored[scored["Attrition"] == "No"]
    if scored.empty:
        empty_workforce("Add employees to see who is most likely to leave.")
        return
    if active.empty:
        ui.render(ui.empty_state("No active employees", "Everyone in the database is marked as having left.", "user-x"))
        return

    departments, levels, top_n = _filters(active)
    pool = active
    if departments:
        pool = pool[pool["Department"].isin(departments)]
    if levels:
        pool = pool[pool["RiskLevel"].isin(levels)]
    total_matching = len(pool)
    pool = pool.sort_values("RiskScore", ascending=False).head(top_n)
    if pool.empty:
        ui.render(ui.empty_state("No employees match", "Try different departments or risk levels.", "search"))
        return

    impacts = feature_impacts(predictor, pool, population_baseline(scored))

    table_rows, export = [], []
    for rank, r in enumerate(pool.itertuples(), start=1):
        drivers = driver_labels(pd.Series(r._asdict()), impacts.loc[r.Index])
        picks = ranked_levers(effects.loc[r.Index]) if r.Index in effects.index else []
        driver_html = "".join(ui.chip(d, "muted") for d in drivers) or '<span class="dim">No standout factor</span>'
        if picks:
            lever, delta = picks[0]
            fix = ui.chip(f"{lever.title} {fmt.signed_points(delta)}", "accent", lever.icon, tip=ui.tip(
                f"Intervention suggested: {lever.title}", [("Change", lever.action), ("Estimated risk", fmt.signed_points(delta))],
                "Model estimate, not a guarantee."))
            fix_text = f"{lever.title} ({fmt.signed_points(delta)})"
        else:
            fix, fix_text = ui.chip("No single fix stands out", "muted"), ""
        table_rows.append((r.EmployeeID, r.Name, (
            f'<div class="rk-grid"><div class="pc r dim num">{rank}</div>'
            f'<div class="pc who">{ui.avatar(r.Name, r.Department, 32)}<div><b>{ui.escape(r.Name)}</b>'
            f'<span class="meta">{ui.dept_tag(r.Department)}</span></div></div>'
            f'<div class="pc">{ui.risk_chip(r.RiskLevel, r.RiskScore)}</div>'
            f'<div class="pc chips">{driver_html}</div><div class="pc">{fix}</div></div>')))
        export.append({"ID": r.EmployeeID, "Name": r.Name, "Department": r.Department, "Risk score %": round(r.RiskScore * 100),
                       "Level": r.RiskLevel, "Main risk factors": "; ".join(drivers), "Suggested action": fix_text})

    _mix_card(pool, effects)
    with st.container(key="card_risklist"):
        a, b = st.columns([8, 1.6], vertical_alignment="center")
        a.html(ui.card_head("Highest predicted risk", f"{fmt.number(len(pool))} of {fmt.number(total_matching)} matching active employees"))
        b.download_button("Download list", pd.DataFrame(export).to_csv(index=False).encode("utf-8"), key=ibtn("file-down", "risk_dl"),
                          file_name="at_risk_employees.csv", mime="text/csv", width="stretch")
        with st.container(key="rkrow_head"):
            st.columns([14, 0.6])[0].html('<div class="rk-grid rk-head"><div class="pc r">#</div><div class="pc">Employee</div><div class="pc">Risk</div>'
                                          '<div class="pc">Main factors</div><div class="pc">Suggested action</div></div>')
        for employee_id, name, html in table_rows:
            with st.container(key=f"rkrow_{employee_id}"):
                cols = st.columns([14, 0.6], vertical_alignment="center", gap="small")
                cols[0].html(html)
                cols[1].button("Open in what-if lab", key=f"icon__flask_conical__r{employee_id}", type="tertiary",
                               help=f"Open {name} in the what-if lab", on_click=open_lab, args=(int(employee_id),))
        ui.render('<p class="card-foot">Factors compare each person with the typical employee. '
                  'Suggested actions are model estimates of what would lower the score, not guarantees.</p>')
