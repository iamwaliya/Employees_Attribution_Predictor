"""People: search and quick filters, a selectable table with risk chips, and add / edit / delete / import / export."""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import streamlit as st

from src.database import EmployeeDB
from ui import components as ui
from ui import dialogs
from ui import format as fmt
from ui.icons import ibtn
from ui.pages.common import empty_workforce, open_lab

PAGE_SIZE = 20

# label -> (kind, predicate). "team" filters combine with OR, the rest narrow the list (AND).
QUICK_FILTERS = {
    "High overtime": ("attr", lambda f: f["OverTime"] == "Yes"),
    "Low satisfaction": ("attr", lambda f: f["JobSatisfaction"] <= 2),
    "High risk": ("attr", lambda f: (f["RiskLevel"] == "High") & (f["Attrition"] == "No")),
    "New joiners": ("attr", lambda f: f["YearsAtCompany"] < 1),
    "Engineering team": ("team", lambda f: f["Department"] == "Engineering"),
    "Sales team": ("team", lambda f: f["Department"] == "Sales"),
}
SORTS = {
    "Highest risk": ("RiskScore", False),
    "Name A to Z": ("Name", True),
    "Highest income": ("MonthlyIncome", False),
    "Longest tenure": ("YearsAtCompany", False),
    "Newest first": ("EmployeeID", False),
}
STATUSES = ["All", "Active", "Left"]


def apply_filters(scored: pd.DataFrame, search: str = "", status: str = "All", quick: list[str] | None = None,
                  sort: str = "Highest risk") -> pd.DataFrame:
    """Filter and sort the workforce table (pure function, unit-tested)."""
    view = scored
    term = (search or "").strip().lower()
    if term:
        hit = (view["Name"].str.lower().str.contains(term, regex=False)
               | view["EmployeeID"].astype(str).str.contains(term, regex=False)
               | view["Department"].str.lower().str.contains(term, regex=False))
        view = view[hit]
    if status in ("Active", "Left"):
        view = view[view["Attrition"] == ("Yes" if status == "Left" else "No")]
    quick = quick or []
    teams = [QUICK_FILTERS[q][1](view) for q in quick if QUICK_FILTERS[q][0] == "team"]
    if teams:
        mask = teams[0]
        for extra in teams[1:]:
            mask = mask | extra
        view = view[mask]
    for label in quick:
        kind, predicate = QUICK_FILTERS[label]
        if kind == "attr":
            view = view[predicate(view)]
    column, ascending = SORTS.get(sort, SORTS["Highest risk"])
    if column == "RiskScore":  # people who already left have no actionable risk: list them last
        return (view.assign(_left=(view["Attrition"] == "Yes")).sort_values(["_left", column], ascending=[True, False], kind="stable")
                .drop(columns="_left"))
    return view.sort_values(column, ascending=ascending, kind="stable")


# --------------------------------------------------------------------------- #
# Selection (kept in session state so it survives paging and filtering)
# --------------------------------------------------------------------------- #
def _selection() -> set[int]:
    return st.session_state.setdefault("people_sel", set())


def _toggle(employee_id: int) -> None:
    selected = _selection()
    selected.symmetric_difference_update({employee_id})


def _select_page(ids: list[int]) -> None:
    selected = _selection()
    if st.session_state.get("people_all"):
        selected.update(ids)
    else:
        selected.difference_update(ids)


def _clear_selection() -> None:
    st.session_state["people_sel"] = set()
    st.session_state["people_all"] = False


# --------------------------------------------------------------------------- #
def _row_cells(r) -> str:
    left = r.Attrition == "Yes"
    overtime = ui.chip("Overtime", "accent", "timer") if r.OverTime == "Yes" else '<span class="dim">No</span>'
    risk = '<span class="dim">Left company</span>' if left else ui.risk_chip(r.RiskLevel, r.RiskScore)
    return (
        f'<div class="pc who">{ui.avatar(r.Name, r.Department, 32)}<div><b>{ui.escape(r.Name)}</b>'
        f'<span class="meta num">ID {r.EmployeeID}</span></div></div>'
        f'<div class="pc">{ui.dept_tag(r.Department)}</div>'
        f'<div class="pc r num">{fmt.inr(r.MonthlyIncome)}</div>'
        f'<div class="pc r num">{r.YearsAtCompany:g} yr</div>'
        f'<div class="pc">{ui.score_dots(int(r.JobSatisfaction))}</div>'
        f'<div class="pc">{overtime}</div>'
        f'<div class="pc">{ui.status_chip(left)}</div>'
        f'<div class="pc">{risk}</div>')


HEADER_CELLS = ["Employee", "Department", "Income / month", "Tenure", "Satisfaction", "Overtime", "Status", "Risk"]
HEADER_HTML = "".join(f'<div class="pc{" r" if label in ("Income / month", "Tenure") else ""}">{label}</div>' for label in HEADER_CELLS)
COLS = [0.42, 10.2, 1.1]


def _toolbar(db: EmployeeDB, total: int) -> None:
    with st.container(key="people_toolbar"):
        left, right = st.columns([4, 6], vertical_alignment="center")
        left.html(f'<p class="toolbar-note"><span class="num">{fmt.number(total)}</span> employees in the database</p>')
        with right:
            with st.container(horizontal=True, horizontal_alignment="right", gap="small", key="people_actions"):
                if st.button("Import", key=ibtn("upload", "import")):
                    dialogs.import_dialog(db)
                st.download_button("Export", db.list().to_csv(index=False).encode("utf-8"), key=ibtn("download", "export"),
                                   file_name=f"employees_{datetime.now():%Y%m%d}.csv", mime="text/csv")
                with st.popover("More", key=ibtn("ellipsis", "more")):
                    if st.button("Reset to sample data", key=ibtn("rotate-ccw", "reset"), type="tertiary", width="stretch"):
                        dialogs.reset_dialog(db)
                    if st.button("Delete all employees", key="danger__" + ibtn("trash-2", "wipe"), type="tertiary", width="stretch"):
                        dialogs.wipe_dialog(db)
                if st.button("Add employee", key=ibtn("plus", "add"), type="primary"):
                    dialogs.add_dialog(db)


def _filters(scored: pd.DataFrame) -> tuple[str, str, list[str], str]:
    with st.container(key="card_filters"):
        a, b, c = st.columns([5, 2.2, 2.4], vertical_alignment="bottom")
        search = a.text_input("Search", key="people_search", placeholder="Search by name, ID or department",
                              label_visibility="collapsed")
        status = b.segmented_control("Status", STATUSES, default="All", key="people_status", label_visibility="collapsed") or "All"
        sort = c.selectbox("Sort by", list(SORTS), key="people_sort", label_visibility="collapsed")
        counts = {k: int(pred(scored).sum()) for k, (_, pred) in QUICK_FILTERS.items()}
        quick = st.pills("Quick filters", list(QUICK_FILTERS), selection_mode="multi", key="people_quick",
                         format_func=lambda k: f"{k}  {fmt.number(counts[k])}", label_visibility="collapsed") or []
    return search, status, list(quick), sort


def _pager(page: int, pages: int, shown: int, total: int, first: int, last: int) -> None:
    left, mid, right = st.columns([5, 4, 3], vertical_alignment="center")
    left.html(f'<p class="toolbar-note">Showing <span class="num">{fmt.number(first)}-{fmt.number(last)}</span> of '
              f'<span class="num">{fmt.number(shown)}</span>'
              + (f' <span class="dim">(filtered from {fmt.number(total)})</span>' if shown != total else "") + "</p>")
    with right:
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small", key="pager"):
            st.button("Previous", key=ibtn("chevron-left", "prev"), disabled=page <= 0, on_click=lambda: st.session_state.update(people_page=page - 1))
            st.html(f'<span class="toolbar-note">Page <span class="num">{page + 1}</span> of <span class="num">{pages}</span></span>')
            st.button("Next", key=ibtn("chevron-right", "next"), disabled=page >= pages - 1, on_click=lambda: st.session_state.update(people_page=page + 1))


def render(db: EmployeeDB, scored: pd.DataFrame) -> None:
    if scored.empty:
        _toolbar(db, 0)
        empty_workforce("Add your first employee, import a CSV, or load the 1,500 sample records.")
        return

    _toolbar(db, len(scored))
    search, status, quick, sort = _filters(scored)

    signature = (search, status, tuple(quick), sort, len(scored))
    if st.session_state.get("people_sig") != signature:
        st.session_state["people_sig"] = signature
        st.session_state["people_page"] = 0
    view = apply_filters(scored, search, status, quick, sort)

    if view.empty:
        ui.render(ui.empty_state("No employees match", "Try a different search or clear some filters.", "search"))
        return

    pages = max(1, math.ceil(len(view) / PAGE_SIZE))
    page = min(max(int(st.session_state.get("people_page", 0)), 0), pages - 1)
    st.session_state["people_page"] = page
    chunk = view.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    ids = [int(i) for i in chunk["EmployeeID"]]
    selected = _selection()
    selected &= set(int(i) for i in scored["EmployeeID"])

    # Bulk bar
    if selected:
        chosen = scored[scored["EmployeeID"].isin(selected)]
        with st.container(key="bulkbar"):
            a, b, c, d = st.columns([6, 1.3, 1.1, 1.3], vertical_alignment="center", gap="small")
            a.html(f'<p class="bulk-text"><b class="num">{fmt.number(len(selected))}</b> selected</p>')
            if b.button("Delete", key="danger__" + ibtn("trash-2", "bulk_delete"), width="stretch"):
                dialogs.delete_dialog(db, chosen)
            c.button("Clear", key=ibtn("x", "clear_sel"), type="tertiary", on_click=_clear_selection, width="stretch")

    with st.container(key="ptable"):
        with st.container(key="prow_head"):
            head = st.columns(COLS, vertical_alignment="center", gap="small")
            st.session_state["people_all"] = all(i in selected for i in ids)
            head[0].checkbox("Select page", key="people_all", label_visibility="collapsed",
                             on_change=_select_page, args=(ids,))
            head[1].html(f'<div class="prow-grid phead">{HEADER_HTML}</div>')
        for r in chunk.itertuples():
            with st.container(key=f"prow_{r.EmployeeID}"):
                cols = st.columns(COLS, vertical_alignment="center", gap="small")
                st.session_state[f"sel_{r.EmployeeID}"] = int(r.EmployeeID) in selected
                cols[0].checkbox(f"Select {r.Name}", key=f"sel_{r.EmployeeID}",
                                 label_visibility="collapsed", on_change=_toggle, args=(int(r.EmployeeID),))
                cols[1].html(f'<div class="prow-grid">{_row_cells(r)}</div>')
                with cols[2]:
                    with st.container(horizontal=True, horizontal_alignment="right", gap="xxsmall", key=f"racts_{r.EmployeeID}"):
                        if r.Attrition == "No":
                            st.button("Open in what-if lab", key=f"icon__flask_conical__l{r.EmployeeID}", type="tertiary",
                                      help="Open in the what-if lab", on_click=open_lab, args=(int(r.EmployeeID),))
                        if st.button("Edit", key=f"icon__pencil__e{r.EmployeeID}", type="tertiary", help="Edit employee"):
                            employee = db.get(int(r.EmployeeID))
                            if employee:
                                dialogs.edit_dialog(db, employee)

    first, last = page * PAGE_SIZE + 1, page * PAGE_SIZE + len(chunk)
    _pager(page, pages, len(view), len(scored), first, last)
