"""Helpers shared by several pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database import seed_database
from ui import components as ui
from ui import state
from ui.config import PAGE_BY_SLUG
from ui.icons import ibtn


def go_to(slug: str) -> None:
    """Callback: switch the top-bar navigation to another page."""
    st.session_state["nav"] = PAGE_BY_SLUG[slug].label
    st.query_params["page"] = slug


def empty_workforce(message: str) -> None:
    """Friendly empty state with the two ways to get employees in."""
    ui.render(ui.empty_state("No employees yet", message))
    left, middle, right = st.columns([1, 1, 4], gap="small")
    left.button("Add employees", key=ibtn("users", "empty_people"), type="primary", on_click=go_to, args=("people",))
    if middle.button("Load sample data", key=ibtn("database", "empty_seed")):
        seed_database(state.get_db())
        state.refresh_data()
        state.flash("Loaded 1,500 sample employees.")
        st.rerun()


def rate_table(frame: pd.DataFrame, by: str | pd.Series) -> pd.DataFrame:
    """Employees, leavers and attrition rate per group."""
    left = (frame["Attrition"] == "Yes").astype(int)
    keys = frame[by] if isinstance(by, str) else by
    grouped = left.groupby(keys, observed=False).agg(Employees="count", Left="sum")
    grouped["Rate"] = (grouped["Left"] / grouped["Employees"]).where(grouped["Employees"] > 0, 0.0)
    return grouped.reset_index()


def open_lab(employee_id: int) -> None:
    """Callback: open the what-if lab on this employee."""
    st.session_state["lab_employee"] = employee_id
    go_to("lab")
