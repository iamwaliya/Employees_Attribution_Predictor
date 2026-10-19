"""Modal dialogs: add, edit and delete employees, import a CSV, wipe or reset the database."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database import EmployeeDB, seed_database
from src.predict import InputValidationError
from src.utils import DEPARTMENTS
from ui import components as ui
from ui import format as fmt
from ui import state
from ui.icons import ibtn

NEW_EMPLOYEE = {"Name": "", "Age": 30, "Department": "Engineering", "MonthlyIncome": 60000.0,
                "YearsAtCompany": 2.0, "JobSatisfaction": 3, "WorkLifeBalance": 3,
                "PerformanceRating": 3, "OverTime": "No", "DistanceFromHome": 10.0, "Attrition": "No"}
SCORE_HELP = {"JobSatisfaction": "1 = very low, 5 = very high",
              "WorkLifeBalance": "1 = poor, 5 = excellent",
              "PerformanceRating": "1 = poor, 5 = outstanding"}


def show_errors(exc: InputValidationError) -> None:
    lines = "".join(f"<li><b>{ui.escape(field)}</b> {ui.escape(message)}</li>" for field, message in exc.errors.items())
    ui.render(f'<div class="notice crit"><i class="ic i-circle-alert" style="--s:16px"></i><div><b>Check these fields</b>'
              f'<ul class="err-list">{lines}</ul></div></div>')


def employee_fields(defaults: dict, key: str) -> dict:
    """Render the employee form widgets and return the raw values."""
    left, right = st.columns(2)
    name = left.text_input("Full name", defaults["Name"], key=f"{key}_name", max_chars=80, placeholder="e.g. Aarav Sharma")
    department = right.selectbox("Department", DEPARTMENTS, DEPARTMENTS.index(defaults["Department"]), key=f"{key}_dept")
    age = left.number_input("Age", 0, 120, int(defaults["Age"]), 1, key=f"{key}_age")
    income = right.number_input("Monthly income (₹)", 0.0, 10_000_000.0, float(defaults["MonthlyIncome"]), 1000.0,
                                key=f"{key}_income", format="%.0f")
    years = left.number_input("Years at company", 0.0, 80.0, float(defaults["YearsAtCompany"]), 0.5, key=f"{key}_years")
    distance = right.number_input("Distance from home (km)", 0.0, 1000.0, float(defaults["DistanceFromHome"]), 1.0,
                                  key=f"{key}_dist")
    c1, c2, c3 = st.columns(3)
    satisfaction = c1.slider("Job satisfaction", 1, 5, int(defaults["JobSatisfaction"]), help=SCORE_HELP["JobSatisfaction"],
                             key=f"{key}_sat")
    balance = c2.slider("Work-life balance", 1, 5, int(defaults["WorkLifeBalance"]), help=SCORE_HELP["WorkLifeBalance"],
                        key=f"{key}_wlb")
    performance = c3.slider("Performance rating", 1, 5, int(defaults["PerformanceRating"]), help=SCORE_HELP["PerformanceRating"],
                            key=f"{key}_perf")
    o1, o2 = st.columns(2)
    overtime = o1.segmented_control("Works overtime", ["No", "Yes"], default=defaults["OverTime"], key=f"{key}_ot") or defaults["OverTime"]
    status = o2.segmented_control("Status", ["Active", "Left"], default="Left" if defaults["Attrition"] == "Yes" else "Active",
                                  key=f"{key}_status",
                                  help="'Left' means this person has left the company. It is the label used when retraining.") or "Active"
    return {"Name": name, "Age": age, "Department": department, "MonthlyIncome": income, "YearsAtCompany": years,
            "JobSatisfaction": satisfaction, "WorkLifeBalance": balance, "PerformanceRating": performance,
            "OverTime": overtime, "DistanceFromHome": distance, "Attrition": "Yes" if status == "Left" else "No"}


def _saved(message: str) -> None:
    state.flash(message)
    state.refresh_data()
    st.session_state.pop("people_sel", None)
    st.rerun()


@st.dialog("Add employee", width="large")
def add_dialog(db: EmployeeDB) -> None:
    with st.form("add_form", border=False):
        raw = employee_fields(NEW_EMPLOYEE, "add")
        if st.form_submit_button("Save employee", type="primary", width="stretch"):
            try:
                new_id = db.add(raw)
            except InputValidationError as exc:
                show_errors(exc)
                return
            _saved(f"Added {raw['Name'].strip()} (ID {new_id}).")


@st.dialog("Edit employee", width="large")
def edit_dialog(db: EmployeeDB, employee: dict) -> None:
    st.caption(f"Employee ID {employee['EmployeeID']}")
    with st.form("edit_form", border=False):
        raw = employee_fields(employee, f"edit{employee['EmployeeID']}")
        if st.form_submit_button("Save changes", type="primary", width="stretch"):
            try:
                db.update(employee["EmployeeID"], raw)
            except InputValidationError as exc:
                show_errors(exc)
                return
            except KeyError:
                st.error("This employee no longer exists.")
                return
            _saved(f"Updated {raw['Name'].strip()}.")


@st.dialog("Delete employees")
def delete_dialog(db: EmployeeDB, selected: pd.DataFrame) -> None:
    count = len(selected)
    ui.render(ui.notice("warn", f"Delete {fmt.plural(count, 'employee')}?", "This permanently removes the records. It cannot be undone."))
    names = "".join(f'<li>{ui.escape(r.Name)}<span>{ui.escape(r.Department)}</span></li>' for r in selected.head(8).itertuples())
    more = f'<li class="more">and {fmt.number(count - 8)} more</li>' if count > 8 else ""
    ui.render(f'<ul class="del-list">{names}{more}</ul>')
    left, right = st.columns(2)
    left.button("Cancel", key="del_cancel", width="stretch", on_click=lambda: None)
    if right.button("Delete", key="danger__del_confirm", type="primary", width="stretch"):
        removed = db.delete(selected["EmployeeID"].tolist())
        _saved(f"Deleted {fmt.plural(removed, 'employee')}.")


@st.dialog("Import employees from CSV", width="large")
def import_dialog(db: EmployeeDB) -> None:
    ui.render(ui.notice("info", "Required columns",
                        "Age, Department, MonthlyIncome, YearsAtCompany, JobSatisfaction, WorkLifeBalance, "
                        "PerformanceRating, OverTime, DistanceFromHome. Optional: Name, Attrition (Yes = has left). "
                        "Invalid rows are skipped and listed."))
    upload = st.file_uploader("CSV file", type="csv", label_visibility="collapsed", key="import_file")
    if upload and st.button("Import employees", key=ibtn("upload", "import_go"), type="primary", width="stretch"):
        try:
            frame = pd.read_csv(upload)
            inserted, errors = db.import_dataframe(frame)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            st.error("That file could not be read as a CSV.")
        except InputValidationError as exc:
            show_errors(exc)
        else:
            if inserted:
                _saved(f"Imported {fmt.plural(inserted, 'employee')}" + (f"; skipped {len(errors)} invalid row(s)." if errors else "."))
            st.error("No valid rows found.")
            for line in errors[:15]:
                st.write(f"- {line}")


@st.dialog("Delete every employee")
def wipe_dialog(db: EmployeeDB) -> None:
    ui.render(ui.notice("crit", "This empties the database", "Every employee record is removed. Export a CSV first if you want a copy."))
    confirm = st.checkbox("I understand this deletes every employee.", key="wipe_confirm")
    if st.button("Delete all employees", key="danger__wipe_go", type="primary", disabled=not confirm, width="stretch"):
        removed = db.delete_all()
        _saved(f"Deleted all {fmt.number(removed)} employees. The database is empty.")


@st.dialog("Reset to sample data")
def reset_dialog(db: EmployeeDB) -> None:
    ui.render(ui.notice("warn", "Replace everything with 1,500 sample employees",
                        "Your current records are removed first."))
    confirm = st.checkbox("I understand my current employees will be replaced.", key="reset_confirm")
    if st.button("Reset to sample data", key="danger__reset_go", type="primary", disabled=not confirm, width="stretch"):
        seed_database(db)
        _saved("Restored the sample workforce.")
