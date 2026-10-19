"""Smoke tests for the Streamlit app (runs it headlessly against a temp database)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import streamlit as st  # noqa: E402

from src import database  # noqa: E402
from src.utils import MODEL_PATH, PROJECT_ROOT  # noqa: E402
from ui.config import PAGES  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "app_test.db")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    st.cache_resource.clear()
    st.cache_data.clear()


def test_first_run_creates_and_seeds_database(app, tmp_path):
    app.run()
    assert not app.exception
    db = database.EmployeeDB(tmp_path / "app_test.db")
    assert db.count() == 1500
    assert app.radio(key="nav").value == "Overview"


@pytest.mark.parametrize("page", [p.label for p in PAGES])
def test_every_page_renders(app, page):
    app.run()
    app.radio(key="nav").set_value(page).run()
    assert not app.exception
    assert app.radio(key="nav").value == page


@pytest.mark.parametrize("page", [p.label for p in PAGES])
def test_every_page_survives_an_empty_database(app, tmp_path, page):
    app.run()
    database.EmployeeDB(tmp_path / "app_test.db").delete_all()
    app.radio(key="nav").set_value(page).run()
    assert not app.exception


def test_dark_theme_renders(app):
    app.query_params["theme"] = "dark"
    app.run()
    assert not app.exception
    assert app.session_state["theme"] == "dark"


def test_new_employee_shows_up_in_the_people_list(app, tmp_path):
    app.run()
    db = database.EmployeeDB(tmp_path / "app_test.db")
    new_id = db.add({"Name": "Zed Testperson", "Age": 31, "Department": "Sales", "MonthlyIncome": 41000,
                     "YearsAtCompany": 1, "JobSatisfaction": 2, "WorkLifeBalance": 2, "PerformanceRating": 3,
                     "OverTime": "Yes", "DistanceFromHome": 25, "Attrition": "No"})
    app.radio(key="nav").set_value("People").run()
    assert not app.exception
    app.text_input(key="people_search").set_value("Zed Testperson").run()
    assert not app.exception
    checkbox_keys = [c.key for c in app.checkbox if c.key]
    assert f"sel_{new_id}" in checkbox_keys
    assert len([k for k in checkbox_keys if k.startswith("sel_")]) == 1


def test_lab_opens_on_the_requested_employee(app, tmp_path):
    app.run()
    app.session_state["lab_employee"] = 1005
    app.radio(key="nav").set_value("What-if lab").run()
    assert not app.exception
    assert app.session_state["lab_choice"] == 1005
