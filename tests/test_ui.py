"""Unit tests for the front-end helpers: formatting, icons, theme tokens, filters and interventions."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ui import components as ui
from ui import format as fmt
from ui import icons, theme
from ui.icon_data import ICONS
from ui.config import PAGES
from ui.pages.people import QUICK_FILTERS, SORTS, apply_filters


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value, expected", [
    (0, "0"), (999, "999"), (1500, "1,500"), (59100, "59,100"), (108700, "1,08,700"), (12345678, "1,23,45,678"),
])
def test_number_uses_indian_grouping(value, expected):
    assert fmt.number(value) == expected


def test_number_handles_negatives_decimals_and_missing():
    assert fmt.number(-8050.5, 1) == "-8,050.5"
    assert fmt.number(None) == "-"
    assert fmt.number(float("nan")) == "-"


def test_inr_full_and_compact():
    assert fmt.inr(71700) == "₹71,700"
    assert fmt.inr(108700) == "₹1,08,700"
    assert fmt.inr(59100, compact=True) == "₹59.1k"
    assert fmt.inr(108700, compact=True) == "₹1.09L"
    assert fmt.inr(25_000_000, compact=True) == "₹2.5Cr"
    assert fmt.inr(950, compact=True) == "₹950"
    assert fmt.inr(None) == "-"


def test_percent_and_point_changes():
    assert fmt.pct(0.1593) == "15.9%"
    assert fmt.pct(None) == "-"
    assert fmt.signed_points(-0.142) == "-14 pts"
    assert fmt.signed_points(0.083) == "+8 pts"
    assert fmt.signed_points(0.004) == "no change"


def test_initials_plural_and_escape():
    assert fmt.initials("Aarav Sharma") == "AS"
    assert fmt.initials("  madonna ") == "MA"
    assert fmt.initials("") == "?"
    assert fmt.plural(1, "employee") == "1 employee"
    assert fmt.plural(1200, "employee") == "1,200 employees"
    assert fmt.escape('<b onclick="x">&') == "&lt;b onclick=&quot;x&quot;&gt;&amp;"


def test_relative_time():
    now = datetime(2026, 10, 19, 12, 0)
    assert fmt.relative_time(now - timedelta(seconds=10), now) == "just now"
    assert fmt.relative_time(now - timedelta(minutes=5), now) == "5 min ago"
    assert fmt.relative_time(now - timedelta(hours=3), now) == "3 h ago"
    assert fmt.relative_time(now - timedelta(days=2), now) == "2 d ago"


# --------------------------------------------------------------------------- #
# Icons and theme
# --------------------------------------------------------------------------- #
def test_every_page_icon_exists():
    for page in PAGES:
        assert page.icon in ICONS


def test_icon_helpers():
    assert 'class="ic i-users' in icons.icon("users")
    with pytest.raises(KeyError):
        icons.icon("does-not-exist")
    assert icons.ibtn("trash-2", "bulk") == "ibtn__trash_2__bulk"
    assert icons.ibtn("pencil", "e1", icon_only=True) == "icon__pencil__e1"


def test_icon_css_defines_every_icon_and_has_no_markup_characters():
    css = icons.icon_css()
    for name in ICONS:
        assert f"--i-{name}:" in css
    assert "<" not in css


def test_stylesheet_is_safe_for_st_html():
    # st.html drops a whole <style> block if it contains "<".
    assert "<" not in theme._build_css()


def test_design_tokens():
    assert theme.LIGHT["bg"] == "#FAFAFA"
    assert theme.DARK["bg"] == "#0F172A"
    assert set(theme.LIGHT) == set(theme.DARK)
    assert len(theme.LIGHT["dept"]) == len(theme.DARK["dept"]) == 5
    assert "--bg:#0F172A" in theme._variables("dark")


def _rgb(color: str) -> np.ndarray:
    if color.startswith("#"):
        h = color.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)
    r, g, b, _ = [float(v) for v in color[color.index("(") + 1:-1].split(",")]
    return np.array([r, g, b])


def _blend(tint: str, surface: str) -> np.ndarray:
    values = [float(v) for v in tint[tint.index("(") + 1:-1].split(",")]
    alpha = values[3]
    return np.array(values[:3]) * alpha + _rgb(surface) * (1 - alpha)


def _luminance(rgb: np.ndarray) -> float:
    c = rgb / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _contrast(a: np.ndarray, b: np.ndarray) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_risk_chip_text_is_readable(mode):
    t = theme.tokens(mode)
    for kind in ("good", "warn", "crit"):
        background = _blend(t[f"{kind}_bg"], t["surface"])
        assert _contrast(_rgb(t[f"{kind}_fg"]), background) >= 4.5, f"{kind} chip in {mode}"
    assert _contrast(_rgb(t["ink"]), _rgb(t["bg"])) >= 7
    assert _contrast(_rgb(t["ink_3"]), _rgb(t["surface"])) >= 4.5
    assert _contrast(_rgb(t["accent_strong"]), _blend(t["accent_soft"], t["surface"])) >= 4.5


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def test_risk_chip_shows_label_and_score():
    html = ui.risk_chip("High", 0.884)
    assert "risk-high" in html and "High risk" in html and "88%" in html
    assert "Low risk" in ui.risk_chip("Low")


def test_user_text_is_escaped_in_components():
    assert "<script>" not in ui.avatar("<script>alert(1)</script>", "Sales")
    assert "<script>" not in ui.chip("<script>")
    assert "<script>" not in ui.tip("<script>", [("k", "<v>")])


def test_pill_bar_is_clamped():
    assert "width:100.00%" in ui.pill_bar(5, "red", 1.0)
    assert "width:0.00%" in ui.pill_bar(-1, "red", 1.0)
    assert "width:50.00%" in ui.pill_bar(0.5, "red", 1.0)


def test_sparkline_and_gauge():
    assert ui.sparkline([1, 2, 3], "#2a78d6").startswith("<img")
    assert ui.sparkline([1], "#2a78d6") == ""
    assert 'aria-label="62% probability of leaving"' in ui.gauge(0.62, "warn")


# --------------------------------------------------------------------------- #
# People filters
# --------------------------------------------------------------------------- #
@pytest.fixture()
def workforce():
    return pd.DataFrame({
        "EmployeeID": [1, 2, 3, 4, 5],
        "Name": ["Asha Rao", "Bala Nair", "Chitra Iyer", "Dev Shah", "Esha Gupta"],
        "Department": ["Engineering", "Sales", "Sales", "HR", "Engineering"],
        "OverTime": ["Yes", "Yes", "No", "No", "No"],
        "JobSatisfaction": [1, 4, 2, 5, 3],
        "YearsAtCompany": [0.5, 3, 8, 0, 2],
        "MonthlyIncome": [30000, 50000, 70000, 40000, 90000],
        "Attrition": ["No", "No", "No", "Yes", "No"],
        "RiskScore": [0.9, 0.5, 0.35, 0.95, 0.1],
        "RiskLevel": ["High", "Medium", "Medium", "High", "Low"],
    })


def test_search_matches_name_id_and_department(workforce):
    assert list(apply_filters(workforce, search="nair")["EmployeeID"]) == [2]
    assert list(apply_filters(workforce, search="3")["EmployeeID"]) == [3]
    assert set(apply_filters(workforce, search="sales")["EmployeeID"]) == {2, 3}


def test_status_filter(workforce):
    assert list(apply_filters(workforce, status="Left")["EmployeeID"]) == [4]
    assert 4 not in set(apply_filters(workforce, status="Active")["EmployeeID"])


def test_attribute_pills_narrow_and_team_pills_widen(workforce):
    assert set(apply_filters(workforce, quick=["High overtime"])["EmployeeID"]) == {1, 2}
    assert set(apply_filters(workforce, quick=["High overtime", "Low satisfaction"])["EmployeeID"]) == {1}
    assert set(apply_filters(workforce, quick=["Engineering team", "Sales team"])["EmployeeID"]) == {1, 2, 3, 5}
    assert set(apply_filters(workforce, quick=["Engineering team", "Low satisfaction"])["EmployeeID"]) == {1}
    assert set(apply_filters(workforce, quick=["New joiners"])["EmployeeID"]) == {1, 4}


def test_high_risk_pill_ignores_people_who_left(workforce):
    assert set(apply_filters(workforce, quick=["High risk"])["EmployeeID"]) == {1}


def test_risk_sort_puts_leavers_last(workforce):
    order = list(apply_filters(workforce, sort="Highest risk")["EmployeeID"])
    assert order == [1, 2, 3, 5, 4]
    assert list(apply_filters(workforce, sort="Highest income")["EmployeeID"])[0] == 5
    assert list(apply_filters(workforce, sort="Name A to Z")["Name"])[0] == "Asha Rao"


def test_all_declared_filters_and_sorts_work(workforce):
    for label in QUICK_FILTERS:
        apply_filters(workforce, quick=[label])
    for label in SORTS:
        assert len(apply_filters(workforce, sort=label)) == len(workforce)
