"""Themed Altair charts: layered gradient area charts with hover tooltips.

Legends are drawn as HTML next to the chart (see ``components.legend``), so the
charts themselves have none. Colours come from the active theme tokens.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ui import theme

FONT = "Inter, system-ui, sans-serif"


def _t() -> dict:
    return theme.tokens(theme.current_mode())


def _gradient(color: str, top_opacity: float = 0.30) -> alt.Gradient:
    return alt.Gradient(
        gradient="linear", x1=1, x2=1, y1=0, y2=1,
        stops=[alt.GradientStop(color=_rgba(color, top_opacity), offset=0),
               alt.GradientStop(color=_rgba(color, 0.0), offset=1)])


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _style(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    t = _t()
    return (chart.properties(height=height, padding={"left": 2, "right": 8, "top": 8, "bottom": 2})
            .configure(background="rgba(0,0,0,0)", font=FONT)
            .configure_view(stroke=None)
            .configure_axis(labelFont=FONT, labelColor=t["ink_3"], labelFontSize=11.5, labelPadding=6,
                            titleFont=FONT, titleColor=t["ink_3"], titleFontSize=11.5, titleFontWeight=500, titlePadding=10,
                            domainColor=t["axis"], domainWidth=1, tickColor=t["axis"], tickSize=4, tickWidth=1,
                            gridColor=t["grid"], gridWidth=1)
            .configure_axisY(domain=False, ticks=False, gridDash=[2, 4])
            .configure_axisX(grid=False, tickSize=4)
            .configure_legend(disable=True))


def show(chart: alt.TopLevelMixin, height: int = 230) -> None:
    st.altair_chart(_style(chart, height), width="stretch", theme=None)


# --------------------------------------------------------------------------- #
def risk_distribution(risk_scores: pd.Series) -> alt.TopLevelMixin:
    """How the predicted risk of active employees is spread, with the Low / Medium / High bands."""
    t = _t()
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(risk_scores.clip(0, 1 - 1e-9), bins=edges)
    total = max(int(counts.sum()), 1)
    data = pd.DataFrame({
        "lo": edges[:-1], "hi": edges[1:], "mid": (edges[:-1] + edges[1:]) / 2, "Employees": counts,
        "Range": [f"{a * 100:.0f}-{b * 100:.0f}%" for a, b in zip(edges[:-1], edges[1:])],
        "Share": [f"{c / total * 100:.1f}% of active staff" for c in counts],
    })
    bands = pd.DataFrame({
        "start": [0, 0.30, 0.60], "end": [0.30, 0.60, 1.0], "Band": ["Low", "Medium", "High"],
        "color": [t["good"], t["warn"], t["crit"]]})
    x = alt.X("mid:Q", scale=alt.Scale(domain=[0, 1], nice=False), axis=alt.Axis(format="%", values=[0, .2, .4, .6, .8, 1], title=None))
    y = alt.Y("Employees:Q", axis=alt.Axis(title=None, tickCount=4, format="d"))

    band_rects = (alt.Chart(bands).mark_rect(opacity=0.07)
                  .encode(x=alt.X("start:Q", scale=alt.Scale(domain=[0, 1])), x2="end:Q", color=alt.Color("color:N", scale=None)))
    band_text = (alt.Chart(bands).mark_text(align="left", dx=8, dy=10, baseline="top", fontSize=11.5, fontWeight=600, color=t["ink_3"])
                 .encode(x="start:Q", y=alt.value(0), text="Band:N"))
    area = alt.Chart(data).mark_area(interpolate="monotone", line={"color": t["accent"], "strokeWidth": 2}, color=_gradient(t["accent"])).encode(x=x, y=y)
    hover = alt.selection_point(name="hover", fields=["mid"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    points = (alt.Chart(data).mark_point(opacity=0, size=400).encode(x=x, y=y).add_params(hover))
    rule = (alt.Chart(data).mark_rule(color=t["border_strong"], strokeWidth=1)
            .encode(x=x, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                    tooltip=[alt.Tooltip("Range:N", title="Risk score"), alt.Tooltip("Employees:Q", title="Employees"),
                             alt.Tooltip("Share:N", title="Share")]))
    dot = (alt.Chart(data).mark_circle(size=80, color=t["accent"], stroke=t["surface"], strokeWidth=2)
           .encode(x=x, y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0))))
    return alt.layer(band_rects, band_text, area, rule, points, dot)


def attrition_curve(data: pd.DataFrame, average: float, label: str, order: list[str]) -> alt.TopLevelMixin:
    """Attrition rate across ordered bands (tenure, satisfaction): smooth gradient area, company average as a dashed rule.

    ``data`` needs columns Band, Rate, Employees, Left.
    """
    t = _t()
    data = data.copy()
    data["Versus"] = [("+" if r - average >= 0 else "-") + f"{abs(r - average) * 100:.1f} pts vs company" for r in data["Rate"]]
    top = float(max(data["Rate"].max(), average) * 1.25) or 0.1
    x = alt.X("Band:N", sort=order, scale=alt.Scale(padding=0.4), axis=alt.Axis(title=None, labelAngle=0))
    y = alt.Y("Rate:Q", scale=alt.Scale(domain=[0, top], nice=True), axis=alt.Axis(title=None, format=".0%", tickCount=4))
    area = alt.Chart(data).mark_area(interpolate="monotone", line={"color": t["accent"], "strokeWidth": 2}, color=_gradient(t["accent"])).encode(x=x, y=y)
    hover = alt.selection_point(name="hover", fields=["Band"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    points = (alt.Chart(data).mark_circle(size=70, color=t["accent"], stroke=t["surface"], strokeWidth=2)
              .encode(x=x, y=y, opacity=alt.value(1))
              .add_params(hover))
    halo = (alt.Chart(data).mark_circle(size=380, color=t["accent"], opacity=0)
            .encode(x=x, y=y, opacity=alt.condition(hover, alt.value(0.16), alt.value(0)),
                    tooltip=[alt.Tooltip("Band:N", title=label), alt.Tooltip("Rate:Q", title="Attrition rate", format=".1%"),
                             alt.Tooltip("Left:Q", title="Left"), alt.Tooltip("Employees:Q", title="Employees"),
                             alt.Tooltip("Versus:N", title="Compared with")]))
    avg_df = pd.DataFrame({"Rate": [average], "Label": [f"Company {average * 100:.1f}%"]})
    avg_rule = alt.Chart(avg_df).mark_rule(color=t["ink_3"], strokeDash=[4, 4], strokeWidth=1).encode(y="Rate:Q")
    avg_text = (alt.Chart(avg_df).mark_text(align="right", baseline="bottom", dy=-4, dx=-2, fontSize=11, color=t["ink_3"], fontWeight=500)
                .encode(y="Rate:Q", text="Label:N", x=alt.value(1000)))
    return alt.layer(area, avg_rule, halo, points)
