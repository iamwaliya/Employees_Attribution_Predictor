"""HTML building blocks (chips, KPI cards, pill bars, gauges, sparklines).

Everything here returns an HTML string; ``render`` hands it to ``st.html``.
Strings that come from data (employee names) are escaped.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence
from urllib.parse import quote

import streamlit as st

from src.utils import DEPARTMENTS
from ui import format as fmt
from ui import theme
from ui.format import escape
from ui.icons import icon

RISK_LEVELS = ("Low", "Medium", "High")
RISK_ICON = {"Low": "circle-check", "Medium": "triangle-alert", "High": "flame"}


def render(markup: str) -> None:
    st.html(markup)


def tk() -> dict:
    """Design tokens of the active theme."""
    return theme.tokens(theme.current_mode())


def dept_color(department: str) -> str:
    try:
        return tk()["dept"][DEPARTMENTS.index(department)]
    except ValueError:
        return tk()["ink_3"]


def risk_var(level: str) -> str:
    return {"Low": "good", "Medium": "warn", "High": "crit"}[level]


# --------------------------------------------------------------------------- #
# Chips and tags
# --------------------------------------------------------------------------- #
def risk_chip(level: str, score: float | None = None) -> str:
    """'High risk | 88%' pill with a tinted background, a dot and a text label."""
    pct = f'<span class="sep"></span><span class="num">{score * 100:.0f}%</span>' if score is not None else ""
    return f'<span class="chip risk-{level.lower()}"><i class="dot"></i>{level} risk{pct}</span>'


def chip(text: str, kind: str = "neutral", icon_name: str | None = None, tip: str | None = None) -> str:
    glyph = icon(icon_name, 13) if icon_name else ""
    inner = f'{glyph}{escape(text)}'
    if tip:
        return f'<span class="chip {kind} has-tip">{inner}<span class="tip">{tip}</span></span>'
    return f'<span class="chip {kind}">{inner}</span>'


def dept_tag(department: str) -> str:
    return (f'<span class="tag"><i class="swatch" style="background:{dept_color(department)}"></i>'
            f'{escape(department)}</span>')


def status_chip(left: bool) -> str:
    return (chip("Left", "muted", "log-out") if left else chip("Active", "neutral", "circle-check"))


def avatar(name: str, department: str, size: int = 32) -> str:
    color = dept_color(department)
    return (f'<span class="avatar" style="--s:{size}px;--c:{color}">{escape(fmt.initials(name))}</span>')


def score_dots(value: int, maximum: int = 5) -> str:
    """Five small dots for a 1-5 score; filled count = score."""
    dots = "".join(f'<i class="{"on" if i < value else ""}"></i>' for i in range(maximum))
    return f'<span class="dots" title="{value} of {maximum}">{dots}</span>'


# --------------------------------------------------------------------------- #
# Tooltips (pure CSS, structured rows)
# --------------------------------------------------------------------------- #
def tip(title: str, rows: Sequence[tuple[str, str]] = (), note: str = "") -> str:
    body = "".join(f'<span class="tip-row"><span>{escape(k)}</span><b class="num">{escape(v)}</b></span>' for k, v in rows)
    extra = f'<span class="tip-note">{escape(note)}</span>' if note else ""
    return f'<span class="tip-title">{escape(title)}</span>{body}{extra}'


# --------------------------------------------------------------------------- #
# Bars, gauges, sparklines
# --------------------------------------------------------------------------- #
def pill_bar(value: float, color: str, maximum: float = 1.0, marker: float | None = None,
             tooltip: str | None = None, height: int = 8, still: bool = False) -> str:
    """Rounded progress pill. ``value`` and ``marker`` are in the same units as ``maximum``."""
    width = max(0.0, min(1.0, value / maximum if maximum else 0.0)) * 100
    mark = ""
    if marker is not None and maximum:
        mark = f'<i class="mark" style="left:{max(0.0, min(1.0, marker / maximum)) * 100:.2f}%"></i>'
    tip_html = f'<span class="tip">{tooltip}</span>' if tooltip else ""
    cls = "bar" + (" has-tip" if tooltip else "") + (" still" if still else "")
    return (f'<div class="{cls}" style="--h:{height}px"><i class="fill" style="width:{width:.2f}%;background:{color}"></i>'
            f'{mark}{tip_html}</div>')


def stacked_bar(parts: Sequence[tuple[str, float, str, str]], height: int = 10) -> str:
    """Segmented pill: parts are (label, value, css colour, tooltip html)."""
    total = sum(v for _, v, _, _ in parts) or 1.0
    segments = "".join(
        f'<i class="seg has-tip" style="flex:{max(v, 0) / total:.5f};background:{color}">'
        f'<span class="tip">{tip_html}</span></i>'
        for _, v, color, tip_html in parts if v > 0)
    return f'<div class="stack" style="--h:{height}px">{segments}</div>'


def gauge(fraction: float, color_var: str, caption: str = "probability of leaving", still: bool = True) -> str:
    """270-degree ring gauge (conic gradient ring; the number counts up on first paint)."""
    p = max(0.0, min(1.0, fraction)) * 100
    cls = "gauge" + (" still" if still else "")
    return (f'<div class="{cls}" style="--p:{p:.2f};--c:var(--{color_var});--n:{round(p)}" role="img" aria-label="{round(p)}% {escape(caption)}">'
            f'<div class="gauge-in"><span class="gauge-num"><b class="num count"></b><small>%</small></span><span>{escape(caption)}</span></div></div>')


def _smooth_path(points: list[tuple[float, float]], top: float, bottom: float) -> str:
    """Cubic-bezier path through the points (Catmull-Rom, control points clamped to the plot box)."""
    if len(points) < 2:
        return ""
    d = [f"M{points[0][0]:.2f},{points[0][1]:.2f}"]
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1, p2 = points[i], points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, min(max(p1[1] + (p2[1] - p0[1]) / 6, top), bottom))
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, min(max(p2[1] - (p3[1] - p1[1]) / 6, top), bottom))
        d.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(d)


def sparkline(values: Sequence[float], color: str, width: int = 112, height: int = 34, baseline: float | None = None,
              dot: bool = True) -> str:
    """Smooth line + gradient area as an <img> (st.html strips inline SVG, so a data URI carries it)."""
    values = [float(v) for v in values]
    if len(values) < 2:
        return ""
    pad = 4
    low, high = min(values + ([baseline] if baseline is not None else [])), max(values + ([baseline] if baseline is not None else []))
    span = (high - low) or 1.0
    xs = [pad + i * (width - 2 * pad) / (len(values) - 1) for i in range(len(values))]
    ys = [pad + (1 - (v - low) / span) * (height - 2 * pad) for v in values]
    pts = list(zip(xs, ys))
    line = _smooth_path(pts, pad, height - pad)
    area = f"{line} L{xs[-1]:.2f},{height} L{xs[0]:.2f},{height} Z"
    base = ""
    if baseline is not None:
        by = pad + (1 - (baseline - low) / span) * (height - 2 * pad)
        base = (f'<line x1="0" x2="{width}" y1="{by:.2f}" y2="{by:.2f}" stroke="{tk()["border_strong"]}" '
                f'stroke-width="1" stroke-dasharray="2 3"/>')
    end = (f'<circle cx="{xs[-1]:.2f}" cy="{ys[-1]:.2f}" r="2.6" fill="{color}" stroke="{tk()["surface"]}" stroke-width="1.5"/>'
           if dot else "")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
           f'<defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".28"/>'
           f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
           f'{base}<path d="{area}" fill="url(#g)"/><path d="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
           f'stroke-linecap="round" stroke-linejoin="round"/>{end}</svg>')
    return (f'<img class="spark" alt="" width="{width}" height="{height}" '
            f'src="data:image/svg+xml;utf8,{quote(svg, safe="")}">')


# --------------------------------------------------------------------------- #
# Cards and layout pieces
# --------------------------------------------------------------------------- #
def kpi(label: str, icon_name: str, value: str, foot: str = "", spark: str = "", tooltip: str = "") -> str:
    tip_html = f'<span class="tip">{tooltip}</span>' if tooltip else ""
    return (f'<div class="kpi card{" has-tip" if tooltip else ""}"><div class="kpi-label">{icon(icon_name, 15)}{escape(label)}</div>'
            f'<div class="kpi-row"><div class="kpi-value num">{value}</div>{spark}</div>'
            f'<div class="kpi-foot">{foot}</div>{tip_html}</div>')


def card_head(title: str, sub: str = "", right: str = "") -> str:
    sub_html = f'<p class="card-sub">{escape(sub)}</p>' if sub else ""
    return (f'<div class="card-head"><div><h3 class="card-title">{escape(title)}</h3>{sub_html}</div>'
            f'<div class="card-right">{right}</div></div>')


def card(title: str, body: str, sub: str = "", right: str = "", cls: str = "") -> str:
    return f'<section class="card {cls}">{card_head(title, sub, right)}{body}</section>'


def legend(items: Iterable[tuple[str, str]], shape: str = "dot") -> str:
    inner = "".join(f'<span class="lg"><i class="lg-{shape}" style="background:{c}"></i>{escape(t)}</span>' for t, c in items)
    return f'<div class="legend">{inner}</div>'


def notice(kind: str, title: str, text: str = "", icon_name: str | None = None) -> str:
    glyph = icon_name or {"info": "info", "warn": "triangle-alert", "good": "circle-check", "crit": "circle-alert"}[kind]
    body = f'<span>{escape(text)}</span>' if text else ""
    return f'<div class="notice {kind}">{icon(glyph, 16)}<div><b>{escape(title)}</b>{body}</div></div>'


def empty_state(title: str, text: str, icon_name: str = "users-round") -> str:
    return (f'<div class="empty card"><span class="empty-ic">{icon(icon_name, 22)}</span>'
            f'<h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def delta_chip(delta_fraction: float, lower_is_better: bool = True) -> str:
    """'-7 pts' with an arrow, tinted good when risk went down."""
    pts = delta_fraction * 100
    if round(pts) == 0:
        return '<span class="delta flat"><i class="ic i-minus"></i>no change</span>'
    improved = (pts < 0) == lower_is_better
    arrow = "arrow-down-right" if pts < 0 else "arrow-up-right"
    return (f'<span class="delta {"good" if improved else "crit"}">{icon(arrow, 13)}'
            f'<span class="num">{fmt.signed_points(delta_fraction)}</span></span>')


def page_header(title: str, subtitle: str, updated_html: str) -> str:
    return (f'<header class="pg-head"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
            f'<div class="pg-meta">{updated_html}</div></header>')
