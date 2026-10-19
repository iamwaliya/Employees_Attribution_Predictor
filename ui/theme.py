"""Design tokens (light and dark) and the global stylesheet.

The palette follows the validated reference set: blue is the primary series,
orange the secondary, and the five department hues keep a fixed order. Risk
bands use the fixed status colours and always ship with a dot and a text label,
never colour alone.

Streamlit's own theme is static (``.streamlit/config.toml``, light). Dark mode
is a full CSS-variable override driven by ``st.session_state`` / ``?theme=``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import streamlit as st

from ui.icons import icon_css

CSS_PATH = Path(__file__).with_name("styles.css")

LIGHT = {
    "bg": "#FAFAFA", "surface": "#FFFFFF", "surface_2": "#F4F5F7", "surface_3": "#ECEEF2",
    "border": "#E4E6EB", "border_strong": "#CDD2DB", "grid": "#EDEFF3", "axis": "#CDD2DB",
    "ink": "#0F172A", "ink_2": "#475569", "ink_3": "#64748B",
    "accent": "#2A78D6", "accent_strong": "#1C5CAB", "accent_soft": "rgba(42,120,214,.10)",
    "accent_ring": "rgba(42,120,214,.28)", "on_accent": "#FFFFFF",
    "series_2": "#EB6834",
    "good": "#0CA30C", "warn": "#FAB219", "crit": "#D03B3B",
    "good_bg": "rgba(12,163,12,.11)", "warn_bg": "rgba(250,178,25,.17)", "crit_bg": "rgba(208,59,59,.10)",
    "good_fg": "#0A6B0A", "warn_fg": "#8A5A00", "crit_fg": "#B02525",
    "shadow_1": "0 1px 2px rgba(15,23,42,.04), 0 1px 1px rgba(15,23,42,.03)",
    "shadow_2": "0 1px 2px rgba(15,23,42,.05), 0 4px 12px -2px rgba(15,23,42,.07)",
    "shadow_pop": "0 12px 32px -8px rgba(15,23,42,.18), 0 2px 6px rgba(15,23,42,.06)",
    "glass": "rgba(250,250,250,.82)", "overlay": "rgba(15,23,42,.38)",
    "dept": ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4"],
}
DARK = {
    "bg": "#0F172A", "surface": "#151F35", "surface_2": "#1B2741", "surface_3": "#22304D",
    "border": "#27344F", "border_strong": "#37466A", "grid": "#212D47", "axis": "#37466A",
    "ink": "#F1F5F9", "ink_2": "#CBD5E1", "ink_3": "#94A3B8",
    "accent": "#3987E5", "accent_strong": "#6DA7EC", "accent_soft": "rgba(57,135,229,.16)",
    "accent_ring": "rgba(57,135,229,.40)", "on_accent": "#FFFFFF",
    "series_2": "#D95926",
    "good": "#0CA30C", "warn": "#FAB219", "crit": "#D03B3B",
    "good_bg": "rgba(12,163,12,.18)", "warn_bg": "rgba(250,178,25,.16)", "crit_bg": "rgba(208,59,59,.22)",
    "good_fg": "#5BD65B", "warn_fg": "#F7C24D", "crit_fg": "#FF8B8B",
    "shadow_1": "0 1px 2px rgba(2,6,23,.40)",
    "shadow_2": "0 1px 2px rgba(2,6,23,.45), 0 6px 18px -4px rgba(2,6,23,.45)",
    "shadow_pop": "0 16px 40px -8px rgba(2,6,23,.65), 0 2px 8px rgba(2,6,23,.4)",
    "glass": "rgba(15,23,42,.82)", "overlay": "rgba(2,6,23,.62)",
    "dept": ["#3987E5", "#D95926", "#199E70", "#C98500", "#D55181"],
}


def tokens(mode: str) -> dict:
    return DARK if mode == "dark" else LIGHT


def _variables(mode: str) -> str:
    t = tokens(mode)
    names = {k: v for k, v in t.items() if k != "dept"}
    body = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in names.items())
    dept = ";".join(f"--dept-{i}:{c}" for i, c in enumerate(t["dept"]))
    return f":root{{color-scheme:{mode};{body};{dept}}}"


def _static_css() -> str:
    # Cached unless ATTRITION_UI_DEV=1 (then edits to styles.css show on the next rerun).
    return _build_css() if os.environ.get("ATTRITION_UI_DEV") else _cached_css()


@lru_cache(maxsize=1)
def _cached_css() -> str:
    return _build_css()


def _build_css() -> str:
    css = icon_css() + CSS_PATH.read_text(encoding="utf-8")
    # st.html runs the markup through a sanitiser; a stray "<" (even inside a CSS
    # comment) makes it drop the whole <style> block, so refuse to ship one.
    if "<" in css:
        raise ValueError("styles.css must not contain '<'")
    return css


def nav_css(pages) -> str:
    """Icon for each navigation option (radio buttons in the app bar), by position."""
    rules = []
    for i, page in enumerate(pages, start=1):
        rules.append(f'.st-key-nav [data-testid="stRadioGroup"] > div:nth-child({i}) label::before'
                     f'{{-webkit-mask-image:var(--i-{page.icon});mask-image:var(--i-{page.icon})}}')
    return "\n".join(rules)


def current_mode() -> str:
    return st.session_state.get("theme", "light")


def inject(mode: str, pages=()) -> None:
    """Write fonts, tokens and the stylesheet into the page (call once per run, first)."""
    st.html(f"<style data-app-css>{_variables(mode)}\n{_static_css()}\n{nav_css(pages)}</style>")
