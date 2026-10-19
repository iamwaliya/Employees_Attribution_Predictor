"""App chrome: sticky top bar (brand, workspace switcher, navigation, status, theme, profile),
page header with the last-updated stamp, and the toast queue."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from ui import components as ui
from ui import format as fmt
from ui import state, theme
from ui.config import BRAND_NAME, PAGE_BY_LABEL, PAGE_BY_SLUG, PAGES, USER, WORKSPACES, Page
from ui.icons import ibtn, icon


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Seed theme, page and workspace from the URL on first load (so links and refreshes keep them)."""
    params = st.query_params
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark" if params.get("theme") == "dark" else "light"
    if "nav" not in st.session_state:
        st.session_state["nav"] = PAGE_BY_SLUG.get(params.get("page", ""), PAGES[0]).label
    st.session_state.setdefault("workspace", WORKSPACES[0]["name"])


def _sync_url() -> None:
    st.query_params["page"] = PAGE_BY_LABEL[st.session_state["nav"]].slug
    st.query_params["theme"] = st.session_state["theme"]


def _toggle_theme() -> None:
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
    _sync_url()


def _set_theme(mode: str | None) -> None:
    if mode in ("light", "dark") and mode != st.session_state["theme"]:
        st.session_state["theme"] = mode
        _sync_url()


def _pick_workspace(name: str) -> None:
    if name != st.session_state["workspace"]:
        st.session_state["workspace"] = name
        meta = next(w for w in WORKSPACES if w["name"] == name)
        state.flash(f"Switched to {name}." + (" Preview workspace: it shows the same dataset." if meta["demo"] else ""), "info")


def _refresh() -> None:
    state.refresh_data()
    state.flash("Data refreshed.")


# --------------------------------------------------------------------------- #
# Top bar
# --------------------------------------------------------------------------- #
def _status_pill(model_name: str, trained: str, employees: int) -> str:
    rows = [("Model", model_name), ("Trained", trained), ("Employees", fmt.number(employees)), ("Database", "SQLite")]
    return (f'<div class="status has-tip"><i class="pulse"></i>System active'
            f'<span class="tip tip-down">{ui.tip("All systems running", rows)}</span></div>')


def render_appbar(model_name: str, trained: str, employees: int) -> Page:
    mode = st.session_state["theme"]
    workspace = st.session_state["workspace"]
    with st.container(key="appbar"):
        left, middle, right = st.columns([3.2, 5.0, 3.5], vertical_alignment="center", gap="small")

        with left:
            with st.container(horizontal=True, vertical_alignment="center", gap="small", key="brandrow"):
                st.html(f'<div class="brand"><span class="brand-mark">{icon("radar", 17)}</span>'
                        f'<span class="brand-name">{BRAND_NAME}</span></div>')
                st.html('<span class="crumb-sep"></span>')
                with st.popover(workspace, key=ibtn("building-2", "workspace")):
                    st.html('<div class="menu-head">Workspaces</div>')
                    for index, ws in enumerate(WORKSPACES):
                        current = ws["name"] == workspace
                        st.button(ws["name"], key=ibtn("check" if current else "building-2", f"ws{index}"),
                                  type="tertiary", width="stretch", disabled=current,
                                  help=ws["meta"] + (" (preview)" if ws["demo"] else ""),
                                  on_click=_pick_workspace, args=(ws["name"],))
                    st.html('<div class="menu-foot">Preview workspaces use the same dataset.</div>')

        with middle:
            st.radio("Navigate", [p.label for p in PAGES], key="nav", horizontal=True,
                     label_visibility="collapsed", on_change=_sync_url)

        with right:
            with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small",
                              key="rightrow"):
                st.html(_status_pill(model_name, trained, employees))
                st.button("Switch to dark theme" if mode == "light" else "Switch to light theme",
                          key=ibtn("moon" if mode == "light" else "sun", "theme", icon_only=True),
                          on_click=_toggle_theme, type="tertiary",
                          help="Dark theme" if mode == "light" else "Light theme")
                with st.popover(USER["name"], key=ibtn("user-round", "profile")):
                    st.html(f'<div class="profile"><span class="avatar" style="--s:36px;--c:var(--accent)">'
                            f'{fmt.initials(USER["name"])}</span><div><b>{USER["name"]}</b>'
                            f'<span>{USER["role"]}</span></div></div>')
                    st.html('<div class="menu-head">Appearance</div>')
                    st.session_state["theme_pick"] = mode  # keep the control in step with the toolbar button
                    st.segmented_control("Appearance", ["light", "dark"], key="theme_pick",
                                         format_func=str.capitalize, label_visibility="collapsed",
                                         on_change=lambda: _set_theme(st.session_state.get("theme_pick")))
                    st.button("Refresh data", key=ibtn("refresh-cw", "refresh"), type="tertiary", width="stretch",
                              on_click=_refresh)
    return PAGE_BY_LABEL[st.session_state["nav"]]


# --------------------------------------------------------------------------- #
# Page header, toasts
# --------------------------------------------------------------------------- #
def updated_stamp() -> str:
    moment = state.last_updated()
    if moment is None:
        return ""
    now = datetime.now()
    return (f'<span class="updated has-tip">{icon("clock", 14)}Last updated '
            f'<b class="num">{fmt.relative_time(moment, now)}</b>'
            f'<span class="tip tip-down">{ui.tip("Employee data", [("Saved", fmt.absolute_time(moment))], "From the database file")}</span></span>')


def render_header(page: Page) -> None:
    ui.render(ui.page_header(page.title, page.subtitle, updated_stamp()))


def show_flash() -> None:
    if "flash" in st.session_state:
        kind, message = st.session_state.pop("flash")
        st.toast(message, icon=":material/check:" if kind == "success" else ":material/info:")
