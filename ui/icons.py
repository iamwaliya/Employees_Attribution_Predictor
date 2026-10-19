"""Lucide icons rendered as CSS masks.

Streamlit's ``st.html`` strips inline ``<svg>``, so icons are shipped as CSS
custom properties (``--i-<name>: url(data:image/svg+xml,...)``) and drawn with
``mask`` so they inherit ``currentColor`` and follow the light / dark theme.

    icon("users")                 -> <i class="ic i-users"></i>
    button key ``ibtn__plus__x``  -> a Streamlit button gets a "plus" icon before its label
    button key ``icon__plus__x``  -> icon-only button (the label stays for screen readers)
"""

from __future__ import annotations

from urllib.parse import quote

from ui.icon_data import ICONS


def _mask_markup(name: str, stroke: float = 2) -> str:
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
        f"stroke-width='{stroke}' stroke-linecap='round' stroke-linejoin='round'>"
        + ICONS[name].replace('"', "'")
        + "</svg>"
    )


def icon(name: str, size: int | None = None, cls: str = "") -> str:
    """HTML for an inline icon. ``size`` is in px (defaults to the 1em of the context)."""
    if name not in ICONS:
        raise KeyError(f"Unknown icon '{name}'")
    style = f' style="--s:{size}px"' if size else ""
    return f'<i class="ic i-{name} {cls}"{style} aria-hidden="true"></i>'


def _key_token(name: str) -> str:
    """Icon name as it appears in a widget key (hyphens become underscores)."""
    return name.replace("-", "_")


def ibtn(icon_name: str, name: str, icon_only: bool = False) -> str:
    """Widget key that gives a button / popover an icon (see module docstring)."""
    if icon_name not in ICONS:
        raise KeyError(f"Unknown icon '{icon_name}'")
    return f"{'icon' if icon_only else 'ibtn'}__{_key_token(icon_name)}__{name}"


def icon_css() -> str:
    """CSS defining every icon as a custom property plus helper classes."""
    variables = [f"--i-{name}:url(\"data:image/svg+xml,{quote(_mask_markup(name), safe='')}\")" for name in ICONS]
    classes = [f".i-{name}{{--i:var(--i-{name})}}" for name in ICONS]
    # Streamlit buttons / popovers pick an icon from their key: ``ibtn__<icon>__<anything>``.
    buttons = [
        f'[class*="st-key-{prefix}__{_key_token(name)}__"] button::before{{-webkit-mask-image:var(--i-{name});mask-image:var(--i-{name})}}'
        for name in ICONS
        for prefix in ("ibtn", "icon")
    ]
    return ":root{" + ";".join(variables) + "}\n" + "\n".join(classes) + "\n" + "\n".join(buttons) + "\n"
