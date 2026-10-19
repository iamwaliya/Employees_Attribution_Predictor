"""Number, currency and time formatting.

Amounts use the Indian digit grouping (1,23,456) to match the rupee figures in
the data. Everything returns plain strings; tables apply ``font-variant-numeric:
tabular-nums`` through the ``.num`` CSS class so digits line up in columns.
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

RUPEE = "₹"


def _group_indian(integer_digits: str) -> str:
    """'1234567' -> '12,34,567'."""
    if len(integer_digits) <= 3:
        return integer_digits
    head, tail = integer_digits[:-3], integer_digits[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return f"{head},{tail}"


def number(value: float | int | None, decimals: int = 0) -> str:
    """Indian-grouped number: 1500 -> '1,500', 1234567 -> '12,34,567'."""
    if value is None or pd.isna(value):
        return "-"
    text = f"{abs(float(value)):.{decimals}f}"
    integer, _, fraction = text.partition(".")
    grouped = _group_indian(integer) + (f".{fraction}" if fraction else "")
    return f"-{grouped}" if float(value) < 0 and float(grouped.replace(",", "") or 0) != 0 else grouped


def inr(value: float | int | None, decimals: int = 0, compact: bool = False) -> str:
    """Rupee amount. ``compact=True`` gives '₹59.1k', '₹1.09L' or '₹2.4Cr'."""
    if value is None or pd.isna(value):
        return "-"
    amount = float(value)
    if compact:
        magnitude = abs(amount)
        for threshold, suffix in ((1e7, "Cr"), (1e5, "L"), (1e3, "k")):
            if magnitude >= threshold:
                scaled = amount / threshold
                text = f"{scaled:.2f}".rstrip("0").rstrip(".") if suffix == "L" else f"{scaled:.1f}".rstrip("0").rstrip(".")
                return f"{RUPEE}{text}{suffix}"
    return f"{RUPEE}{number(amount, decimals)}"


def pct(fraction: float | None, decimals: int = 1) -> str:
    """0.159 -> '15.9%'."""
    if fraction is None or pd.isna(fraction):
        return "-"
    return f"{float(fraction) * 100:.{decimals}f}%"


def signed_points(delta_fraction: float, decimals: int = 0) -> str:
    """Signed percentage-point change: 0.083 -> '+8 pts', -0.14 -> '-14 pts'."""
    points = delta_fraction * 100
    if round(points, decimals) == 0:
        return "no change"
    sign = "+" if points > 0 else "-"
    return f"{sign}{abs(points):.{decimals}f} pts"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    return f"{number(count)} {singular if count == 1 else (plural_form or singular + 's')}"


def initials(name: str) -> str:
    parts = [part for part in re.split(r"\s+", str(name).strip()) if part]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def relative_time(moment: datetime, now: datetime | None = None) -> str:
    """'just now', '5 min ago', '3 h ago', '2 d ago'."""
    seconds = max(0, int(((now or datetime.now()) - moment).total_seconds()))
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        return f"{max(1, seconds // 60)} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def absolute_time(moment: datetime) -> str:
    return moment.strftime("%d %b %Y, %H:%M")


def escape(text: object) -> str:
    """Minimal HTML escaping for user-provided strings (employee names)."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))
