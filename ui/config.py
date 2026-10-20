"""Brand, workspace and navigation constants. Edit these to rebrand the app."""

from __future__ import annotations

from dataclasses import dataclass

BRAND_NAME = "Attrition Predictor"
BRAND_TAGLINE = "People analytics"

# The workspace switcher and profile menu are presentation-only (the app has one
# shared database). Change the strings below to match your own demo.
WORKSPACES = [
    {"name": "Pune HQ", "meta": "All departments", "demo": False},
    {"name": "Mumbai Operations", "meta": "Sales and Marketing", "demo": True},
    {"name": "Bengaluru Tech Hub", "meta": "Engineering", "demo": True},
]
USER = {"name": "Waliya Rizwan", "role": "Workspace Admin"}


@dataclass(frozen=True)
class Page:
    slug: str
    label: str
    icon: str
    title: str
    subtitle: str


PAGES = [
    Page("overview", "Overview", "layout-dashboard", "Workforce overview",
         "Who is leaving, where it is happening and how exposed the active workforce is."),
    Page("people", "People", "users", "People",
         "Everyone in the database. Add, edit, import and export employee records."),
    Page("risk", "Risk radar", "shield-alert", "Risk radar",
         "Active employees ranked by predicted attrition risk, with the factors behind each score."),
    Page("lab", "What-if lab", "flask-conical", "What-if lab",
         "Change a profile and watch the predicted risk move. See which interventions help most."),
    Page("model", "Model", "cpu", "Model",
         "How the prediction model performs, what it relies on, and retraining on your own data."),
]
PAGE_BY_LABEL = {page.label: page for page in PAGES}
PAGE_BY_SLUG = {page.slug: page for page in PAGES}
