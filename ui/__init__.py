"""Front-end design system for the Employee Attrition Predictor (Streamlit).

Layout of this package
----------------------
config.py        brand, workspace and navigation constants
theme.py         design tokens (light / dark), fonts and the global stylesheet
styles.css       the stylesheet itself, written against the tokens
icons.py         Lucide icons as CSS masks (``icon_data.py`` holds the paths)
format.py        currency / number / time formatting (Indian digit grouping)
components.py    small HTML building blocks: chips, KPI cards, pill bars, gauges
charts.py        themed Altair charts
shell.py         sticky app bar, navigation and workspace / profile menus
dialogs.py       add / edit / delete employee dialogs
recommendations.py   model-estimated interventions used by the lab and the risk radar
pages/           one module per page
"""
