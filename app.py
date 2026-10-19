from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src.predict import ModelLoadError, PredictionError
from src.utils import METRICS_PATH
from ui import shell, state, theme
from ui.config import BRAND_NAME, PAGES

st.set_page_config(page_title=BRAND_NAME, page_icon="static/favicon.png", layout="wide",
                   initial_sidebar_state="collapsed")


def _model_summary() -> tuple[str, str]:
    """Name and training time of the model that is currently loaded."""
    bundle = state.get_predictor().bundle
    trained = str(bundle.get("trained_at", ""))[:16].replace("T", " ")
    return str(bundle.get("model_name", "Model")), (f"{trained} UTC" if trained else "unknown")


def main() -> None:
    shell.init_state()
    theme.inject(st.session_state["theme"], PAGES)

    try:
        db = state.get_db()
        predictor = state.get_predictor()
    except ModelLoadError as exc:
        st.error(f"Could not load the model: {exc}")
        st.info("Train it first: `python -m src.generate_data` then `python -m src.train_model`.")
        st.stop()
    except Exception as exc:  # the database file may be locked or corrupted
        st.error(f"Could not open the employee database: {exc}")
        st.stop()

    try:
        scored, effects = state.load_workforce(state.data_stamp())
    except PredictionError as exc:
        st.error(str(exc))
        st.stop()

    model_name, trained = _model_summary()
    page = shell.render_appbar(model_name, trained, len(scored))
    shell.show_flash()

    with st.container(key=f"page_{page.slug}"):
        shell.render_header(page)
        if page.slug == "overview":
            from ui.pages import overview
            overview.render(scored, effects)
        elif page.slug == "people":
            from ui.pages import people
            people.render(db, scored)
        elif page.slug == "risk":
            from ui.pages import risk
            risk.render(predictor, scored, effects)
        elif page.slug == "lab":
            from ui.pages import lab
            lab.render(db, predictor, scored)
        elif page.slug == "model":
            from ui.pages import model
            model.render(db)


main()
