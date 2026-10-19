"""Model-estimated interventions."""

import numpy as np
import pandas as pd
import pytest

from src.predict import AttritionPredictor
from src.utils import MODEL_PATH
from ui.recommendations import LEVER_BY_KEY, LEVERS, apply_lever_to_profile, lever_effects, ranked_levers

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")

PROFILE = {"Age": 27, "Department": "Sales", "MonthlyIncome": 35000.0, "YearsAtCompany": 1.0, "JobSatisfaction": 2,
           "WorkLifeBalance": 2, "PerformanceRating": 3, "OverTime": "Yes", "DistanceFromHome": 30.0}


@pytest.fixture(scope="module")
def predictor():
    return AttritionPredictor(MODEL_PATH)


def test_levers_apply_to_a_profile():
    assert apply_lever_to_profile(LEVER_BY_KEY["overtime"], PROFILE)["OverTime"] == "No"
    assert apply_lever_to_profile(LEVER_BY_KEY["engagement"], PROFILE)["JobSatisfaction"] == 3
    assert apply_lever_to_profile(LEVER_BY_KEY["balance"], PROFILE)["WorkLifeBalance"] == 3
    assert apply_lever_to_profile(LEVER_BY_KEY["pay"], PROFILE)["MonthlyIncome"] == pytest.approx(40250, abs=100)
    assert apply_lever_to_profile(LEVER_BY_KEY["remote"], PROFILE)["DistanceFromHome"] == 15


def test_effects_are_negative_where_a_lever_applies_and_nan_where_it_does_not(predictor):
    happy = {**PROFILE, "OverTime": "No", "JobSatisfaction": 5, "WorkLifeBalance": 5, "DistanceFromHome": 3.0}
    effects = lever_effects(predictor, pd.DataFrame([PROFILE, happy]))
    assert list(effects.columns) == [lever.key for lever in LEVERS]
    assert effects.loc[0, "overtime"] < 0
    assert effects.loc[0, "pay"] < 0
    assert np.isnan(effects.loc[1, "overtime"])
    assert np.isnan(effects.loc[1, "engagement"])
    assert np.isnan(effects.loc[1, "remote"])


def test_ranked_levers_are_sorted_and_thresholded(predictor):
    effects = lever_effects(predictor, pd.DataFrame([PROFILE]))
    ranked = ranked_levers(effects.iloc[0])
    assert ranked, "a high-risk profile should have at least one useful lever"
    deltas = [delta for _, delta in ranked]
    assert deltas == sorted(deltas)
    assert all(delta <= -0.02 for delta in deltas)
    assert ranked_levers(pd.Series({"pay": -0.001, "overtime": np.nan})) == []
