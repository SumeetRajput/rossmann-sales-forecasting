"""Smoke tests for all src/ modules. Used by GitHub Actions CI."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    from src import FeatureEngineer, build_rf_pipeline, build_gbm_pipeline
    from src import rmspe, evaluate_predictions, get_logger
    assert FeatureEngineer is not None


def test_rmspe_perfect():
    import numpy as np
    from src import rmspe
    assert rmspe(np.array([100, 200, 300]),
                 np.array([100, 200, 300])) == 0.0


def test_rmspe_handles_zeros():
    import numpy as np
    from src import rmspe
    result = rmspe(np.array([100, 0, 300]),
                   np.array([110, 50, 310]))
    assert result > 0 and not np.isnan(result) and not np.isinf(result)


def test_evaluate_returns_all_5_metrics():
    import numpy as np
    from src import evaluate_predictions
    m = evaluate_predictions(
        np.array([100, 200, 300, 400, 500]),
        np.array([110, 190, 310, 390, 510])
    )
    assert set(m.keys()) == {"RMSE", "MAE", "MAPE", "RMSPE", "R2"}
    for v in m.values():
        assert isinstance(v, float)


def test_pipelines_are_sklearn_pipelines():
    from sklearn.pipeline import Pipeline
    from src import build_rf_pipeline, build_gbm_pipeline
    assert isinstance(build_rf_pipeline(), Pipeline)
    assert isinstance(build_gbm_pipeline(), Pipeline)


def test_feature_engineer_produces_more_columns():
    import pandas as pd
    from src import FeatureEngineer
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "Date": pd.to_datetime(["2015-01-01", "2015-06-15", "2015-12-25"]),
        "DayOfWeek": [3, 0, 4], "Open": [1, 1, 1], "Promo": [1, 0, 1],
        "StateHoliday": ["None", "None", "c"], "SchoolHoliday": [0, 0, 0],
        "StoreType": ["a", "b", "c"], "Assortment": ["a", "b", "c"],
        "CompetitionDistance": [1000.0, 2000.0, 500.0],
        "CompetitionOpenSinceMonth": [3, 6, 9],
        "CompetitionOpenSinceYear": [2010, 2012, 2014],
        "Promo2": [0, 1, 1],
        "Promo2SinceWeek": [0, 10, 20],
        "Promo2SinceYear": [0, 2014, 2014],
        "PromoInterval": ["", "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"],
    })
    fe = FeatureEngineer()
    fe.fit(df)
    out = fe.transform(df)
    assert out.shape[1] > df.shape[1]
    for col in ["Year", "Month", "IsWeekend", "DaysToChristmas",
                "DaysToEaster", "Promo2Active", "CompetitionOpenMonths"]:
        assert col in out.columns, f"Missing: {col}"


def test_logger():
    from src import get_logger
    logger = get_logger("test_logger", log_dir="/tmp/test_logs")
    assert logger is not None
    logger.info("Test message")
