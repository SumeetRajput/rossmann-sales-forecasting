"""
Pipeline Builder Module (Tasks 2.2 and 2.3)
===========================================
Builds sklearn Pipelines chaining: FeatureEngineer -> Imputer -> Scaler -> Model.

Includes the RMSPE loss function — the official Rossmann Kaggle metric.
"""

import numpy as np
from sklearn.pipeline       import Pipeline
from sklearn.impute         import SimpleImputer
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics        import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)

from .feature_engineering import FeatureEngineer


def build_rf_pipeline(n_estimators=400, max_depth=25, random_state=42):
    """Random Forest pipeline — tuned for higher R² and lower RMSPE."""
    return Pipeline([
        ("fe",      FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features=0.5,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])

def build_gbm_pipeline(n_estimators=500, learning_rate=0.03,
                      max_depth=8, random_state=42):
    """Gradient Boosting pipeline — tuned for higher R² and lower RMSPE."""
    return Pipeline([
        ("fe",      FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=random_state,
        )),
    ])

def rmspe(y_true, y_pred):
    """
    Root Mean Squared Percentage Error.

    Official Rossmann Kaggle metric. Chosen because stores have very
    different revenue scales — RMSPE measures percentage error so
    a 10% mistake counts equally whether the store earns EUR 1k or EUR 15k.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    pct_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_errors ** 2)) * 100)


def evaluate_predictions(y_true, y_pred):
    """Compute all 5 evaluation metrics in one call."""
    return {
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "MAPE":  float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "RMSPE": rmspe(y_true, y_pred),
        "R2":    float(r2_score(y_true, y_pred)),
    }
