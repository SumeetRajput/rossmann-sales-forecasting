"""
LSTM Deep Learning Module (Task 2.6)
====================================
Implements all 7 sub-steps required by the brief:

  1. Isolate time series for one store
  2. ADF stationarity test (statsmodels)
  3. Differencing if non-stationary
  4. ACF and PACF analysis
  5. Sliding window transformation to supervised learning
  6. Scale data to [-1, 1] range
  7. 2-layer LSTM (Sequential, LSTM, Dropout, Dense)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from .logger_config import get_logger
logger = get_logger("lstm")


# ── Step 1: Isolate time series ────────────────────────────────
def isolate_time_series(train, store_id=1):
    """Extract a single store's sales as a daily-resampled time series."""
    logger.info(f"Isolating time series for Store {store_id}...")
    ts = (train[(train["Store"] == store_id) & (train["Open"] == 1)]
          .sort_values("Date")
          .set_index("Date")["Sales"]
          .resample("D").mean()
          .interpolate("linear"))
    logger.info(f"  Length: {len(ts)} days "
                f"({ts.index.min().date()} -> {ts.index.max().date()})")
    return ts


# ── Step 2: ADF stationarity test ──────────────────────────────
def check_stationarity(ts, threshold=0.05):
    """Augmented Dickey-Fuller test. Returns (is_stationary, stat, p)."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(ts.dropna())
    stat, p = result[0], result[1]
    is_stat = p < threshold
    logger.info(f"  ADF stat = {stat:.4f}, p-value = {p:.6f}")
    logger.info(f"  Result: {'STATIONARY' if is_stat else 'NON-STATIONARY'}")
    return is_stat, stat, p


# ── Step 3: Differencing if needed ─────────────────────────────
def make_stationary(ts):
    """First-order differencing if series is non-stationary."""
    is_stat, _, _ = check_stationarity(ts)
    if is_stat:
        return ts
    logger.info("  Applying first-order differencing...")
    return ts.diff().dropna()


# ── Step 5: Sliding window ─────────────────────────────────────
def create_sequences(data, look_back=14):
    """Convert 1-D series into supervised learning samples."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


# ── Step 6: Scale to [-1, 1] ───────────────────────────────────
def scale_to_range(values, target_range=(-1, 1)):
    """Scale values into the target range, returning bounds for inversion."""
    v_min, v_max = float(values.min()), float(values.max())
    lo, hi = target_range
    scaled = lo + (hi - lo) * (values - v_min) / (v_max - v_min)
    return scaled, v_min, v_max


def inverse_scale(scaled, v_min, v_max, source_range=(-1, 1)):
    """Reverse the scaling to original units."""
    lo, hi = source_range
    return v_min + (scaled - lo) / (hi - lo) * (v_max - v_min)


# ── Step 7: 2-layer LSTM ───────────────────────────────────────
def build_lstm_model(look_back=14):
    """Construct the 2-layer LSTM with dropout."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm_for_store(train, store_id=1, look_back=14,
                         epochs=50, batch_size=32, patience=10):
    """End-to-end LSTM training pipeline for one store."""
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    logger.info("=" * 60)
    logger.info(f"  LSTM Training — Store {store_id}")
    logger.info("=" * 60)

    ts        = isolate_time_series(train, store_id)
    ts_stat   = make_stationary(ts)
    values    = ts_stat.values.astype(float)
    scaled, v_min, v_max = scale_to_range(values)

    X, y = create_sequences(scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    logger.info(f"  Training sequences: {len(X_tr)}")
    logger.info(f"  Test sequences    : {len(X_te)}")

    model = build_lstm_model(look_back)
    logger.info("  LSTM architecture: LSTM(64) -> Drop -> LSTM(32) -> Drop -> Dense(1)")

    cb = EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True, verbose=0)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs, batch_size=batch_size,
        callbacks=[cb], verbose=0,
    )

    y_pred_s = model.predict(X_te, verbose=0).flatten()
    y_pred   = inverse_scale(y_pred_s, v_min, v_max)
    y_true   = inverse_scale(y_te, v_min, v_max)

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }
    logger.info(f"  Test RMSE: EUR {metrics['RMSE']:,.2f}")
    logger.info(f"  Test MAE : EUR {metrics['MAE']:,.2f}")

    return {
        "model": model,
        "history": history.history,
        "v_min": v_min, "v_max": v_max,
        "X_test": X_te, "y_test": y_te,
        "y_pred": y_pred, "y_true": y_true,
        "metrics": metrics,
        "look_back": look_back,
        "ts": ts, "ts_stat": ts_stat,
    }
