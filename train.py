"""
train.py — Main Training Pipeline
=================================
Orchestrates the complete Rossmann ML pipeline:

  1. Load and clean data       (uses src/data_loader.py)
  2. EDA charts                (saves to outputs/plots/)
  3. Time-based train/val split (last 6 weeks reserved for validation)
  4. Train Random Forest       (logged to MLflow)
  5. Train Gradient Boosting   (logged to MLflow)
  6. Compare, pick best by RMSPE
  7. Feature importance, confidence intervals, residual analysis
  8. Serialize best model (timestamped .pkl)
  9. Train LSTM deep learning model on Store 1
 10. Generate predictions for test set -> submission.csv

Usage:
    python train.py                 # full data, ~20 minutes
    python train.py --sample 0.2    # 20% sample, ~3 minutes for testing
    python train.py --skip-lstm     # skip deep learning step
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

# Force MLflow to use SQLite backend
os.makedirs("outputs", exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + os.path.abspath("outputs/mlruns.db").replace("\\", "/")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import argparse
import json
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, norm
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

from src import (
    load_and_prepare,
    build_rf_pipeline, build_gbm_pipeline,
    rmspe, evaluate_predictions,
    get_logger,
    train_lstm_for_store,
)

logger = get_logger("train")


# ── Helpers ────────────────────────────────────────────────────
def time_based_split(train_open, X_all, y_all, weeks=6):
    """Split data by date — train on past, validate on last N weeks."""
    cutoff = train_open["Date"].max() - pd.Timedelta(weeks=weeks)
    mask_train = train_open["Date"] <  cutoff
    mask_val   = train_open["Date"] >= cutoff

    X_train = X_all[mask_train.values]
    X_val   = X_all[mask_val.values]
    y_train = y_all[mask_train.values]
    y_val   = y_all[mask_val.values]

    logger.info(f"  Cutoff date    : {cutoff.date()}")
    logger.info(f"  Training rows  : {len(X_train):,}")
    logger.info(f"  Validation rows: {len(X_val):,}")
    return X_train, X_val, y_train, y_val


def log_metrics_console(metrics, name):
    """Pretty-print evaluation metrics."""
    logger.info(f"  --- {name} ---")
    logger.info(f"    RMSE  : EUR {metrics['RMSE']:>10,.2f}")
    logger.info(f"    MAE   : EUR {metrics['MAE']:>10,.2f}")
    logger.info(f"    MAPE  :     {metrics['MAPE']:>10,.2f}%")
    logger.info(f"    RMSPE :     {metrics['RMSPE']:>10,.2f}%")
    logger.info(f"    R2    :     {metrics['R2']:>10,.4f}")


def train_and_log_model(pipeline, name, X_train, y_train, X_val, y_val, params):
    """Train a pipeline and log everything to MLflow."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(params)
        mlflow.log_param("model_type", name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))

        logger.info(f"\n  Training {name}...")
        pipeline.fit(X_train, y_train)

        y_pred  = pipeline.predict(X_val)
        metrics = evaluate_predictions(y_val, y_pred)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Save model artifact to MLflow
        try:
            mlflow.sklearn.log_model(pipeline, "model")
        except Exception as e:
            logger.warning(f"  Could not log model to MLflow: {e}")

        log_metrics_console(metrics, name)
        return pipeline, y_pred, metrics


# ── EDA charts ─────────────────────────────────────────────────
def generate_eda_charts(train_open, train, test):
    """Save the 5 standard EDA charts to outputs/plots/."""
    logger.info("\nGenerating EDA charts (saving to outputs/plots/)...")
    sns.set_theme(style="whitegrid", palette="muted")
    out = "outputs/plots"

    # 1. Sales over time
    daily = train_open.groupby("Date")["Sales"].mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily.values, color="#0984E3", lw=1)
    ax.fill_between(daily.index, daily.values, alpha=0.1, color="#0984E3")
    ax.set_title("Average Daily Sales Across All Stores",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Avg Sales (EUR)")
    plt.tight_layout()
    plt.savefig(f"{out}/chart1_sales_over_time.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 2. Promo effect
    ps = train_open.groupby("Promo")["Sales"].mean()
    lift = (ps[1] - ps[0]) / ps[0] * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["No Promo", "Promo"], ps.values,
                  color=["#B2BABB", "#F39C12"], edgecolor="white", lw=2, width=0.5)
    for b, v in zip(bars, ps.values):
        ax.text(b.get_x() + b.get_width()/2, v + 50, f"EUR {v:,.0f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_title(f"Promo Effect on Sales (+{lift:.1f}% lift)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg Daily Sales (EUR)")
    plt.tight_layout()
    plt.savefig(f"{out}/chart2_promo_effect.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Promo lifts sales by {lift:.1f}%")

    # 3. Seasonality
    df_s = train_open.copy()
    df_s["Month"] = df_s["Date"].dt.month
    df_s["Year"]  = df_s["Date"].dt.year
    monthly = df_s.groupby(["Year", "Month"])["Sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    cols_yr = ["#0984E3", "#00B894", "#E17055", "#6C5CE7"]
    for i, (yr, grp) in enumerate(monthly.groupby("Year")):
        ax.plot(grp["Month"], grp["Sales"], marker="o",
                lw=2.5, color=cols_yr[i % 4], label=str(yr))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Monthly Average Sales by Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg Daily Sales (EUR)")
    ax.legend(title="Year")
    plt.tight_layout()
    plt.savefig(f"{out}/chart3_monthly_sales.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 4. Correlation
    sample = train_open.sample(10000, random_state=42)
    r = sample["Sales"].corr(sample["Customers"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(sample["Customers"], sample["Sales"],
                    alpha=0.2, color="#6C5CE7", s=8)
    m, b = np.polyfit(sample["Customers"], sample["Sales"], 1)
    x_line = np.linspace(sample["Customers"].min(), sample["Customers"].max(), 100)
    axes[0].plot(x_line, m*x_line + b, color="#D63031", lw=2)
    axes[0].set_title(f"Sales vs Customers (r={r:.3f})", fontweight="bold")
    axes[0].set_xlabel("Customers"); axes[0].set_ylabel("Sales (EUR)")
    corr_m = sample[["Sales", "Customers", "Promo", "SchoolHoliday"]].corr()
    sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=axes[1], linewidths=0.5, vmin=-1, vmax=1)
    axes[1].set_title("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out}/chart4_correlation.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Customers <-> Sales correlation: r = {r:.3f}")

    # 5. Day of week
    df_d = train_open.copy()
    df_d["DayName"] = df_d["Date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    avg = df_d.groupby("DayName")["Sales"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(12, 5))
    bc = ["#74B9FF"]*5 + ["#FD79A8", "#FD79A8"]
    bars = ax.bar(order, avg.values, color=bc, edgecolor="white", lw=1.5)
    for b, v in zip(bars, avg.values):
        ax.text(b.get_x() + b.get_width()/2, v + 30, f"EUR{v:,.0f}",
                ha="center", fontsize=8, rotation=90)
    ax.set_title("Avg Sales by Day of Week", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg Sales (EUR)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out}/chart5_weekday_sales.png", dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("  All 5 EDA charts saved!")


# ── Post-prediction analysis ───────────────────────────────────
def make_feature_importance_plot(pipeline, X_val, model_name):
    """Generate the feature importance plot."""
    model_step = pipeline.named_steps["model"]
    fe_step    = pipeline.named_steps["fe"]
    X_fe       = fe_step.transform(X_val.head(200).copy())
    feat_names = X_fe.columns.tolist()
    imps = pd.Series(model_step.feature_importances_,
                     index=feat_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    imps.tail(20).plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top 20 Feature Importances ({model_name})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for i, (v, n) in enumerate(zip(imps.tail(20).values, imps.tail(20).index)):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/plots/chart6_feature_importance.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    return imps


def make_residual_plot(y_val, y_pred, model_name):
    """3-panel residual analysis chart."""
    residuals = np.asarray(y_val) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.2, s=6, color="#636E72")
    axes[0].axhline(0, color="#D63031", lw=2)
    axes[0].set_xlabel("Predicted (EUR)")
    axes[0].set_ylabel("Residual (EUR)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=70, color="#74B9FF",
                 edgecolor="white", density=True)
    x_c = np.linspace(residuals.min(), residuals.max(), 200)
    axes[1].plot(x_c, norm.pdf(x_c, residuals.mean(), residuals.std()),
                 "r-", lw=2.5, label="Normal")
    axes[1].set_title("Residual Distribution")
    axes[1].legend()

    probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")

    fig.suptitle(f"Residual Analysis — {model_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/plots/chart8_residuals.png",
                dpi=120, bbox_inches="tight")
    plt.close()


def make_confidence_plot(y_val, y_pred, n=150):
    """Predictions with confidence band."""
    std = (np.asarray(y_val) - np.asarray(y_pred)).std()
    ci_lo = y_pred - 1.96 * std
    ci_hi = y_pred + 1.96 * std

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(np.arange(n), ci_lo[:n], ci_hi[:n],
                    alpha=0.3, color="#74B9FF", label="95% CI")
    ax.plot(y_pred[:n], color="#0984E3", lw=2, label="Predicted")
    ax.plot(y_val[:n], "r--", lw=1.2, alpha=0.8, label="Actual")
    ax.set_title("Predictions with 95% Confidence Band",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Sales (EUR)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/chart7_confidence.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    inside = float(np.mean((y_val[:n] >= ci_lo[:n]) &
                          (y_val[:n] <= ci_hi[:n])) * 100)
    return inside


def save_model_with_timestamp(pipeline, name):
    """Serialise the model with a timestamp in the filename (Task 2.5)."""
    ts   = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    path = os.path.join("outputs", "models",
                        f"{ts}_{name.replace(' ', '_')}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    size_kb = os.path.getsize(path) / 1024
    logger.info(f"  Model saved: {path} ({size_kb:.1f} KB)")
    return path


def generate_test_predictions(pipeline, test):
    """Generate the final submission.csv."""
    test_copy = test.copy()
    test_copy["Open"] = test_copy["Open"].fillna(1)

    preds = np.zeros(len(test_copy))
    closed_mask = test_copy["Open"] == 0
    open_data   = test_copy[~closed_mask].copy()

    for col in ["Id", "Customers"]:
        if col in open_data.columns:
            open_data = open_data.drop(columns=[col])

    preds[~closed_mask] = pipeline.predict(open_data).clip(min=0)

    submission = pd.DataFrame({
        "Id":    test_copy["Id"],
        "Sales": preds.round(2),
    })
    out_path = "outputs/data/submission.csv"
    submission.to_csv(out_path, index=False)
    logger.info(f"  Submission saved: {out_path} ({len(submission):,} rows)")
    logger.info(f"  Min: EUR {submission['Sales'].min():,.2f}  "
                f"Mean: EUR {submission['Sales'].mean():,.2f}  "
                f"Max: EUR {submission['Sales'].max():,.2f}")
    return submission


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training step")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA chart generation")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri("file:./outputs/mlruns")
    mlflow.set_experiment("rossmann-sales-forecasting")

    logger.info("=" * 60)
    logger.info("  ROSSMANN PHARMACEUTICALS — Sales Forecasting")
    logger.info(f"  Training session started: {datetime.datetime.now()}")
    logger.info("=" * 60)

    # 1-3. Load and clean data
    train, test, store, train_open = load_and_prepare(args.data_dir)

    # 4. EDA
    if not args.skip_eda:
        generate_eda_charts(train_open, train, test)

    # 5. Prepare features
    y_all = train_open["Sales"].values
    X_all = train_open.drop(columns=["Sales", "Customers"])

    if args.sample < 1.0:
        n = int(len(X_all) * args.sample)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), n, replace=False)
        X_all      = X_all.iloc[idx].reset_index(drop=True)
        y_all      = y_all[idx]
        train_open = train_open.iloc[idx].reset_index(drop=True)
        logger.info(f"\n  Sampled to {n:,} rows ({args.sample*100:.0f}%)")

    # 6. Time-based split
    logger.info("\nPerforming time-based train/validation split...")
    X_train, X_val, y_train, y_val = time_based_split(train_open, X_all, y_all)

    # 7. Train both models with MLflow tracking
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING MODELS (logged to MLflow)")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="rossmann_session") as parent:
        mlflow.log_param("sample_fraction", args.sample)
        mlflow.log_param("split_strategy", "time_based_last_6_weeks")

        # Random Forest
        rf_params = {"n_estimators": 300, "max_depth": 20, "min_samples_split": 4}
        rf, y_pred_rf, rf_metrics = train_and_log_model(
            build_rf_pipeline(), "RandomForest",
            X_train, y_train, X_val, y_val, rf_params
        )

        # Gradient Boosting
        gbm_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6}
        gbm, y_pred_gbm, gbm_metrics = train_and_log_model(
            build_gbm_pipeline(), "GradientBoosting",
            X_train, y_train, X_val, y_val, gbm_params
        )

        # 8. Pick winner
        logger.info("\n" + "=" * 60)
        logger.info("  MODEL COMPARISON")
        logger.info("=" * 60)
        if rf_metrics["RMSPE"] < gbm_metrics["RMSPE"]:
            best, best_name = rf, "RandomForest"
            best_metrics, y_pred = rf_metrics, y_pred_rf
        else:
            best, best_name = gbm, "GradientBoosting"
            best_metrics, y_pred = gbm_metrics, y_pred_gbm

        logger.info(f"  Winner: {best_name}")
        logger.info(f"  RMSPE : {best_metrics['RMSPE']:.2f}%")
        logger.info(f"  R^2   : {best_metrics['R2']:.4f}")

        mlflow.log_param("winning_model", best_name)
        mlflow.log_metric("best_RMSPE", best_metrics["RMSPE"])
        mlflow.log_metric("best_R2",    best_metrics["R2"])

        # 9. Post-prediction analysis
        logger.info("\nGenerating analysis plots...")
        imps   = make_feature_importance_plot(best, X_val, best_name)
        ci_pct = make_confidence_plot(y_val, y_pred)
        make_residual_plot(y_val, y_pred, best_name)

        try:
            mlflow.log_artifact("outputs/plots/chart6_feature_importance.png")
            mlflow.log_artifact("outputs/plots/chart7_confidence.png")
            mlflow.log_artifact("outputs/plots/chart8_residuals.png")
        except Exception as e:
            logger.warning(f"  MLflow artifact log failed: {e}")

        top3 = imps.tail(3).index[::-1].tolist()
        logger.info(f"  Top 3 features: {top3}")
        logger.info(f"  Actuals inside 95% CI: {ci_pct:.1f}% (target ~95%)")

        # 10. Save best model (Task 2.5)
        logger.info("\nSerialising best model...")
        model_path = save_model_with_timestamp(best, best_name)
        try:
            mlflow.log_artifact(model_path)
        except Exception:
            pass

        # 11. Generate test predictions
        logger.info("\nGenerating final test predictions...")
        submission = generate_test_predictions(best, test)
        try:
            mlflow.log_artifact("outputs/data/submission.csv")
        except Exception:
            pass

        # Save metrics to JSON for DVC
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "best_model": best_name,
                "RandomForest":     rf_metrics,
                "GradientBoosting": gbm_metrics,
                "best":             best_metrics,
            }, f, indent=2)

        # 12. LSTM (Task 2.6)
        if not args.skip_lstm:
            logger.info("\n" + "=" * 60)
            logger.info("  LSTM DEEP LEARNING (Task 2.6)")
            logger.info("=" * 60)
            with mlflow.start_run(run_name="LSTM_Store1", nested=True):
                lstm_result = train_lstm_for_store(train, store_id=1, look_back=14)
                mlflow.log_param("model_type", "LSTM")
                mlflow.log_param("look_back", 14)
                mlflow.log_param("architecture", "LSTM(64)+Drop+LSTM(32)+Drop+Dense(1)")
                mlflow.log_metric("LSTM_RMSE", lstm_result["metrics"]["RMSE"])
                mlflow.log_metric("LSTM_MAE",  lstm_result["metrics"]["MAE"])

                lstm_ts   = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                lstm_path = os.path.join("outputs", "models",
                                         f"{lstm_ts}_LSTM_Store1.keras")
                try:
                    lstm_result["model"].save(lstm_path)
                    logger.info(f"  LSTM saved: {lstm_path}")
                except Exception as e:
                    logger.warning(f"  LSTM save failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best model RMSPE: {best_metrics['RMSPE']:.2f}%")
    logger.info(f"  Best model R^2  : {best_metrics['R2']:.4f}")
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    1. View MLflow dashboard:")
    logger.info("         mlflow ui --backend-store-uri outputs/mlruns --port 5000")
    logger.info("    2. Launch Streamlit dashboard:")
    logger.info("         streamlit run app.py")


if __name__ == "__main__":
    main()
