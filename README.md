# Rossmann Pharmaceuticals — Sales Forecasting

End-to-end machine learning pipeline predicting daily sales for 1,115 stores 6 weeks ahead.
Built for NextHikes IT Solutions sprint project · Final submission: 15 June 2026

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Versioned-13ADC7)](https://dvc.org/)
[![Heroku](https://img.shields.io/badge/Heroku-Deployable-430098)](https://heroku.com/)

---

## What This Project Does

Rossmann store managers currently forecast sales by gut feeling. The finance team needs accurate
6-week-ahead predictions across all 1,115 stores to plan staff, inventory, and budget. This project
delivers that prediction system end-to-end.

```
Raw CSVs  →  Cleaning  →  Feature Engineering  →  Model Training  →  Predictions  →  Dashboard
   ↓             ↓                ↓                    ↓                  ↓             ↓
 1M+ rows    0 missing       17 features        RF + GBM + LSTM    submission.csv  Streamlit UI
                                                  MLflow tracked                   8 pages
```

---

## How to Run It

### Step 1 — Install dependencies
```bash
.\py311.bat -m pip install -r requirements.txt
```

### Step 2 — Place CSV files
```
data/
├── train.csv
├── store.csv
└── test.csv
```

### Step 3 — Train the model
```bash
.\py311.bat train.py                 # full data, ~20 min
.\py311.bat train.py --sample 0.2    # 20% sample, ~3 min for testing
.\py311.bat train.py --skip-lstm     # skip deep learning
```

This generates:
- `outputs/models/<timestamp>_GradientBoosting.pkl` — best sklearn pipeline
- `outputs/models/<timestamp>_LSTM_Store1.keras` — LSTM neural network
- `outputs/data/submission.csv` — 41,088 test predictions
- `outputs/plots/*.png` — EDA + feature importance + residual charts
- `outputs/mlruns/` — MLflow experiment tracking data
- `outputs/logs/<timestamp>.log` — full session log

### Step 4 — View MLflow dashboard
```bash
.\py311.bat -m mlflow ui --backend-store-uri outputs/mlruns --port 5000
```
Open http://localhost:5000

### Step 5 — Launch Streamlit dashboard
```bash
.\py311.bat -m streamlit run app.py
```
Open http://localhost:8501

---

## Project Structure

```
rossmann_final/
├── src/                          # Modular source package
│   ├── __init__.py
│   ├── data_loader.py            # Load, merge, clean CSVs
│   ├── feature_engineering.py    # FeatureEngineer transformer (17 features)
│   ├── pipeline_builder.py       # sklearn pipelines + RMSPE
│   ├── lstm_model.py             # LSTM training (Task 2.6)
│   └── logger_config.py          # Centralised logger (Task 1.2)
│
├── train.py                      # Main training orchestrator with MLflow
├── app.py                        # Streamlit dashboard (8 pages)
│
├── tests/
│   └── test_modules.py           # 7 smoke tests for CI
│
├── data/                         # Raw CSV files (DVC-tracked)
├── outputs/
│   ├── plots/                    # All generated charts
│   ├── models/                   # Pickled models with timestamps
│   ├── data/                     # submission.csv
│   ├── logs/                     # Training logs
│   ├── mlruns/                   # MLflow tracking data
│   └── metrics.json              # Latest run metrics (DVC tracked)
│
├── .github/workflows/ci.yml      # GitHub Actions CI/CD pipeline
├── dvc.yaml + params.yaml        # DVC pipeline + hyperparameters
├── Procfile, setup.sh, runtime.txt  # Heroku deployment files
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Streamlit Dashboard — 8 Pages

| Page | What it shows |
|------|---------------|
| **🏠 Overview**          | Hero, 4 KPIs, sales trend chart, pipeline visualisation, tech stack |
| **📊 EDA Explorer**      | 7 tabs: Promotions, Seasonality, Correlation, Store Types, Day of Week, Competition, Customers |
| **🔮 Sales Forecast**    | Pick any store + promo + weeks → predict + chart + CSV download |
| **🤖 Model Analysis**    | Feature importance with adjustable top-N, metric reference guide |
| **🧠 LSTM Deep Learning** | Time-series chart, ADF test, ACF/PACF, architecture diagram |
| **📁 Batch Predict**     | Upload CSV → predict → download results CSV |
| **📈 MLflow Tracking**   | View all training runs and compare metrics |
| **💡 Business Insights** | Strategic findings + production recommendations |

---

## Brief Compliance — Every Requirement

| Brief Section | Status | Where in Code |
|---------------|--------|---------------|
| Task 1.1 EDA — 7+ questions answered | ✅ | EDA Explorer page (7 tabs) |
| Task 1.2 Logging | ✅ | `src/logger_config.py` |
| Task 2.1 Preprocessing | ✅ | `src/feature_engineering.py` (17 features) |
| Task 2.2 sklearn Pipelines | ✅ | `src/pipeline_builder.py` |
| Task 2.3 RMSPE loss | ✅ | `pipeline_builder.rmspe()` |
| Task 2.4 Feature importance + CI | ✅ | `train.py` post-prediction analysis |
| Task 2.5 Model serialisation | ✅ | `save_model_with_timestamp()` |
| Task 2.6 LSTM (2-layer, sliding window, [-1,1]) | ✅ | `src/lstm_model.py` |
| Task 2.7 MLflow | ✅ | `train.py` `mlflow.log_*` calls |
| Task 3 Streamlit dashboard | ✅ | `app.py` (8 pages) |
| Bonus DVC | ✅ | `dvc.yaml` + `params.yaml` |
| Bonus Heroku | ✅ | `Procfile` + `setup.sh` + `runtime.txt` |
| Bonus CI/CD | ✅ | `.github/workflows/ci.yml` |
| Bonus Code Modularisation | ✅ | `src/` package with 5 modules |

---

## Why Certain Design Choices

### Why RMSPE?
Stores have very different revenue scales. A €500 error is 50% off for a small store but only 3%
for a large one. RMSPE measures **percentage** error, treating all stores fairly. It is also the
official Rossmann Kaggle metric.

### Why time-based train/val split?
Random splitting leaks future data into training. We use **last 6 weeks as validation** — the same
horizon we predict in production. This gives an honest estimate of real-world performance.

### Why sklearn Pipelines?
1. **No data leakage** — scaler only learns from training data
2. **Reproducibility** — same steps run in same order every time
3. **Deployment** — save ONE object that handles everything

### Why 17 engineered features?
Models cannot learn from a raw `Date` string. We extract Year, Month, Day, DayOfWeek, IsWeekend,
Quarter, Week, plus Christmas/Easter/NewYear distances and month-position flags — 15 from Date alone,
plus Promo2Active and CompetitionOpenMonths.

---

## CI/CD

`.github/workflows/ci.yml` automatically:
1. **Every push** → lint with flake8, run pytest, verify imports
2. **Merge to main** → deploy to Heroku

To enable Heroku deployment, set these GitHub secrets:
- `HEROKU_API_KEY`
- `HEROKU_APP_NAME`
- `HEROKU_EMAIL`

---

## DVC Data Versioning

```bash
dvc init
dvc add data/train.csv data/store.csv data/test.csv
dvc repro                       # run the pipeline
dvc metrics diff                # compare metrics across versions
```

---

## Heroku Deployment

```bash
heroku create rossmann-forecast
heroku buildpacks:set heroku/python
git push heroku main
heroku open
```

`Procfile` runs Streamlit on the port Heroku assigns. `setup.sh` configures the dark theme.

---

## Author

**Sumeet Rajput** · NextHikes IT Solutions Sprint Project
Final submission deadline: 15 June 2026
