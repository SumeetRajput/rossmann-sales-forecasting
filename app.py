"""
app.py — Interactive Streamlit Dashboard
=========================================
Production-grade dashboard showcasing the complete Rossmann ML pipeline.

8 interactive pages:
  1. Overview            - KPIs, pipeline visualisation
  2. EDA Explorer        - 7 tabs of interactive analytics
  3. Sales Forecast      - Pick any store, forecast 6 weeks ahead
  4. Model Analysis      - Feature importance, metrics, residuals
  5. LSTM Deep Learning  - Time series ADF, ACF/PACF, architecture
  6. Batch Predict       - Upload CSV, download predictions
  7. MLflow Tracking     - Compare all training runs
  8. Business Insights   - Strategic takeaways and recommendations
"""

import os
import sys
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Make src package importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src import FeatureEngineer
except Exception:
    pass


# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────
# CUSTOM CSS — Dark, professional theme
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0A0E1A; color: #E8EAF0; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1421 0%, #0A0E1A 100%);
    border-right: 1px solid #1F2A3F;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #131826 0%, #0F1421 100%);
    border: 1px solid #1F2A3F;
    border-radius: 14px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3B82F6, transparent);
}
[data-testid="stMetricLabel"] {
    color: #6B7896 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #E8EAF0 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; color: #9CA8C2 !important; }

.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 26px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(59,130,246,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.4);
}

.stTabs [data-baseweb="tab-list"] {
    background: #131826;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #1F2A3F;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6B7896;
    border-radius: 9px;
    font-weight: 500;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #3B82F6 !important;
    color: white !important;
}

.section-header {
    font-size: 11px;
    font-weight: 700;
    color: #6B7896;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 28px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1F2A3F;
}

.info-card {
    background: linear-gradient(135deg, #131826 0%, #0F1421 100%);
    border: 1px solid #1F2A3F;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
}
.info-card h3 {
    color: #E8EAF0;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 10px;
}
.info-card p {
    color: #9CA8C2;
    font-size: 13.5px;
    line-height: 1.7;
    margin: 0;
}

.insight {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(99,102,241,0.04));
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 3px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13.5px;
    color: #C5D2EE;
    line-height: 1.7;
}

.hero {
    background:
        radial-gradient(circle at 90% 10%, rgba(59,130,246,0.15) 0%, transparent 50%),
        linear-gradient(135deg, #131826 0%, #0F1421 100%);
    border: 1px solid #1F2A3F;
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 30px;
}
.hero h1 {
    font-size: 32px;
    font-weight: 700;
    color: #E8EAF0;
    margin: 0 0 10px;
}
.hero .subtitle { color: #9CA8C2; font-size: 15px; line-height: 1.6; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93C5FD;
    font-size: 11px;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 22px;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #3B82F6;
    border-radius: 50%;
    box-shadow: 0 0 8px #3B82F6;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 10px;
    background: #131826;
    border: 1px solid #1F2A3F;
    margin-bottom: 8px;
    transition: all 0.2s;
}
.pipeline-step:hover { border-color: #3B82F6; transform: translateX(4px); }
.step-num {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(99,102,241,0.1));
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    color: #93C5FD;
    flex-shrink: 0;
}
.step-title { font-size: 13.5px; font-weight: 600; color: #E8EAF0; }
.step-desc  { font-size: 12px; color: #6B7896; margin-top: 2px; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.status-pill.success {
    background: rgba(16,185,129,0.15);
    color: #6EE7B7;
    border: 1px solid rgba(16,185,129,0.3);
}
.status-pill.warning {
    background: rgba(245,158,11,0.15);
    color: #FCD34D;
    border: 1px solid rgba(245,158,11,0.3);
}
.status-pill::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# DATA + MODEL LOADERS
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv", low_memory=False, parse_dates=["Date"])
    store = pd.read_csv("data/store.csv", low_memory=False)
    test  = pd.read_csv("data/test.csv",  low_memory=False, parse_dates=["Date"])
    train = train.merge(store, on="Store", how="left")
    test  = test.merge(store,  on="Store", how="left")
    for df in [train, test]:
        df["CompetitionDistance"]       = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
        df["CompetitionOpenSinceYear"]  = df["CompetitionOpenSinceYear"].fillna(0)
        df["Promo2SinceWeek"]           = df["Promo2SinceWeek"].fillna(0)
        df["Promo2SinceYear"]           = df["Promo2SinceYear"].fillna(0)
        df["PromoInterval"]             = df["PromoInterval"].fillna("")
        df["Open"]                      = df["Open"].fillna(1)
        df["StateHoliday"]              = df["StateHoliday"].astype(str).replace("0","None")
    train_open = train[(train["Open"]==1) & (train["Sales"]>0)].copy()
    return train, test, store, train_open


@st.cache_resource
def load_latest_model():
    d = "outputs/models"
    if not os.path.exists(d):
        return None, None
    pkls = sorted([f for f in os.listdir(d) if f.endswith(".pkl")])
    if not pkls:
        return None, None
    path = os.path.join(d, pkls[-1])
    try:
        with open(path, "rb") as f:
            return pickle.load(f), pkls[-1]
    except Exception:
        return None, None


def style_chart(fig):
    fig.patch.set_facecolor("#131826")
    for ax in fig.get_axes():
        ax.set_facecolor("#131826")
        ax.tick_params(colors="#6B7896", labelsize=10)
        ax.xaxis.label.set_color("#9CA8C2")
        ax.yaxis.label.set_color("#9CA8C2")
        if ax.title:
            ax.title.set_color("#E8EAF0")
        for spine in ax.spines.values():
            spine.set_color("#1F2A3F")
        if ax.get_legend():
            ax.get_legend().get_frame().set_facecolor("#131826")
            ax.get_legend().get_frame().set_edgecolor("#1F2A3F")
            for txt in ax.get_legend().get_texts():
                txt.set_color("#9CA8C2")
    return fig


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 28px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#3B82F6,#6366F1);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:18px">🏪</div>
            <div>
                <div style="font-size:18px;font-weight:700;color:#E8EAF0">Rossmann</div>
                <div style="font-size:11px;color:#6B7896">Sales Forecasting Suite</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", [
        "🏠  Overview",
        "📊  EDA Explorer",
        "🔮  Sales Forecast",
        "🤖  Model Analysis",
        "🧠  LSTM Deep Learning",
        "📁  Batch Predict",
        "📈  MLflow Tracking",
        "💡  Business Insights",
    ], label_visibility="collapsed")

    st.markdown('<div class="section-header">Project Status</div>', unsafe_allow_html=True)
    try:
        _, _, _, train_open_s = load_data()
        st.markdown('<span class="status-pill success">Data Loaded</span>', unsafe_allow_html=True)
        st.metric("Stores", f"{train_open_s['Store'].nunique():,}")
        st.metric("Records", f"{len(train_open_s):,}")
    except Exception:
        st.markdown('<span class="status-pill warning">No Data</span>', unsafe_allow_html=True)
        st.caption("Place CSVs in data/ folder")

    m_obj, m_name = load_latest_model()
    if m_obj is not None:
        st.markdown('<span class="status-pill success">Model Loaded</span>', unsafe_allow_html=True)
        st.caption(f"`{m_name[:32]}…`")
    else:
        st.markdown('<span class="status-pill warning">No Model</span>', unsafe_allow_html=True)
        st.caption("Run train.py first")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#6B7896;line-height:1.6">
        <b style="color:#9CA8C2">NextHikes Sprint Project</b><br>
        Submission: 15 June 2026<br>
        Built by Sumeet Rajput
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# LOAD ONCE
# ──────────────────────────────────────────────────────────
try:
    train, test, store, train_open = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    data_error  = str(e)

model, model_name = load_latest_model()
model_loaded      = model is not None


# ──────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ──────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">NextHikes Sprint · Sales Forecasting</div>
        <h1>Rossmann Pharmaceuticals<br>End-to-End ML Pipeline</h1>
        <p class="subtitle">
            Predicting daily sales for 1,115 stores up to 6 weeks ahead<br>
            From raw CSVs to deployed predictions — built with scikit-learn,
            TensorFlow LSTM, MLflow tracking and Streamlit interface
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error(f"Cannot load data: {data_error}")
        st.info("Place train.csv, store.csv, test.csv in the data/ folder")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Stores", f"{train['Store'].nunique():,}", "pharmacies")
    with c2: st.metric("Training Rows", f"{len(train):,}", "Jan 2013 → Jul 2015")
    with c3:
        avg_s = train_open['Sales'].mean()
        st.metric("Avg Daily Sales", f"€{avg_s:,.0f}", "per store")
    with c4:
        lift = (train_open[train_open['Promo']==1]['Sales'].mean() /
                train_open[train_open['Promo']==0]['Sales'].mean() - 1) * 100
        st.metric("Promo Lift", f"+{lift:.1f}%", "vs no promo")

    st.markdown("")
    cL, cR = st.columns([2, 1])
    with cL:
        st.markdown('<div class="section-header">Sales Trend Over Time</div>', unsafe_allow_html=True)
        daily = train_open.groupby("Date")["Sales"].mean()
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.fill_between(daily.index, daily.values, alpha=0.15, color="#3B82F6")
        ax.plot(daily.index, daily.values, color="#3B82F6", lw=1.5)
        ax.set_ylabel("Avg Sales (€)")
        ax.grid(axis="y", alpha=0.08, color="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()

    with cR:
        st.markdown('<div class="section-header">Pipeline Stages</div>', unsafe_allow_html=True)
        stages = [
            ("01", "Load & Merge", "3 CSVs → 1M rows"),
            ("02", "Clean Data", "Zero missing values"),
            ("03", "EDA Charts", "Business insights"),
            ("04", "Feature Engineer", "17 new features"),
            ("05", "ML Pipeline", "FE → Imputer → Scaler"),
            ("06", "Train Models", "RF + GBM compared"),
            ("07", "LSTM Network", "Deep learning"),
            ("08", "Deploy", "Streamlit dashboard"),
        ]
        for num, title, desc in stages:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">{num}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)
    cs1, cs2, cs3, cs4 = st.columns(4)
    cards = [
        ("Data", ["pandas", "numpy", "scipy"]),
        ("ML Models", ["scikit-learn", "RandomForest", "GradientBoosting"]),
        ("Deep Learning", ["TensorFlow", "Keras LSTM", "EarlyStopping"]),
        ("MLOps", ["MLflow", "DVC", "Streamlit · Heroku"]),
    ]
    for col, (title, items) in zip([cs1, cs2, cs3, cs4], cards):
        with col:
            items_html = "<br>".join([f"• {it}" for it in items])
            st.markdown(f"""
            <div class="info-card">
                <h3>{title}</h3>
                <p>{items_html}</p>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# PAGE 2 — EDA EXPLORER
# ──────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>📊 EDA Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Interactive analysis answering 7 key business questions</p>", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data not loaded"); st.stop()

    tabs = st.tabs([
        "  Promotions  ", "  Seasonality  ", "  Correlation  ",
        "  Store Types  ", "  Day of Week  ", "  Competition  ",
        "  Customers  "
    ])

    # Tab 1: Promotions
    with tabs[0]:
        st.markdown('<div class="section-header">Promo Effect on Sales & Customers</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ps = train_open.groupby("Promo")["Sales"].mean()
            lift_s = (ps[1]-ps[0])/ps[0]*100
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["No Promo","Promo Active"], ps.values,
                          color=["#374151","#3B82F6"], edgecolor="#1F2A3F", width=0.5)
            for b, v in zip(bars, ps.values):
                ax.text(b.get_x()+b.get_width()/2, v+50, f"€{v:,.0f}",
                        ha="center", fontsize=11, fontweight="600", color="#E8EAF0")
            ax.set_title(f"Avg Sales (+{lift_s:.1f}%)")
            ax.set_ylabel("Avg Daily Sales (€)")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        with c2:
            pc = train_open.groupby("Promo")["Customers"].mean()
            lift_c = (pc[1]-pc[0])/pc[0]*100
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["No Promo","Promo Active"], pc.values,
                          color=["#374151","#10B981"], edgecolor="#1F2A3F", width=0.5)
            for b, v in zip(bars, pc.values):
                ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:,.0f}",
                        ha="center", fontsize=11, fontweight="600", color="#E8EAF0")
            ax.set_title(f"Avg Customers (+{lift_c:.1f}%)")
            ax.set_ylabel("Avg Customers")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="insight">💡 <b>Finding:</b> Promotions lift sales by <b>{lift_s:.1f}%</b> and
        customer count by <b>{lift_c:.1f}%</b>. Sales lift exceeds customer lift, meaning existing
        customers also spend more per visit during promos — not just new ones being attracted.</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Train vs Test Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            tc = train["Promo"].value_counts().sort_index()
            ax.pie(tc.values, labels=["No Promo","Promo"], autopct="%1.1f%%",
                   colors=["#374151","#3B82F6"], startangle=90,
                   wedgeprops={"edgecolor":"#1F2A3F","linewidth":2},
                   textprops={"color":"#E8EAF0","fontsize":11})
            ax.set_title("Train Set", color="#E8EAF0")
            fig.patch.set_facecolor("#131826")
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            tc = test["Promo"].value_counts().sort_index()
            ax.pie(tc.values, labels=["No Promo","Promo"], autopct="%1.1f%%",
                   colors=["#374151","#3B82F6"], startangle=90,
                   wedgeprops={"edgecolor":"#1F2A3F","linewidth":2},
                   textprops={"color":"#E8EAF0","fontsize":11})
            ax.set_title("Test Set", color="#E8EAF0")
            fig.patch.set_facecolor("#131826")
            st.pyplot(fig); plt.close()

    # Tab 2: Seasonality
    with tabs[1]:
        st.markdown('<div class="section-header">Monthly Patterns by Year</div>', unsafe_allow_html=True)
        df_s = train_open.copy()
        df_s["Month"] = df_s["Date"].dt.month
        df_s["Year"]  = df_s["Date"].dt.year
        monthly = df_s.groupby(["Year","Month"])["Sales"].mean().reset_index()
        mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig, ax = plt.subplots(figsize=(12, 4))
        cols_yr = ["#3B82F6","#10B981","#F59E0B","#8B5CF6"]
        for i, (yr, grp) in enumerate(monthly.groupby("Year")):
            ax.plot(grp["Month"], grp["Sales"], marker="o", markersize=7,
                    lw=2.5, color=cols_yr[i % 4], label=str(yr))
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(mnames)
        ax.set_title("Monthly Average Sales by Year")
        ax.set_ylabel("Avg Daily Sales (€)")
        ax.legend(title="Year")
        ax.axvspan(11.5, 12.5, alpha=0.12, color="#F59E0B")
        ax.grid(axis="y", alpha=0.08, color="white")
        plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        st.markdown("""
        <div class="insight">💡 <b>Finding:</b> December peaks every year — Christmas drives a ~30%
        spike. January always dips. This is why we created DaysToChristmas and DaysToEaster features.</div>
        """, unsafe_allow_html=True)

    # Tab 3: Correlation
    with tabs[2]:
        st.markdown('<div class="section-header">Sales vs Customers</div>', unsafe_allow_html=True)
        sample = train_open.sample(min(8000, len(train_open)), random_state=42)
        r = sample["Sales"].corr(sample["Customers"])
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(sample["Customers"], sample["Sales"], alpha=0.2, color="#3B82F6", s=6)
            m, b = np.polyfit(sample["Customers"], sample["Sales"], 1)
            x_l = np.linspace(sample["Customers"].min(), sample["Customers"].max(), 100)
            ax.plot(x_l, m*x_l+b, color="#F59E0B", lw=2, label=f"Trend (r={r:.3f})")
            ax.set_title("Sales vs Customers")
            ax.set_xlabel("Customers"); ax.set_ylabel("Sales (€)")
            ax.legend()
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        with c2:
            corr_m = sample[["Sales","Customers","Promo","SchoolHoliday"]].corr()
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="Blues",
                        ax=ax, linewidths=0.5, linecolor="#1F2A3F",
                        annot_kws={"color":"black","fontweight":"600"})
            ax.set_title("Correlation Heatmap")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        st.markdown(f"""
        <div class="insight">💡 <b>r = {r:.3f}</b> — strong positive correlation. Customer count is
        the single strongest predictor of sales.</div>
        """, unsafe_allow_html=True)

    # Tab 4: Store Types
    with tabs[3]:
        st.markdown('<div class="section-header">Sales by Store Type & Assortment</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st_avg = train_open.groupby("StoreType")["Sales"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(st_avg.index, st_avg.values,
                   color=["#3B82F6","#10B981","#F59E0B","#8B5CF6"][:len(st_avg)],
                   edgecolor="#1F2A3F", width=0.5)
            for i, v in enumerate(st_avg.values):
                ax.text(i, v+50, f"€{v:,.0f}", ha="center",
                        fontsize=10, fontweight="600", color="#E8EAF0")
            ax.set_title("By Store Type"); ax.set_ylabel("Avg Sales (€)")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        with c2:
            df_a = train_open.copy()
            df_a["AL"] = df_a["Assortment"].map({"a":"Basic","b":"Extra","c":"Extended"})
            av = df_a.groupby("AL")["Sales"].mean().reindex(["Basic","Extra","Extended"])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(av.index, av.values, color=["#374151","#3B82F6","#10B981"],
                   edgecolor="#1F2A3F", width=0.5)
            for i, v in enumerate(av.values):
                ax.text(i, v+50, f"€{v:,.0f}", ha="center",
                        fontsize=10, fontweight="600", color="#E8EAF0")
            ax.set_title("By Assortment"); ax.set_ylabel("Avg Sales (€)")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        st.markdown("""
        <div class="insight">💡 <b>Finding:</b> Store Type b earns most per store despite being least
        common. Extended assortment outperforms Basic by ~40%.</div>
        """, unsafe_allow_html=True)

    # Tab 5: Day of Week
    with tabs[4]:
        st.markdown('<div class="section-header">By Day of Week</div>', unsafe_allow_html=True)
        df_d = train_open.copy()
        df_d["DN"] = df_d["Date"].dt.day_name()
        ordr = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        avg = df_d.groupby("DN")["Sales"].mean().reindex(ordr)
        fig, ax = plt.subplots(figsize=(10, 4))
        bc = ["#3B82F6"]*5 + ["#8B5CF6","#8B5CF6"]
        bars = ax.bar(ordr, avg.values, color=bc, edgecolor="#1F2A3F", linewidth=1.5)
        for b, v in zip(bars, avg.values):
            ax.text(b.get_x()+b.get_width()/2, v+30, f"€{v:,.0f}",
                    ha="center", fontsize=9, color="#E8EAF0")
        ax.set_title("Avg Sales by Day of Week (blue=weekday, purple=weekend)")
        ax.set_ylabel("Avg Sales (€)")
        plt.xticks(rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.08, color="white")
        plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        st.markdown("""
        <div class="insight">💡 <b>Finding:</b> Sunday appears highest due to <b>selection bias</b> —
        only special stores open on Sundays in Germany. Monday is the highest ordinary trading day.</div>
        """, unsafe_allow_html=True)

    # Tab 6: Competition
    with tabs[5]:
        st.markdown('<div class="section-header">Sales vs Competitor Distance</div>', unsafe_allow_html=True)
        df2 = train_open.dropna(subset=["CompetitionDistance"]).copy()
        df2["DB"] = pd.cut(df2["CompetitionDistance"],
            bins=[0, 500, 1000, 2000, 5000, np.inf],
            labels=["<500m","500m-1km","1-2km","2-5km",">5km"])
        d_avg = df2.groupby("DB", observed=True)["Sales"].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(d_avg.index.astype(str), d_avg.values,
                      color=plt.cm.viridis(np.linspace(0.3, 0.9, len(d_avg))),
                      edgecolor="#1F2A3F", width=0.6)
        for b, v in zip(bars, d_avg.values):
            ax.text(b.get_x()+b.get_width()/2, v+50, f"€{v:,.0f}",
                    ha="center", fontsize=10, color="#E8EAF0", fontweight="600")
        ax.set_title("Avg Sales by Competitor Distance")
        ax.set_ylabel("Avg Sales (€)")
        plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        st.markdown("""
        <div class="insight">💡 <b>Counterintuitive finding:</b> Stores with competitors very close
        (<500m) often have HIGHER sales because they are in busy city centres with high foot traffic.</div>
        """, unsafe_allow_html=True)

    # Tab 7: Customers
    with tabs[6]:
        st.markdown('<div class="section-header">Customer Spending</div>', unsafe_allow_html=True)
        df_open = train_open.copy()
        df_open["SPC"] = df_open["Sales"] / df_open["Customers"]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            spc = df_open[df_open["SPC"] < 30]["SPC"]
            ax.hist(spc, bins=50, color="#3B82F6", edgecolor="#1F2A3F")
            ax.axvline(spc.median(), color="#F59E0B", lw=2,
                      label=f"Median: €{spc.median():.2f}")
            ax.set_title("Spend Per Customer")
            ax.set_xlabel("€/customer"); ax.set_ylabel("Frequency"); ax.legend()
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        with c2:
            promo_spc = df_open.groupby("Promo")["SPC"].median()
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["No Promo","Promo"], promo_spc.values,
                          color=["#374151","#3B82F6"], edgecolor="#1F2A3F", width=0.5)
            for b, v in zip(bars, promo_spc.values):
                ax.text(b.get_x()+b.get_width()/2, v+0.1, f"€{v:.2f}",
                        ha="center", fontsize=11, fontweight="600", color="#E8EAF0")
            ax.set_title("Median Spend by Promo")
            ax.set_ylabel("€ per Customer")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────
# PAGE 3 — FORECAST
# ──────────────────────────────────────────────────────────
elif "Forecast" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>🔮 Sales Forecast</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Predict daily sales for any store up to 6 weeks ahead</p>", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data not loaded"); st.stop()
    if not model_loaded:
        st.warning("⚠️ No trained model. Run `python train.py` first."); st.stop()

    cI, cO = st.columns([1, 2])
    with cI:
        st.markdown('<div class="section-header">Store Parameters</div>', unsafe_allow_html=True)
        sid   = st.selectbox("Store ID", sorted(train["Store"].unique()))
        promo = st.selectbox("Promo Running?", [1, 0],
                            format_func=lambda x: "✅ Yes" if x else "❌ No")
        weeks = st.slider("Forecast Weeks", 1, 6, 6)
        sch   = st.selectbox("School Holiday?", [0, 1],
                            format_func=lambda x: "Yes" if x else "No")

        si = store[store["Store"] == sid].iloc[0]
        st.markdown('<div class="section-header">Store Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <p>
            <b style="color:#E8EAF0">Type:</b> {si['StoreType']}<br>
            <b style="color:#E8EAF0">Assortment:</b> {si['Assortment']}<br>
            <b style="color:#E8EAF0">Competition:</b> {si['CompetitionDistance']:.0f}m away<br>
            <b style="color:#E8EAF0">Promo2 Active:</b> {'Yes' if si['Promo2'] else 'No'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔮 Generate Forecast", use_container_width=True):
            st.session_state["run_fc"]    = True
            st.session_state["fc_sid"]    = sid
            st.session_state["fc_promo"]  = promo
            st.session_state["fc_weeks"]  = weeks
            st.session_state["fc_school"] = sch

    with cO:
        if st.session_state.get("run_fc"):
            sid    = st.session_state["fc_sid"]
            p      = st.session_state["fc_promo"]
            sch    = st.session_state["fc_school"]
            n_days = st.session_state["fc_weeks"] * 7

            si = store[store["Store"] == sid].iloc[0]
            last_date = test["Date"].max()
            dates = pd.date_range(start=last_date - pd.Timedelta(days=n_days-1),
                                 periods=n_days, freq="D")

            rows = []
            for d in dates:
                rows.append({
                    "Store": sid, "Date": d, "DayOfWeek": d.dayofweek + 1,
                    "Open": 0 if d.dayofweek == 6 and si["StoreType"] != "b" else 1,
                    "Promo": p, "StateHoliday": "None", "SchoolHoliday": sch,
                    "StoreType": si["StoreType"], "Assortment": si["Assortment"],
                    "CompetitionDistance": si["CompetitionDistance"],
                    "CompetitionOpenSinceMonth": si.get("CompetitionOpenSinceMonth", 0),
                    "CompetitionOpenSinceYear":  si.get("CompetitionOpenSinceYear", 0),
                    "Promo2": si["Promo2"],
                    "Promo2SinceWeek": si.get("Promo2SinceWeek", 0),
                    "Promo2SinceYear": si.get("Promo2SinceYear", 0),
                    "PromoInterval":   si.get("PromoInterval", ""),
                })
            df_pred = pd.DataFrame(rows)

            preds = np.zeros(len(df_pred))
            open_mask = df_pred["Open"] == 1
            if open_mask.sum() > 0:
                preds[open_mask] = model.predict(df_pred[open_mask]).clip(min=0)

            st.markdown("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Forecast", f"€{preds.sum():,.0f}")
            m2.metric("Daily Average", f"€{preds[preds>0].mean():,.0f}" if (preds>0).any() else "€0")
            m3.metric("Peak Day", f"€{preds.max():,.0f}")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.fill_between(dates, preds, alpha=0.18, color="#3B82F6")
            ax.plot(dates, preds, color="#3B82F6", lw=2.5, marker="o", markersize=5)
            ax.set_title(f"Store {sid} — {n_days}-Day Sales Forecast",
                         fontsize=13, fontweight="600")
            ax.set_ylabel("Predicted Sales (€)")
            plt.xticks(rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.08, color="white")
            plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()

            df_show = pd.DataFrame({
                "Date": [d.strftime("%a %d %b %Y") for d in dates],
                "Open": ["Yes" if o else "No" for o in df_pred["Open"]],
                "Predicted Sales": [f"€{p:,.2f}" for p in preds],
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True, height=320)

            csv = pd.DataFrame({
                "Date": dates, "Store": sid,
                "Open": df_pred["Open"].values,
                "Predicted_Sales": preds.round(2)
            }).to_csv(index=False)
            st.download_button("⬇️ Download Forecast CSV", csv,
                              f"forecast_store{sid}.csv", "text/csv",
                              use_container_width=True)
        else:
            st.markdown("""
            <div class="info-card" style="margin-top:48px;text-align:center;padding:48px">
                <h3>Configure Parameters →</h3>
                <p>Select a store and forecast settings on the left,<br>
                   then click <b style="color:#3B82F6">Generate Forecast</b></p>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# PAGE 4 — MODEL ANALYSIS
# ──────────────────────────────────────────────────────────
elif "Model Analysis" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>🤖 Model Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Feature importance, metrics, and post-prediction analysis</p>", unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ No trained model. Run `python train.py` first."); st.stop()
    if not data_loaded:
        st.error("Data not loaded"); st.stop()

    st.markdown(f"""
    <div class="info-card">
        <h3>Active Model</h3>
        <p>File: <code style="color:#93C5FD">{model_name}</code></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    try:
        m_step = model.named_steps["model"]
        f_step = model.named_steps["fe"]
        X_sample = train_open.drop(columns=["Sales","Customers"]).head(200)
        X_fe = f_step.transform(X_sample)
        fn = X_fe.columns.tolist()
        imps = pd.Series(m_step.feature_importances_, index=fn).sort_values(ascending=True)

        top_n = st.slider("Top N features", 5, len(imps), 15)
        fig, ax = plt.subplots(figsize=(10, max(4, top_n*0.35)))
        col_fi = plt.cm.Blues(np.linspace(0.4, 0.95, top_n))
        imps.tail(top_n).plot(kind="barh", ax=ax, color=col_fi)
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.set_xlabel("Importance")
        for i, (v, nm) in enumerate(zip(imps.tail(top_n).values, imps.tail(top_n).index)):
            ax.text(v+0.001, i, f"{v:.4f}", va="center", fontsize=9, color="#6B7896")
        plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()

        top3 = imps.tail(3).index[::-1].tolist()
        st.markdown(f"""
        <div class="insight">💡 <b>Top 3 features:</b> {', '.join(f'<code>{x}</code>' for x in top3)}.</div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Feature importance error: {e}")

    st.markdown('<div class="section-header">Metric Reference Guide</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    metrics_info = [
        ("RMSE",  "Root Mean Squared Error", "Avg error in €. Same unit as Sales. Penalises big mistakes."),
        ("MAE",   "Mean Absolute Error",     "Avg absolute error in €. More intuitive than RMSE."),
        ("MAPE",  "Mean Absolute % Error",   "Percentage error — store-size independent."),
        ("RMSPE", "Root Mean Sq. % Error",   "Official Rossmann metric. Our chosen loss function."),
        ("R²",    "R-Squared",               "0 = useless, 1 = perfect. Variance explained."),
    ]
    for i, (name, full, desc) in enumerate(metrics_info):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"""
            <div class="info-card">
                <h3>{name} — {full}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# PAGE 5 — LSTM
# ──────────────────────────────────────────────────────────
elif "LSTM" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>🧠 LSTM Deep Learning</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Recurrent neural network for time-series sales prediction</p>", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data not loaded"); st.stop()

    st.markdown("""
    <div class="info-card">
        <h3>Steps Implemented (Task 2.6)</h3>
        <p>1. Isolate time series for one store<br>
        2. Check stationarity with ADF test<br>
        3. Apply differencing if non-stationary<br>
        4. Check ACF and PACF<br>
        5. Sliding window transformation<br>
        6. Scale data to [-1, 1]<br>
        7. Build 2-layer LSTM with dropout</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Time Series for Selected Store</div>', unsafe_allow_html=True)
    sid_lstm = st.selectbox("Store ID", sorted(train["Store"].unique())[:100])

    ts = (train[(train["Store"]==sid_lstm) & (train["Open"]==1)]
          .sort_values("Date").set_index("Date")["Sales"]
          .resample("D").mean().interpolate("linear"))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts.index, ts.values, color="#3B82F6", lw=1.2)
    ax.fill_between(ts.index, ts.values, alpha=0.1, color="#3B82F6")
    ax.set_title(f"Store {sid_lstm} — Daily Sales Time Series")
    ax.set_ylabel("Sales (€)")
    plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">ADF Stationarity Test</div>', unsafe_allow_html=True)
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(ts.dropna())
            p = result[1]
            is_stat = p < 0.05
            st.metric("ADF Statistic", f"{result[0]:.4f}")
            st.metric("p-value", f"{p:.6f}")
            if is_stat:
                st.markdown('<span class="status-pill success">STATIONARY (p < 0.05)</span>',
                           unsafe_allow_html=True)
                st.caption("No differencing needed")
            else:
                st.markdown('<span class="status-pill warning">NON-STATIONARY</span>',
                           unsafe_allow_html=True)
                st.caption("Apply first-order differencing")
        except ImportError:
            st.warning("Install statsmodels")

    with c2:
        st.markdown('<div class="section-header">LSTM Architecture</div>', unsafe_allow_html=True)
        st.code("""
Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(14, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1),
])

Optimizer: Adam | Loss: MSE
EarlyStopping(patience=10)
        """, language="python")

    st.markdown('<div class="section-header">Autocorrelation (ACF & PACF)</div>', unsafe_allow_html=True)
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(ts.dropna(), lags=40, ax=axes[0], alpha=0.05)
        plot_pacf(ts.dropna(), lags=40, ax=axes[1], alpha=0.05, method="ywm")
        axes[0].set_title("ACF")
        axes[1].set_title("PACF")
        plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()
        st.markdown("""
        <div class="insight">💡 Bars outside the blue band indicate significant autocorrelation.
        Strong values at lag 7 confirm a weekly cycle.</div>
        """, unsafe_allow_html=True)
    except ImportError:
        st.warning("statsmodels needed for ACF/PACF")


# ──────────────────────────────────────────────────────────
# PAGE 6 — BATCH PREDICT
# ──────────────────────────────────────────────────────────
elif "Batch" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>📁 Batch Predictions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Upload a CSV to predict for many stores and dates at once</p>", unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ No trained model. Run `python train.py` first."); st.stop()

    st.markdown("""
    <div class="info-card">
        <h3>Expected CSV columns</h3>
        <p><code>Store, Date, Open, Promo, StateHoliday, SchoolHoliday, ...</code><br><br>
        The easiest option: upload your <code>test.csv</code> directly.</p>
    </div>
    """, unsafe_allow_html=True)

    up = st.file_uploader("Upload CSV file", type=["csv"])
    if up:
        try:
            df_up = pd.read_csv(up, parse_dates=["Date"])
            st.success(f"Uploaded: {len(df_up):,} rows × {df_up.shape[1]} cols")
            with st.expander("Preview"):
                st.dataframe(df_up.head(), use_container_width=True, hide_index=True)

            if st.button("🔮 Run Predictions", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    df_p = df_up.copy()
                    id_col = "Id" if "Id" in df_p.columns else None
                    if "StoreType" not in df_p.columns:
                        df_p = df_p.merge(store, on="Store", how="left")
                    for col in ["Id","Customers","Sales"]:
                        if col in df_p.columns:
                            df_p = df_p.drop(columns=[col])
                    df_p["Open"] = df_p["Open"].fillna(1)
                    closed = df_p["Open"] == 0
                    preds = np.zeros(len(df_p))
                    open_df = df_p[~closed].copy()
                    if len(open_df) > 0:
                        preds[~closed] = model.predict(open_df).clip(min=0)

                    res = pd.DataFrame()
                    if id_col: res["Id"] = df_up["Id"]
                    res["Date"]  = df_up["Date"]
                    res["Store"] = df_up["Store"]
                    res["Predicted_Sales"] = preds.round(2)

                st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", f"{len(res):,}")
                m2.metric("Total", f"€{res['Predicted_Sales'].sum():,.0f}")
                m3.metric("Mean", f"€{res['Predicted_Sales'].mean():,.0f}")

                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.hist(res["Predicted_Sales"], bins=60, color="#3B82F6", edgecolor="#1F2A3F")
                ax.set_title("Predicted Sales Distribution")
                ax.set_xlabel("Predicted Sales (€)")
                plt.tight_layout(); style_chart(fig); st.pyplot(fig); plt.close()

                st.dataframe(res.head(20), use_container_width=True, hide_index=True)
                csv = res.to_csv(index=False)
                st.download_button("⬇️ Download Full CSV", csv,
                                  "batch_predictions.csv", "text/csv",
                                  use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


# ──────────────────────────────────────────────────────────
# PAGE 7 — MLFLOW
# ──────────────────────────────────────────────────────────
elif "MLflow" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>📈 MLflow Tracking</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Compare all training runs and their metrics</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>What MLflow does</h3>
        <p>Tracks every training run — parameters, metrics, and the resulting model.
        Each time you run <code>python train.py</code> a new run is created.
        Lets you compare model versions side by side.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Launch Full MLflow UI</div>', unsafe_allow_html=True)
    st.code("mlflow ui --backend-store-uri outputs/mlruns --port 5000", language="bash")
    st.caption("Open http://localhost:5000 after running this command")
    mlruns_db = "outputs/mlruns.db"
    if os.path.exists(mlruns_db):
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{mlruns_db}")
            client = mlflow.tracking.MlflowClient()
            try:
                exp = client.get_experiment_by_name("rossmann-sales-forecasting")
                if exp:
                    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
                    if runs:
                        st.markdown('<div class="section-header">Recent Runs</div>', unsafe_allow_html=True)
                        rows = []
                        for r in runs[:15]:
                            rows.append({
                                "Run": r.data.tags.get("mlflow.runName", "unnamed"),
                                "Model": r.data.params.get("model_type", "—"),
                                "RMSPE": f"{r.data.metrics.get('RMSPE', 0):.2f}%" if r.data.metrics.get('RMSPE') else "—",
                                "R²": f"{r.data.metrics.get('R2', 0):.4f}" if r.data.metrics.get('R2') else "—",
                                "Started": datetime.datetime.fromtimestamp(r.info.start_time/1000).strftime("%Y-%m-%d %H:%M"),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No runs yet — run `python train.py`")
                else:
                    st.info("Experiment not created — run `python train.py`")
            except Exception as e:
                st.warning(f"Query failed: {e}")
        except ImportError:
            st.warning("Install MLflow")
    else:
        st.info("No MLflow data yet")


# ──────────────────────────────────────────────────────────
# PAGE 8 — INSIGHTS
# ──────────────────────────────────────────────────────────
elif "Insights" in page:
    st.markdown("<h2 style='color:#E8EAF0;font-weight:700'>💡 Business Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA8C2'>Key findings and actionable recommendations</p>", unsafe_allow_html=True)

    insights = [
        ("📈", "Promotions deliver 38.8% sales lift",
         "Promotional days outperform regular days by ~€1,900 per store per day. "
         "Lift varies dramatically — some stores see negative impact. "
         "Recommendation: A/B test promo removal for bottom 50 stores."),
        ("🎄", "Christmas drives every annual peak",
         "December spikes consistently every year. Model leverages DaysToChristmas "
         "as a top feature. Recommendation: ensure inventory and staffing for the "
         "two weeks before Dec 25."),
        ("👥", "Customers explain 68% of sales variance",
         "r = 0.83 between Customers and Sales. Foot traffic is the strongest predictor. "
         "Recommendation: invest in location intelligence and storefront optimisation."),
        ("🏪", "Store Type b leads by ~50%",
         "Type b stores earn dramatically more despite being rarest. "
         "Recommendation: study what makes Type b successful and replicate."),
        ("📍", "Closer competitors = HIGHER sales",
         "Counterintuitively, stores with competitors <500m away outperform isolated ones. "
         "Dense urban locations have natural high foot traffic. "
         "Distance is a proxy for location quality, not threat."),
        ("🛒", "Extended assortment beats Basic by 40%",
         "Stores with broader product range earn significantly more. "
         "Recommendation: expand assortment in underperforming Basic stores."),
    ]

    for emoji, title, desc in insights:
        st.markdown(f"""
        <div class="info-card">
            <h3>{emoji} &nbsp; {title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Production Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3>Next Steps to Improve the Model</h3>
        <p>
        • <b>Retrain weekly</b> with most recent 90 days<br>
        • <b>Add weather data</b> — temperature and rain affect foot traffic<br>
        • <b>Try XGBoost</b> — typically beats GBM by 2-5% RMSPE<br>
        • <b>Hyperparameter tuning</b> with RandomizedSearchCV<br>
        • <b>Deploy per-store LSTMs</b> for high-revenue stores<br>
        • <b>A/B test</b> against manager forecasts to measure lift
        </p>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="margin-top:60px;padding:24px 0;border-top:1px solid #1F2A3F;text-align:center">
    <div style="color:#6B7896;font-size:12px;line-height:1.8">
        <b style="color:#9CA8C2">Rossmann Sales Forecasting Dashboard</b><br>
        Built for NextHikes IT Solutions · Production-grade ML pipeline
    </div>
</div>
""", unsafe_allow_html=True)
