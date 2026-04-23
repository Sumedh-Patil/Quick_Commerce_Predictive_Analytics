"""
Quick Commerce Predictive Analytics Dashboard
==============================================
Premium Streamlit application for Fast Delivery prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_data, get_data_summary, train_model, load_trained_model, predict_single
from utils import (
    plot_delivery_time_distribution, plot_distance_boxplot,
    plot_distance_vs_time, plot_correlation_heatmap,
    plot_delivery_time_density, plot_city_delivery_rate,
    plot_company_delivery_rate, plot_category_distribution,
    plot_payment_distribution, plot_feature_importance,
    plot_confusion_matrix, plot_roc_curve,
    plot_order_value_distribution, plot_prediction_gauge, COLORS,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quick Commerce Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #111633 40%, #0d1117 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
/* Preserve Streamlit's Material Icons */
span[data-testid="stIconMaterial"],
.material-symbols-rounded,
[class*="Icon"] span,
[data-baseweb] span[aria-hidden="true"] {
    font-family: 'Material Symbols Rounded' !important;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1230 0%, #0a0e27 100%);
    border-right: 1px solid rgba(79,143,255,0.12);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(79,143,255,0.1);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(17,22,51,0.9), rgba(26,31,69,0.7));
    border: 1px solid rgba(79,143,255,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(79,143,255,0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8fff, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0 0.25rem;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.metric-icon { font-size: 1.8rem; }

/* Section Header */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(79,143,255,0.2);
    background: linear-gradient(135deg, #4f8fff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Glass Card */
.glass-card {
    background: linear-gradient(135deg, rgba(17,22,51,0.85), rgba(26,31,69,0.6));
    border: 1px solid rgba(79,143,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, rgba(79,143,255,0.12), rgba(124,58,237,0.08));
    border: 1px solid rgba(79,143,255,0.15);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8fff, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* Prediction Result */
.pred-fast {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.pred-std {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(17,22,51,0.6);
    border-radius: 8px;
    color: #94a3b8;
    border: 1px solid rgba(79,143,255,0.1);
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(79,143,255,0.15) !important;
    color: #4f8fff !important;
    border-color: rgba(79,143,255,0.3) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f8fff, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79,143,255,0.35);
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Divider */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,143,255,0.3), transparent);
    margin: 1.5rem 0;
}

/* Selectbox / Input */
.stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
    background: rgba(17,22,51,0.6) !important;
    border-color: rgba(79,143,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def cached_train_model(sample_size):
    df = cached_load_data()
    return train_model(df, sample_size=sample_size)


# ── Helper: Metric Card ─────────────────────────────────────────────────────
def metric_card(icon, label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <div style="font-size:2.5rem;">⚡</div>
        <div style="font-size:1.3rem; font-weight:800;
             background: linear-gradient(135deg, #4f8fff, #7c3aed);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Quick Commerce
        </div>
        <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:2px;
             text-transform:uppercase; margin-top:2px;">
            Predictive Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Data Explorer", "📈 Visualizations",
         "🔮 Predict"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Quick stats in sidebar
    with st.spinner("Loading dataset..."):
        df = cached_load_data()
    summary = get_data_summary(df)

    st.markdown(f"""
    <div class="glass-card" style="padding:1rem; font-size:0.82rem;">
        <div style="color:#4f8fff; font-weight:700; margin-bottom:0.6rem;
             font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">
            Dataset Overview
        </div>
        <div style="display:flex; justify-content:space-between; margin:4px 0;">
            <span style="color:#94a3b8;">Orders</span>
            <span style="color:#e2e8f0; font-weight:600;">{summary['total_orders']:,}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin:4px 0;">
            <span style="color:#94a3b8;">Companies</span>
            <span style="color:#e2e8f0; font-weight:600;">{summary['total_companies']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin:4px 0;">
            <span style="color:#94a3b8;">Cities</span>
            <span style="color:#e2e8f0; font-weight:600;">{summary['total_cities']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin:4px 0;">
            <span style="color:#94a3b8;">Fast Delivery %</span>
            <span style="color:#10b981; font-weight:600;">{summary['fast_delivery_pct']}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding-top:1rem; font-size:0.65rem; color:#475569;">
        Built by Sumedh · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚡ Quick Commerce Analytics</div>
        <p class="hero-sub">
            Predictive analytics dashboard for fast delivery optimization
            across India's leading quick commerce platforms
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("📦", "Total Orders", f"{summary['total_orders']:,}")
    with c2: metric_card("⚡", "Fast Delivery", f"{summary['fast_delivery_pct']}%")
    with c3: metric_card("⏱️", "Avg Time", f"{summary['avg_delivery_time']} min")
    with c4: metric_card("📍", "Avg Distance", f"{summary['avg_distance']} km")
    with c5: metric_card("💰", "Avg Order", f"₹{summary['avg_order_value']}")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_city_delivery_rate(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_company_delivery_rate(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_category_distribution(df), use_container_width=True)
    with col4:
        st.plotly_chart(plot_payment_distribution(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Explorer":
    st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📊 Statistics", "ℹ️ Column Info"])

    with tab1:
        n_rows = st.slider("Rows to display", 10, 500, 100)
        st.dataframe(df.head(n_rows), use_container_width=True, height=450)

    with tab2:
        st.markdown("**Numerical Feature Statistics**")
        st.dataframe(df.describe().T.round(2), use_container_width=True)

        st.markdown("**Categorical Feature Counts**")
        cat_col = st.selectbox("Select column", ["Company", "City", "Product_Category", "Payment_Method"])
        st.dataframe(df[cat_col].value_counts().reset_index().rename(
            columns={"index": cat_col, "count": "Count"}
        ), use_container_width=True)

    with tab3:
        info_data = []
        for col in df.columns:
            info_data.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": df[col].notna().sum(),
                "Null": df[col].isna().sum(),
                "Unique": df[col].nunique(),
            })
        st.dataframe(pd.DataFrame(info_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown('<div class="section-header">📈 Visual Analytics</div>', unsafe_allow_html=True)

    viz_tab = st.tabs(["Distribution", "Relationships", "Correlation"])

    with viz_tab[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_delivery_time_distribution(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_delivery_time_density(df), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_order_value_distribution(df), use_container_width=True)
        with c4:
            st.plotly_chart(plot_distance_boxplot(df), use_container_width=True)

    with viz_tab[1]:
        st.plotly_chart(plot_distance_vs_time(df), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_city_delivery_rate(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_company_delivery_rate(df), use_container_width=True)

    with viz_tab[2]:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<div class="section-header">🔮 Fast Delivery Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="padding:1rem 1.5rem; font-size:0.9rem; color:#94a3b8;">
        Adjust the order details below — the prediction updates
        <span style="color:#4f8fff; font-weight:600;">in real-time</span> as you change any value.
    </div>
    """, unsafe_allow_html=True)

    # Ensure model is trained
    with st.spinner("Preparing model..."):
        results = cached_train_model(sample_size=100000)
    model_obj = results["model"]
    scaler_obj = results["scaler"]
    columns_obj = results["columns"]

    # ── Input widgets (only the relevant fields) ─────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        company = st.selectbox("Company", sorted(df["Company"].unique()), key="pred_company")
    with r1c2:
        city = st.selectbox("City", sorted(df["City"].unique()), key="pred_city")
    with r1c3:
        category = st.selectbox("Product Category", sorted(df["Product_Category"].unique()), key="pred_cat")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        order_val = st.number_input("Order Value (₹)", 50, 5000, 500, key="pred_oval")
    with r2c2:
        delivery_time = st.number_input("Delivery Time (min)", 5, 120, 30, key="pred_dt")
    with r2c3:
        distance = st.number_input("Distance (km)", 0.5, 30.0, 5.0, step=0.5, key="pred_dist")

    # ── Real-time prediction (hidden fields use median defaults) ──────────────
    input_dict = {
        "Company": company,
        "City": city,
        "Customer_Age": 35,
        "Order_Value": order_val,
        "Delivery_Time_Min": delivery_time,
        "Distance_Km": distance,
        "Items_Count": 3,
        "Product_Category": category,
        "Payment_Method": "UPI",
        "Customer_Rating": 3,
        "Discount_Applied": 0,
        "Delivery_Partner_Rating": 4,
    }

    prediction, probability = predict_single(input_dict, model_obj, scaler_obj, columns_obj)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="pred-fast">
                <div style="font-size:3rem;">⚡</div>
                <div style="font-size:1.6rem; font-weight:800; color:#10b981;
                     margin:0.5rem 0;">FAST DELIVERY</div>
                <div style="color:#94a3b8;">
                    Confidence: <span style="color:#10b981; font-weight:700;">
                    {probability*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-std">
                <div style="font-size:3rem;">📦</div>
                <div style="font-size:1.6rem; font-weight:800; color:#f59e0b;
                     margin:0.5rem 0;">STANDARD DELIVERY</div>
                <div style="color:#94a3b8;">
                    Fast Delivery Probability: <span style="color:#f59e0b; font-weight:700;">
                    {probability*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.plotly_chart(plot_prediction_gauge(probability), use_container_width=True)
