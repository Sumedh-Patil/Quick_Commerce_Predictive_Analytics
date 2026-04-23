"""
utils.py — Visualization & Helper Utilities for Quick Commerce Analytics
========================================================================
All charts use Plotly with a consistent dark theme matching the app design.
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#4f8fff",
    "secondary": "#7c3aed",
    "accent": "#f59e0b",
    "success": "#10b981",
    "danger": "#ef4444",
    "info": "#06b6d4",
    "bg_dark": "#0a0e27",
    "bg_card": "#111633",
    "bg_surface": "#1a1f45",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "border": "rgba(79, 143, 255, 0.15)",
    "gradient_1": "#4f8fff",
    "gradient_2": "#7c3aed",
    "gradient_3": "#06b6d4",
}

CHART_COLORS = [
    "#4f8fff", "#7c3aed", "#f59e0b", "#10b981", "#ef4444",
    "#06b6d4", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
    "#6366f1", "#84cc16",
]

PLOTLY_TEMPLATE = "plotly_dark"


def _base_layout(fig, title="", height=450):
    """Apply consistent styling to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"]), x=0.0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_muted"], size=12),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text"]),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(79, 143, 255, 0.08)",
        zerolinecolor="rgba(79, 143, 255, 0.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(79, 143, 255, 0.08)",
        zerolinecolor="rgba(79, 143, 255, 0.15)",
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Delivery Time Distribution (Histogram)
# ---------------------------------------------------------------------------
def plot_delivery_time_distribution(df):
    """Overlapping histogram of Delivery_Time_Min by Fast_Delivery class."""
    fig = go.Figure()
    for val, name, color in [(0, "Standard", COLORS["accent"]), (1, "Fast", COLORS["success"])]:
        data = df[df["Fast_Delivery"] == val]["Delivery_Time_Min"]
        fig.add_trace(go.Histogram(
            x=data, name=name, opacity=0.65,
            marker_color=color, nbinsx=40,
        ))
    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "Delivery Time Distribution by Delivery Type")


# ---------------------------------------------------------------------------
# 2. Distance vs Fast Delivery (Box Plot)
# ---------------------------------------------------------------------------
def plot_distance_boxplot(df):
    """Box plot of Distance_Km grouped by Fast_Delivery."""
    fig = px.box(
        df.sample(min(50000, len(df)), random_state=42),
        x="Fast_Delivery", y="Distance_Km",
        color="Fast_Delivery",
        color_discrete_map={0: COLORS["accent"], 1: COLORS["success"]},
        labels={"Fast_Delivery": "Delivery Type", "Distance_Km": "Distance (km)"},
        category_orders={"Fast_Delivery": [0, 1]},
    )
    fig.update_traces(boxmean="sd")
    return _base_layout(fig, "Distance Distribution by Delivery Type")


# ---------------------------------------------------------------------------
# 3. Distance vs Delivery Time (Scatter + Trendline)
# ---------------------------------------------------------------------------
def plot_distance_vs_time(df):
    """Scatter plot with trendline — Distance vs Delivery Time."""
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Distance_Km", y="Delivery_Time_Min",
        color="Fast_Delivery",
        color_discrete_map={0: COLORS["accent"], 1: COLORS["success"]},
        opacity=0.4,
        trendline="ols",
        labels={"Distance_Km": "Distance (km)", "Delivery_Time_Min": "Delivery Time (min)"},
    )
    return _base_layout(fig, "Distance vs Delivery Time")


# ---------------------------------------------------------------------------
# 4. Correlation Heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df):
    """Correlation heatmap of all numeric columns."""
    numeric_df = df.select_dtypes(include=np.number)
    if "Order_ID" in numeric_df.columns:
        numeric_df = numeric_df.drop("Order_ID", axis=1)
    corr = numeric_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[
            [0, "#7c3aed"],
            [0.25, "#1a1f45"],
            [0.5, "#0a0e27"],
            [0.75, "#1a3a5c"],
            [1, "#4f8fff"],
        ],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    return _base_layout(fig, "Feature Correlation Matrix", height=550)


# ---------------------------------------------------------------------------
# 5. Density Distribution (KDE approximation with histogram)
# ---------------------------------------------------------------------------
def plot_delivery_time_density(df):
    """KDE-like density plot using Plotly histogram with marginal."""
    sample = df.sample(min(50000, len(df)), random_state=42)
    fig = go.Figure()
    for val, name, color in [(0, "Standard Delivery", COLORS["accent"]), (1, "Fast Delivery", COLORS["success"])]:
        data = sample[sample["Fast_Delivery"] == val]["Delivery_Time_Min"]
        fig.add_trace(go.Histogram(
            x=data, name=name, opacity=0.5,
            marker_color=color, nbinsx=60,
            histnorm="probability density",
        ))
    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "Density Distribution of Delivery Time")


# ---------------------------------------------------------------------------
# 6. Fast Delivery Rate by City
# ---------------------------------------------------------------------------
def plot_city_delivery_rate(df):
    """Horizontal bar chart — Fast Delivery rate per city."""
    city_rate = df.groupby("City")["Fast_Delivery"].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=city_rate.values,
        y=city_rate.index,
        orientation="h",
        marker=dict(
            color=city_rate.values,
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
            line=dict(width=0),
        ),
        text=[f"{v:.1%}" for v in city_rate.values],
        textposition="auto",
        textfont=dict(color="white"),
    ))
    return _base_layout(fig, "Fast Delivery Rate by City", height=500)


# ---------------------------------------------------------------------------
# 7. Fast Delivery Rate by Company
# ---------------------------------------------------------------------------
def plot_company_delivery_rate(df):
    """Horizontal bar chart — Fast Delivery rate per company."""
    company_rate = df.groupby("Company")["Fast_Delivery"].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=company_rate.values,
        y=company_rate.index,
        orientation="h",
        marker=dict(
            color=company_rate.values,
            colorscale=[[0, COLORS["accent"]], [1, COLORS["success"]]],
            line=dict(width=0),
        ),
        text=[f"{v:.1%}" for v in company_rate.values],
        textposition="auto",
        textfont=dict(color="white"),
    ))
    return _base_layout(fig, "Fast Delivery Rate by Company", height=500)


# ---------------------------------------------------------------------------
# 8. Product Category Distribution
# ---------------------------------------------------------------------------
def plot_category_distribution(df):
    """Donut chart of order count by Product Category."""
    cat_counts = df["Product_Category"].value_counts()
    fig = go.Figure(go.Pie(
        labels=cat_counts.index,
        values=cat_counts.values,
        hole=0.5,
        marker=dict(colors=CHART_COLORS[:len(cat_counts)]),
        textinfo="label+percent",
        textfont=dict(color="white", size=12),
    ))
    return _base_layout(fig, "Order Distribution by Product Category")


# ---------------------------------------------------------------------------
# 9. Payment Method Distribution
# ---------------------------------------------------------------------------
def plot_payment_distribution(df):
    """Donut chart of order count by Payment Method."""
    pay_counts = df["Payment_Method"].value_counts()
    fig = go.Figure(go.Pie(
        labels=pay_counts.index,
        values=pay_counts.values,
        hole=0.5,
        marker=dict(colors=CHART_COLORS[:len(pay_counts)]),
        textinfo="label+percent",
        textfont=dict(color="white", size=12),
    ))
    return _base_layout(fig, "Payment Method Distribution")


# ---------------------------------------------------------------------------
# 10. Feature Importance (from trained model)
# ---------------------------------------------------------------------------
def plot_feature_importance(importances, top_n=15):
    """Horizontal bar chart of top-N feature importances."""
    top = importances.head(top_n).sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale=[[0, COLORS["secondary"]], [0.5, COLORS["primary"]], [1, COLORS["info"]]],
            line=dict(width=0),
        ),
        text=[f"{v:.4f}" for v in top.values],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    return _base_layout(fig, f"Top {top_n} Feature Importances (Random Forest)", height=500)


# ---------------------------------------------------------------------------
# 11. Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm):
    """Annotated heatmap for confusion matrix."""
    labels = ["Standard (0)", "Fast (1)"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#111633"], [1, COLORS["primary"]]],
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    return _base_layout(fig, "Confusion Matrix", height=400)


# ---------------------------------------------------------------------------
# 12. ROC Curve
# ---------------------------------------------------------------------------
def plot_roc_curve(fpr, tpr, auc_score):
    """ROC curve with AUC shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC = {auc_score:.4f})",
        line=dict(color=COLORS["primary"], width=3),
        fill="tozeroy",
        fillcolor="rgba(79, 143, 255, 0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random",
        line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return _base_layout(fig, "ROC Curve")


# ---------------------------------------------------------------------------
# 13. Customer Age Distribution
# ---------------------------------------------------------------------------
def plot_age_distribution(df):
    """Histogram of Customer_Age split by Fast_Delivery."""
    fig = go.Figure()
    for val, name, color in [(0, "Standard", COLORS["accent"]), (1, "Fast", COLORS["success"])]:
        data = df[df["Fast_Delivery"] == val]["Customer_Age"]
        fig.add_trace(go.Histogram(
            x=data, name=name, opacity=0.6,
            marker_color=color, nbinsx=30,
        ))
    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "Customer Age Distribution by Delivery Type")


# ---------------------------------------------------------------------------
# 14. Order Value Distribution
# ---------------------------------------------------------------------------
def plot_order_value_distribution(df):
    """Histogram of Order_Value split by Fast_Delivery."""
    fig = go.Figure()
    for val, name, color in [(0, "Standard", COLORS["accent"]), (1, "Fast", COLORS["success"])]:
        data = df[df["Fast_Delivery"] == val]["Order_Value"]
        fig.add_trace(go.Histogram(
            x=data, name=name, opacity=0.6,
            marker_color=color, nbinsx=40,
        ))
    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "Order Value Distribution by Delivery Type")


# ---------------------------------------------------------------------------
# 15. Prediction Confidence Gauge
# ---------------------------------------------------------------------------
def plot_prediction_gauge(probability):
    """Gauge chart showing prediction confidence."""
    color = COLORS["success"] if probability >= 0.5 else COLORS["accent"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=40, color=color)),
        title=dict(
            text="Fast Delivery Probability",
            font=dict(size=16, color=COLORS["text"]),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], dtick=20, tickfont=dict(color=COLORS["text_muted"])),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="rgba(239, 68, 68, 0.15)"),
                dict(range=[30, 60], color="rgba(245, 158, 11, 0.15)"),
                dict(range=[60, 100], color="rgba(16, 185, 129, 0.15)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.8,
                value=probability * 100,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig
