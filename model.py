"""
model.py — Machine Learning Pipeline for Quick Commerce Fast Delivery Prediction
================================================================================
Handles data loading, preprocessing, model training, evaluation, and persistence.
Uses RandomForestClassifier matching the original analysis parameters.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "quick_commerce_with_target.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")


# ---------------------------------------------------------------------------
# Data Loading & Cleaning
# ---------------------------------------------------------------------------
def load_data():
    """Load the Quick Commerce dataset and return a clean DataFrame."""
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    return df


def get_data_summary(df):
    """Return a dictionary of summary statistics about the dataset."""
    return {
        "total_orders": len(df),
        "total_companies": df["Company"].nunique(),
        "total_cities": df["City"].nunique(),
        "total_categories": df["Product_Category"].nunique(),
        "fast_delivery_pct": round(df["Fast_Delivery"].mean() * 100, 1),
        "avg_delivery_time": round(df["Delivery_Time_Min"].mean(), 1),
        "avg_distance": round(df["Distance_Km"].mean(), 2),
        "avg_order_value": round(df["Order_Value"].mean(), 2),
        "avg_customer_rating": round(df["Customer_Rating"].mean(), 2),
        "avg_partner_rating": round(df["Delivery_Partner_Rating"].mean(), 2),
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df):
    """
    Preprocess the DataFrame for model training:
    - Drop Order_ID
    - One-hot encode categorical columns
    - Split into X and y
    """
    df_proc = df.copy()
    if "Order_ID" in df_proc.columns:
        df_proc.drop("Order_ID", axis=1, inplace=True)

    # One-hot encode
    df_proc = pd.get_dummies(df_proc, drop_first=True)

    # Separate features and target
    X = df_proc.drop("Fast_Delivery", axis=1)
    y = df_proc["Fast_Delivery"]

    return X, y


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_model(df, sample_size=None):
    """
    Full training pipeline:
    1. Preprocess
    2. Train/test split (80/20, stratified)
    3. Scale features
    4. Train RandomForestClassifier
    5. Evaluate
    6. Persist model, scaler, column names

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    sample_size : int or None
        If provided, sample the dataset to this size before training (for speed).

    Returns
    -------
    results : dict
        Contains model, scaler, metrics, feature importances, etc.
    """
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    X, y = preprocess(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest (matching original notebook parameters)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Persist
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)

    return {
        "model": rf,
        "scaler": scaler,
        "columns": list(X.columns),
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": importances,
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_trained_model():
    """Load a previously saved model, scaler, and column list."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]):
        return None
    return {
        "model": joblib.load(MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "columns": joblib.load(COLUMNS_PATH),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_single(input_dict, model, scaler, columns):
    """
    Make a prediction for a single order.

    Parameters
    ----------
    input_dict : dict
        Raw feature values (before encoding).
    model : fitted sklearn model
    scaler : fitted StandardScaler
    columns : list of str
        Column names the model was trained on.

    Returns
    -------
    prediction : int (0 or 1)
    probability : float (probability of Fast_Delivery = 1)
    """
    # Separate numeric and categorical fields
    categorical_cols = ["Company", "City", "Product_Category", "Payment_Method"]

    # Start with a zero-filled row matching the training columns
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Fill in numeric features directly
    for key, value in input_dict.items():
        if key not in categorical_cols and key in columns:
            row[key] = value

    # Manually encode categorical features to match get_dummies(drop_first=True)
    for key in categorical_cols:
        if key in input_dict:
            dummy_col = f"{key}_{input_dict[key]}"
            if dummy_col in columns:
                row[dummy_col] = 1

    # Scale
    input_scaled = scaler.transform(row)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return int(prediction), float(probability)
