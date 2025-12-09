"""
Streamlit app to load saved encoders/scaler/models and predict salary >50K.
Run: streamlit run src/predict_streamlit.py
"""
import streamlit as st
import joblib
import numpy as np
import os
import json
import pandas as pd
from models import LogisticRegressionScratch, GaussianNaiveBayes, KNearestNeighbors, DecisionTree

SAVED = os.path.join(os.path.dirname(__file__), '..', 'saved_models')


# -----------------------------------------------------------
# Load all artifacts
# -----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    enc = joblib.load(os.path.join(SAVED, "encoders.joblib"))
    scaler = joblib.load(os.path.join(SAVED, "scaler.joblib"))

    # Load feature column order
    feature_cols = json.load(open(os.path.join(SAVED, "feature_columns.json")))

    # Logistic
    log_w = np.load(os.path.join(SAVED, "logistic.npy"))
    log = LogisticRegressionScratch()
    log.w = log_w

    # Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.load(os.path.join(SAVED, "gnb_params.npz"))

    # KNN
    knn = KNearestNeighbors()
    knn.load(os.path.join(SAVED, "knn_store.npz"))

    # Decision Tree
    dt = DecisionTree()
    dt.load(os.path.join(SAVED, "dtree.pkl"))

    return enc, scaler, feature_cols, log, gnb, knn, dt


# -----------------------------------------------------------
# Build a dataframe from raw UI inputs (unordered)
# -----------------------------------------------------------
def build_input_features(values, encoders):
    numeric_cols = [
        "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
    ]
    cat_cols = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]

    row = {}

    # Numeric first
    for col in numeric_cols:
        row[col] = float(values[col])

    # Encoded categorical
    for col in cat_cols:
        top_values = encoders[col]     # top categories used in training
        selected = values[col]

        # Create 1/0 columns for each known category
        for t in top_values:
            row[f"{col}_{t}"] = 1 if selected == t else 0

        # "Other" bucket
        row[f"{col}_Other"] = 1 if selected not in top_values else 0

    return pd.DataFrame([row])


# -----------------------------------------------------------
# Align columns to match training EXACTLY
# -----------------------------------------------------------
def align_columns(df_input, feature_cols):
    # Add missing columns
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Drop unexpected columns
    df_input = df_input[feature_cols]

    return df_input


# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
def main():
    st.title("Salary >50K Prediction (Adult Dataset)")

    enc, scaler, feature_cols, log, gnb, knn, dt = load_artifacts()

    st.sidebar.header("Input Features")

    # Numeric inputs
    age = st.sidebar.number_input("age", min_value=17, max_value=100, value=30)
    education_num = st.sidebar.number_input("education_num", min_value=1, max_value=16, value=10)
    capital_gain = st.sidebar.number_input("capital_gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("capital_loss", min_value=0, value=0)
    hours_per_week = st.sidebar.number_input("hours_per_week", min_value=1, max_value=99, value=40)

    # Categorical inputs
    selections = {}
    cat_cols = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]

    for col in cat_cols:
        options = enc[col] + ["Other"]
        selections[col] = st.sidebar.selectbox(col, options, index=0)

    # Merge final values
    values = {
        "age": age,
        "education_num": education_num,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
    }
    values.update(selections)

    # Build dataframe + align to training features
    df_input = build_input_features(values, enc)
    df_input = align_columns(df_input, feature_cols)

    # Scale
    Xs = scaler.transform(df_input)

    st.subheader("Predictions")

    # ----------------------------
    # PREDICT BUTTON
    # ----------------------------
    if st.button("Predict"):

        # Logistic Regression
        prob_log = log.predict_proba(Xs)[0]
        pred_log = int(prob_log >= 0.5)
        st.write(f"**Logistic Regression → Prob: {prob_log:.3f} → {'>50K' if pred_log else '<=50K'}**")

        # Gaussian Naive Bayes
        prob_gnb = (
            gnb.predict_proba(Xs)[0]
            if hasattr(gnb, "predict_proba")
            else gnb.predict(Xs)[0]
        )
        pred_gnb = int(prob_gnb >= 0.5)
        st.write(f"**Gaussian Naive Bayes → Prob: {prob_gnb:.3f} → {'>50K' if pred_gnb else '<=50K'}**")

        # KNN
        pred_knn = knn.predict(Xs)[0]
        st.write(f"**KNN → Prediction: {'>50K' if pred_knn == 1 else '<=50K'}**")

        # Decision Tree
        pred_dt = dt.predict(Xs)[0]
        st.write(f"**Decision Tree → Prediction: {'>50K' if pred_dt == 1 else '<=50K'}**")


if __name__ == "__main__":
    main()
