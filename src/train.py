"""
Train script: loads processed data, constructs feature matrix (simple encoding),
applies SMOTE, trains 4 models from scratch, evaluates, and saves model parameters.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from utils import (train_test_split, accuracy, roc_auc_score_simple,
                   sigmoid, confusion_matrix, precision_recall_f1)
from models import (LogisticRegressionScratch, KNearestNeighbors,
                    GaussianNaiveBayes, DecisionTree)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(BASE_DIR, "..", "data", "processed")
SAVED = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED, exist_ok=True)
REPORTS = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORTS, exist_ok=True)


# ------------------------------------------------------------------
# SMOTE visualization helpers (replaces the old Smote.py)
# ------------------------------------------------------------------
def print_distribution(y, title="Class Distribution"):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n{title}")
    print("-" * 50)
    for val, cnt in zip(unique, counts):
        print(f"Class {int(val)}: {cnt} ({cnt/total*100:5.2f}%)")
    print("-" * 50)


def plot_distribution(y_before, y_after, technique="SMOTE"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    unique, counts = np.unique(y_before, return_counts=True)
    plt.bar([str(int(u)) for u in unique], counts, color=['#1f77b4', '#ff7f0e'])
    plt.title("Before " + technique)
    plt.xlabel("Salary (0 = ≤50K, 1 = >50K)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    unique, counts = np.unique(y_after, return_counts=True)
    plt.bar([str(int(u)) for u in unique], counts, color=['#2ca02c', '#d62728'])
    plt.title("After " + technique)
    plt.xlabel("Salary (0 = ≤50K, 1 = >50K)")

    plt.suptitle(f"{technique} - Class Distribution Comparison")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt_path = os.path.join(REPORTS, "smote_comparison.png")
    plt.savefig(plt_path, dpi=200)
    plt.close()
    print(f"SMOTE comparison plot saved → {plt_path}")


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
def load_processed():
    train_path = os.path.join(DATA_PROCESSED, "train_processed.csv")
    test_path  = os.path.join(DATA_PROCESSED, "test_processed.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


# ------------------------------------------------------------------
# Feature engineering (one-hot with top-k + Other)
# ------------------------------------------------------------------
def build_features(df, fit_encoders=None):
    numeric_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'native_country']

    df = df.copy()
    out = df[numeric_cols].astype(float).copy()

    encoders = fit_encoders or {}
    for c in cat_cols:
        topk = 10
        if fit_encoders and c in encoders:
            top = encoders[c]
        else:
            top = df[c].value_counts().nlargest(topk).index.tolist()
        encoders[c] = top

        for val in top:
            out[f"{c}__{val}"] = (df[c] == val).astype(int)
        out[f"{c}__Other"] = (~df[c].isin(top)).astype(int)

    # Save column order for inference
    feature_cols_path = os.path.join(SAVED, "feature_columns.json")
    json.dump(list(out.columns), open(feature_cols_path, "w"))
    return out.values, encoders


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------
def evaluate_model(clf, X_test, y_test, model_name="model"):
    y_pred = clf.predict(X_test)

    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_test)[:, 1] if y_pred.ndim > 1 else clf.predict_proba(X_test)
    else:
        y_score = y_pred

    acc = accuracy(y_test, y_pred)
    prec, rec, f1 = precision_recall_f1(y_test, y_pred)
    auc = roc_auc_score_simple(y_test, y_score)

    print(f"--- {model_name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    train_df, test_df = load_processed()

    # Target
    y_all = train_df['salary'].values
    X_all, encoders = build_features(train_df)

    # Train/validation split (before SMOTE)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # ---- SMOTE balancing (only on training set) ----
    print("\n" + "="*60)
    print("DATASET BALANCING WITH SMOTE")
    print("="*60)

    print_distribution(y_train, "BEFORE SMOTE - Training set")
    y_train_before = y_train.copy()

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print_distribution(y_train, "AFTER SMOTE - Training set")
    print("="*60 + "\n")

    # Visualize the balancing
    plot_distribution(y_train_before, y_train, technique="SMOTE")

    # ---- Scaling (fit on balanced training data) ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Save scaler & encoders for deployment
    joblib.dump(encoders, os.path.join(SAVED, "encoders.joblib"))
    joblib.dump(scaler,   os.path.join(SAVED, "scaler.joblib"))

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60 + "\n")

    # 1. Logistic Regression from scratch
    logreg = LogisticRegressionScratch(lr=0.5, n_iter=2000, l2=0.01, verbose=False)
    logreg.fit(X_train, y_train)
    np.save(os.path.join(SAVED, "logistic.npy"), logreg.w)
    evaluate_model(logreg, X_val, y_val, "LogisticRegression (Scratch)")

    # 2. Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    feature_types = ['num'] * X_train.shape[1]
    gnb.fit(X_train, y_train, feature_types)
    gnb.save(os.path.join(SAVED, "gnb_params.npz"))
    evaluate_model(gnb, X_val, y_val, "GaussianNB")

    # 3. KNN
    knn = KNearestNeighbors(k=7)
    knn.fit(X_train, y_train)
    knn.save(os.path.join(SAVED, "knn_store.npz"))
    evaluate_model(knn, X_val, y_val, "K-Nearest Neighbors (k=7)")

    # 4. Decision Tree
    dt = DecisionTree(max_depth=6, min_samples_split=20)
    dt.fit(X_train, y_train)
    dt.save(os.path.join(SAVED, "dtree.pkl"))
    evaluate_model(dt, X_val, y_val, "Decision Tree")

    print("\n" + "="*60)
    print("Training complete! All models & artifacts saved in ./saved_models/")
    print("="*60)


if __name__ == "__main__":
    main()