import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---- Load models and preprocessing ----
svm = joblib.load("svm.pkl")          # SVC
lr = joblib.load("lr_model.pkl")      # LogisticRegression
rf = joblib.load("rf_model.pkl")      # RandomForestClassifier
knn = joblib.load("knn_model.pkl")    # KNeighborsClassifier

scaler = joblib.load("scaler.pkl")    # StandardScaler

models = [svm, lr, rf, knn]
model_names = ["SVM", "Logistic Regression", "Random Forest", "kNN"]

# Accuracy-based weights
weights = [0.64, 0.6808, 0.5925, 0.6287]  # match models list order

# Manual label mapping
label_names = ['Home Win', 'Draw', 'Away Win']  # 0,1,2

feature_names = [
    'Half Time Home Goals',
    'Half Time Away Goals',
    'Home Shots on Target',
    'Away Shots on Target'
]

st.title("Football Match Outcome Predictor")

# ---- User Inputs ----
inputs = []
for f in feature_names:
    inputs.append(st.number_input(f, min_value=0, step=1))

if st.button("Predict"):
    # Create DataFrame
    X = pd.DataFrame([inputs], columns=feature_names).astype(float)

    X_imp = X

    # ---- Scale features ----
    X_scaled = scaler.transform(X_imp)

    st.subheader("Individual Model Predictions")
    preds = []
    pred_weights = []

    # Get predictions and weighted votes
    for name, model, w in zip(model_names, models, weights):
        pred = model.predict(X_scaled)[0]  # imputed + scaled input
        preds.append(pred)
        pred_weights.append((pred, w))
        st.write(f"{name}: {label_names[pred]}")

    # ---- Weighted Voting ----
    vote_counts = {}
    for cls, w in pred_weights:
        vote_counts[cls] = vote_counts.get(cls, 0) + w

    # Final prediction: class with max weighted votes
    final = max(vote_counts, key=vote_counts.get)

    st.subheader("Final Verdict")
    st.write(f"{label_names[final]}")

    # ---- Confidence ----
    total_weight = sum(vote_counts.values())
    confidence = vote_counts[final] / total_weight
    st.write(f"Confidence: {confidence * 100:.0f}%")
