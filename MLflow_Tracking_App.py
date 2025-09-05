import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="MLflow Experiment Tracking", layout="centered")
st.title("🧪 Day 23 — MLflow: Track Model Experiments")

# Generate synthetic dataset
@st.cache_data
def generate_data(n=500, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] * 2 + rng.randn(n) > 0).astype(int)
    return pd.DataFrame(X, columns=[f"Feature{i}" for i in range(1, 6)]), y

X, y = generate_data()
st.subheader("📂 Sample Data")
st.write(pd.DataFrame(X).head())

# Split dataset
test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Hyperparameters
C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
max_iter = st.slider("Max Iterations", 100, 1000, 200)

if st.button("Run Experiment"):
    with mlflow.start_run():
        # Model training
        model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params, metrics, and model
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        st.success(f"✅ Experiment logged! Accuracy: {acc:.3f}")

        st.info("Check your MLflow UI with: `mlflow ui`")
