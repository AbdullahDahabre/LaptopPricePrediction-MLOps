# visualize.py
# Calling: Generate performance and explainability visualizations including SHAP, and log them to MLflow

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from sklearn.metrics import r2_score
from src.preprocess import load_and_preprocess_data

# Set MLflow experiment
mlflow.set_experiment("laptop_price_visuals")

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])
df_cleaned = pd.read_csv("data/laptop_price_cleaned.csv")

# Load model from registry
model_uri = "models:/LaptopPriceBestModel/Production"
model = mlflow.sklearn.load_model(model_uri)

# Make predictions
y_pred = model.predict(X)

# Detect model type
model_type = type(model).__name__

# Start MLflow run
with mlflow.start_run(run_name="full_visual_report"):

    # Predicted vs Actual
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual Laptop Prices")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    mlflow.log_figure(plt.gcf(), "pred_vs_actual.png")
    plt.clf()

    # Residuals Distribution
    residuals = y - y_pred
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title("Residuals Distribution")
    mlflow.log_figure(plt.gcf(), "residuals_dist.png")
    plt.clf()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cleaned.corr(), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    mlflow.log_figure(plt.gcf(), "feature_corr_heatmap.png")
    plt.clf()

    # Combined Feature Distributions (Top 5 correlated)
    top_features = df_cleaned.corr()["Price"].abs().sort_values(ascending=False)[1:6].index.tolist()
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df_cleaned[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "distributions_all.png")
    plt.clf()

    # SHAP Explainability
    try:
        if "Tree" in model_type or "Forest" in model_type or "XGB" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, X_train)

        shap_values = explainer(X_test)

        # SHAP bar plot
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=10, show=False)
        plt.tight_layout()
        mlflow.log_figure(fig_bar, "shap_summary_bar.png")
        plt.clf()

        # SHAP beeswarm plot
        fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        mlflow.log_figure(fig_bee, "shap_beeswarm.png")
        plt.clf()

        mlflow.log_param("shap_status", "success")
        print("SHAP explainability plots have been generated and logged.")
    except Exception as e:
        mlflow.log_param("shap_status", f"failed: {str(e)}")
        print("SHAP generation failed with error:", e)

print("All visualizations have been logged to MLflow.")