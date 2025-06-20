# train.py
# Calling: Dynamically load the best model type from MLflow registry and retrain on full dataset

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.preprocess import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Set experiment
mlflow.set_experiment("laptop_price_training")

# Fetch best model type from registry
client = MlflowClient()
latest_model_version = client.get_latest_versions(
    name="LaptopPriceBestModel", stages=["None", "Staging", "Production"]
)[-1]

# Extract model_type from run params
run_id = latest_model_version.run_id
model_type = client.get_run(run_id).data.params.get("model_type")

# Map string to class safely
model_classes = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "XGBRegressor": XGBRegressor
}

model_class = model_classes.get(model_type)

# Dynamically name the run
model_name = f"LaptopPricePredictor_{model_type}_r2_{r2_score(y_test, model_class().fit(X_train, y_train).predict(X_test)):.2f}"

# Start MLflow run
with mlflow.start_run(run_name=model_name):
    model = model_class()
    model.fit(X_train, y_train)

    # Test predictions
    preds = model.predict(X_test)

    # Compute metrics
    mse = round(mean_squared_error(y_test, preds), 3)
    mae = round(mean_absolute_error(y_test, preds), 3)
    rmse = round(np.sqrt(mse), 3)
    r2 = round(r2_score(y_test, preds), 3)
    maxerr = round(max_error(y_test, preds), 3)

    # Detect overfitting or underfitting
    train_preds = model.predict(X_train)
    train_r2 = r2_score(y_train, train_preds)
    r2_gap = round(train_r2 - r2, 3)

    fit_status = "Good Fit"
    if train_r2 > 0.85 and r2 < 0.65 and r2_gap > 0.2:
        fit_status = "Overfitting"
    elif train_r2 < 0.5 and r2 < 0.5:
        fit_status = "Underfitting"

    # Log to MLflow
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("fit_status", fit_status)
    mlflow.log_metric("train_r2", round(train_r2, 3))
    mlflow.log_metric("r2_gap", r2_gap)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("max_error", maxerr)

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

    # Console feedback
    print(f"Successfully retrained and logged the best model: {model_name}")
    print(f"Model Fit Status: {fit_status} (Train R²: {train_r2:.3f}, Test R²: {r2:.3f}, Gap: {r2_gap})")

print(f"Best model type fetched from MLflow: {model_type}")