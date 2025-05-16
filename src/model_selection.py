# model_selection.py
# Calling: Evaluate multiple regression models and register the best one using MLflow

# Standard library imports
import numpy as np

# Third-party library imports
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Local application imports
from src.preprocess import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Define model candidates
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "XGBRegressor": XGBRegressor()
}

# Set MLflow experiment
mlflow.set_experiment("laptop_price_model_selection")

# Track best model by highest R2 score
best_model_name = None
best_model = None
best_model_accuracy = 0  # store best r2_score

# Evaluate each model candidate
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calculate r2 score and round to 2 decimals
        r2 = round(r2_score(y_test, preds), 3)

        # Log r2_score metric
        mlflow.log_metric("r2_score", r2)

        # Update best model if current r2 is better
        if r2 > best_model_accuracy:
            best_model_accuracy = r2
            best_model_name = name
            best_model = model

# Final run to register best model
with mlflow.start_run(run_name="best_model_registration"):
    best_model.fit(X_train, y_train)
    final_preds = best_model.predict(X_test)
    final_r2 = r2_score(y_test, final_preds)

    # Log model type and r2 score (rounded)
    mlflow.log_param("model_type", best_model_name)
    mlflow.log_metric("r2_score", round(final_r2, 3))

    # Log the best model to MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="LaptopPriceBestModel"
    )

print(f"Best model selected and registered: {best_model_name}")