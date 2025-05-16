# tune.py
# Calling: Hyperparameter tuning with Hyperopt + MLflow, depending on the best selected model

import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from src.preprocess import load_and_preprocess_data
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Load dataset and split into train/test sets
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Define mapping of model names to their classes for dynamic instantiation
models = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "XGBRegressor": XGBRegressor
}

# Fetch the best model info from MLflow model registry
client = mlflow.tracking.MlflowClient()
latest_model_version = client.get_latest_versions(
    name="LaptopPriceBestModel", stages=["None", "Staging", "Production"]
)[-1]

# Extract the model type string from the MLflow run parameters
run_id = latest_model_version.run_id
model_type = client.get_run(run_id).data.params.get("model_type")

# Get the corresponding model class from the dictionary
best_model_class = models[model_type]

# Define hyperparameter search space based on the selected model type
if model_type == "RandomForestRegressor":
    space = {
        "n_estimators": hp.choice("n_estimators", [50, 100, 150]),
        "max_depth": hp.choice("max_depth", [5, 10, 15, 20]),
        "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.3)
    }
elif model_type == "XGBRegressor":
    space = {
        "n_estimators": hp.choice("n_estimators", [50, 100, 150]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "max_depth": hp.choice("max_depth", [5, 10, 15, 20]),
        "subsample": hp.uniform("subsample", 0.5, 1.0)
    }
elif model_type == "SVR":
    space = {
        "C": hp.loguniform("C", 0, 10),
        "epsilon": hp.uniform("epsilon", 0.01, 0.1),
        "kernel": hp.choice("kernel", ["linear", "rbf"])
    }
elif model_type == "DecisionTreeRegressor":
    space = {
        "max_depth": hp.choice("max_depth", [5, 10, 15, 20]),
        "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.3)
    }
elif model_type == "KNeighborsRegressor":
    space = {
        "n_neighbors": hp.choice("n_neighbors", [5, 10, 15]),
        "weights": hp.choice("weights", ["uniform", "distance"]),
        "p": hp.choice("p", [1, 2])
    }
elif model_type in ["Ridge", "Lasso"]:
    # Linear models mainly tune 'alpha' regularization parameter
    space = {
        "alpha": hp.loguniform("alpha", -5, 5)
    }

# Define the objective function for Hyperopt tuning
def objective(params):
    with mlflow.start_run(nested=True):
        # Initialize the model with current hyperparameters
        model = best_model_class(**params)
        model.fit(X_train, y_train)  # Train on training data
        preds = model.predict(X_test)  # Predict on test data

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        maxerr = max_error(y_test, preds)

        # Round metrics to 3 decimals before logging
        mse = round(mse, 3)
        mae = round(mae, 3)
        rmse = round(rmse, 3)
        r2 = round(r2, 3)
        maxerr = round(maxerr, 3)

        # Log hyperparameters and metrics to MLflow for tracking
        mlflow.log_params(params)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("max_error", maxerr)

        # Hyperopt tries to minimize the 'loss' so we use mse here
        return {"loss": mse, "status": STATUS_OK}

# Set the MLflow experiment to track tuning runs
mlflow.set_experiment("laptop_price_tuning")

# Run hyperparameter tuning with Hyperopt and MLflow logging
with mlflow.start_run(run_name=f"{model_type}_hyperopt_tuning"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,  # Number of tuning iterations
        trials=trials
    )

# Map the best results from Hyperopt to model parameters correctly
if model_type == "RandomForestRegressor":
    final_params = {
        "n_estimators": best_result["n_estimators"],
        "max_depth": best_result["max_depth"],
        "min_samples_split": best_result["min_samples_split"]
    }
elif model_type == "XGBRegressor":
    final_params = {
        "n_estimators": best_result["n_estimators"],
        "learning_rate": best_result["learning_rate"],
        "max_depth": best_result["max_depth"],
        "subsample": best_result["subsample"]
    }
elif model_type == "SVR":
    # Convert kernel choice index back to string
    kernel_map = ["linear", "rbf"]
    final_params = {
        "C": best_result["C"],
        "epsilon": best_result["epsilon"],
        "kernel": kernel_map[best_result["kernel"]]
    }
elif model_type == "DecisionTreeRegressor":
    final_params = {
        "max_depth": best_result["max_depth"],
        "min_samples_split": best_result["min_samples_split"]
    }
elif model_type == "KNeighborsRegressor":
    weights_map = ["uniform", "distance"]
    final_params = {
        "n_neighbors": best_result["n_neighbors"],
        "weights": weights_map[best_result["weights"]],
        "p": best_result["p"]
    }
elif model_type in ["Ridge", "Lasso"]:
    final_params = {
        "alpha": best_result["alpha"]
    }

# Train the final model on full training data with tuned hyperparameters
final_model = best_model_class(**final_params)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)

# Calculate final evaluation metrics
final_mse = mean_squared_error(y_test, final_preds)
final_mae = mean_absolute_error(y_test, final_preds)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(y_test, final_preds)
final_maxerr = max_error(y_test, final_preds)

# Round final metrics to 3 decimals before logging
final_mse = round(final_mse, 3)
final_mae = round(final_mae, 3)
final_rmse = round(final_rmse, 3)
final_r2 = round(final_r2, 3)
final_maxerr = round(final_maxerr, 3)

# Log final model and metrics to MLflow
with mlflow.start_run(run_name="final_best_model"):
    mlflow.log_params(final_params)             # Log tuned hyperparameters
    mlflow.log_param("model_type", model_type) # Log model type for reference
    mlflow.log_metric("r2_score", final_r2)
    mlflow.log_metric("mse", final_mse)
    mlflow.log_metric("mae", final_mae)
    mlflow.log_metric("rmse", final_rmse)
    mlflow.log_metric("max_error", final_maxerr)

    # Log the final tuned model as an artifact and register it
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name="LaptopPriceRegressorFinal"
    )

    print(f"Final tuned model for {model_type} has been logged.")