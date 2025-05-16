# train.py
# Calling: Dynamically load the best model type from MLflow registry and retrain on full dataset

# Standard library imports
import numpy as np

# Third-party library imports
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

# Local application imports
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

# Convert string to model class (Dynamic model instantiation)
model_class = eval(model_type)

# Dynamically create model name (Model name includes MSE rounded to 2 decimals)
model_name = f"LaptopPricePredictor_{model_type}_r2_{r2_score(y_test, model_class().fit(X_train, y_train).predict(X_test)):.2f}"

# Start an MLflow run and train the model
with mlflow.start_run(run_name=model_name):
    # Instantiate the model
    model = model_class()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    preds = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    maxerr = max_error(y_test, preds)

    # Round metrics to 3 decimal places
    mse = round(mse, 3)
    mae = round(mae, 3)
    rmse = round(rmse, 3)
    r2 = round(r2, 3)
    maxerr = round(maxerr, 3)

    # Log model and metrics to MLflow
    mlflow.log_param("model_type", model_type)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("max_error", maxerr)

    # Log the retrained model (with dynamic name)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

    print(f"Successfully retrained and logged the best model: {model_name}")

print(f"Best model type fetched from MLflow: {model_type}")