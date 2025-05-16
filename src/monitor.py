# monitor.py
# Purpose: Evaluate the deployed model from MLflow registry on a sample of the preprocessed dataset,
# calculate regression metrics, and log monitoring results within MLflow.

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from src.preprocess import load_and_preprocess_data

# MLflow experiment and registered model configuration
MONITOR_EXP = 'laptop_price_monitoring'
MODEL_NAME = 'LaptopPriceBestModel'   # Registered model name in MLflow Model Registry
MODEL_STAGE = 'Production'                  # Model stage to load

def main():
    # Load preprocessed dataset and split into train and test sets
    _, X_test, _, y_test = load_and_preprocess_data()

    # Sample 10% of test data for evaluation
    sample_size = int(len(X_test) * 0.1)
    X_sample = X_test.sample(n=sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    # Load production model from MLflow Model Registry
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Generate predictions on the sampled data
    y_pred = model.predict(X_sample)

    # Compute regression metrics
    mse = mean_squared_error(y_sample, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_sample, y_pred)
    max_err = max_error(y_sample, y_pred)
    r2 = r2_score(y_sample, y_pred)

    # Log metrics and parameters in MLflow monitoring experiment
    mlflow.set_experiment(MONITOR_EXP)
    with mlflow.start_run():
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('max_error', max_err)
        mlflow.log_metric('r2', r2)
        mlflow.log_param('sample_size', sample_size)
        mlflow.log_param('model_stage', MODEL_STAGE)

    # Output the monitoring results summary with rounded metrics
    print(f"Logged monitoring metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, MSE={mse:.3f}, Max Error={max_err:.3f}, R²={r2:.3f} on {sample_size} samples.")

if __name__ == '__main__':
    main()