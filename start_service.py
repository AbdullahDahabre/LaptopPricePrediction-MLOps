# start_services.py
# Purpose: Launch the MLflow Tracking UI and MLflow Model Server for serving predictions

import subprocess
import time

# Define model name and version to serve
model_uri = "models:/LaptopPriceRegressorFinal/Production"

print("Starting MLflow Tracking UI at http://127.0.0.1:5000/ ...")
mlflow_ui = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Wait a moment to allow the UI to initialize
time.sleep(2)

print(f"Starting MLflow Model Server at http://127.0.0.1:1234/ using model: {model_uri} ...")
model_server = subprocess.Popen([
    "mlflow", "models", "serve",
    "-m", model_uri,
    "-p", "1234",
    "--no-conda"
])

print("\nBoth services are now running.")
print("Press Ctrl+C to stop them manually.")