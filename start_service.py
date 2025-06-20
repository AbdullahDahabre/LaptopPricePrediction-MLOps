# start_services.py
# Purpose: Launch the MLflow Tracking UI, MLflow Model Server, and Streamlit frontend

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

# Optional delay if needed for model server to be ready
time.sleep(2)

print("Launching Streamlit app at http://localhost:8501/ ...")
streamlit_app = subprocess.Popen([
    "streamlit", "run", "streamlit_app.py"
])

print("\nAll services are now running:")
print("MLflow UI        -> http://127.0.0.1:5000")
print("Model Server     -> http://127.0.0.1:1234")
print("Streamlit App    -> http://localhost:8501")
print("Press Ctrl+C to stop them manually.")