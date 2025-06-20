# promote.py
# Purpose: Promote the latest registered model to Staging and Production in MLflow,
# optionally filter outdated models, archive old versions, log changes, and run monitoring.

from mlflow.tracking import MlflowClient
from datetime import datetime
import subprocess

# Define model name
model_name = "LaptopPriceBestModel"
client = MlflowClient()

# Set cutoff date for model creation (format: YYYY, M, D)
cutoff_date = datetime(2025, 4, 20)

# Retrieve the latest unpromoted model
latest_none_versions = client.get_latest_versions(name=model_name, stages=["None"])
if not latest_none_versions:
    print("No new model version found in 'None' stage to promote.")
    exit()

latest_model = latest_none_versions[0]
new_version = latest_model.version
run_id = latest_model.run_id

# Check model creation date
run_info = client.get_run(run_id)
creation_time = datetime.fromtimestamp(run_info.info.start_time / 1000)

if creation_time < cutoff_date:
    print(f"Model version {new_version} was created on {creation_time} and is older than the cutoff ({cutoff_date}). Skipping promotion.")
    exit()

print(f"Promoting model version {new_version}...")

# Helper function to get current version in a stage
def get_current_version(stage):
    versions = client.get_latest_versions(name=model_name, stages=[stage])
    return versions[0].version if versions else None

# Track promotion changes
log_lines = []
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_lines.append(f"Promotion Log — {timestamp}")

# Promote to Staging
prev_staging = get_current_version("Staging")
client.transition_model_version_stage(
    name=model_name,
    version=new_version,
    stage="Staging",
    archive_existing_versions=True
)
log_lines.append(f"Staging: v{prev_staging} -> v{new_version}" if prev_staging else f"Staging: Set to v{new_version}")
print(log_lines[-1])

# Promote to Production
prev_production = get_current_version("Production")
client.transition_model_version_stage(
    name=model_name,
    version=new_version,
    stage="Production",
    archive_existing_versions=True
)
log_lines.append(f"Production: v{prev_production} -> v{new_version}" if prev_production else f"Production: Set to v{new_version}")
print(log_lines[-1])

# Save log to file
with open("promotion_log.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n\n")

print("Promotion completed successfully. Log saved to promotion_log.txt.")