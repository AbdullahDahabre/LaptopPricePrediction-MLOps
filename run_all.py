# run_all.py
# Calling: Execute the full MLOps pipeline sequentially including model selection, training, tuning, promotion, monitoring, and visualizations

import subprocess
import time
import os
import sys

# Full path to venv python executable
venv_python = sys.executable

steps = [
    ("Model Selection", [venv_python, "-m", "src.model_selection"]),
    ("Training Best Model", [venv_python, "-m", "src.train"]),
    ("Hyperparameter Tuning", [venv_python, "-m", "src.tune"]),
    ("Model Promotion", [venv_python, "-m", "src.promote"]),
    ("Model Monitoring", [venv_python, "-m", "src.monitor"]),
    ("Generate Visualizations", [venv_python, "-m", "src.visualize"]),
]

print("Starting full ML pipeline...\n")

for name, command in steps:
    print(f"=== {name} ===")
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    end_time = time.time()

    if result.returncode != 0:
        print(f"\nStep '{name}' failed. Aborting pipeline.")
        print(f"Error Output: {result.stderr}")
        break

    print(f"Completed: {name} in {end_time - start_time:.2f} seconds.")
    print(f"Output: {result.stdout}\n")

print("Pipeline execution complete.")