# run_all.py

import subprocess
import time
import os

# Full path to venv python executable
venv_python = r"C:\Users\dahab\OneDrive\Desktop\Project-AbdullahDahabre-2281427\venv\Scripts\python.exe"

steps = [
    ("Model Selection", [venv_python, "-m", "src.model_selection"]),
    ("Training Best Model", [venv_python, "-m", "src.train"]),
    ("Hyperparameter Tuning", [venv_python, "-m", "src.tune"]),
    ("Model Promotion", [venv_python, "-m", "src.promote"]),
    ("Model Monitoring", [venv_python, "-m", "src.monitor"])
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