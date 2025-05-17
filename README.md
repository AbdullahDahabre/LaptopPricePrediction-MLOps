# Laptop Price Prediction MLOps Project

This project demonstrates an end-to-end MLOps pipeline for laptop price prediction using multiple regression models. It covers data ingestion, preprocessing, model selection, training, hyperparameter tuning, deployment, registry, and monitoring — all automated and managed with MLflow.

---

## Project Overview

- **Objective:** Build and manage a robust machine learning pipeline for predicting laptop prices using real-world data.
- **Scope:** Automate data collection, model experimentation, deployment, and continuous monitoring within a reproducible MLOps framework.
- **Technologies:** Python, MLflow, Scikit-learn, Hyperopt, Kaggle API.

---

## Features

- Automated dataset ingestion from Kaggle using Kaggle API.
- Data preprocessing with cleaning, encoding, and scaling.
- Model selection comparing multiple regression algorithms using R² score.
- Hyperparameter tuning with Hyperopt for optimal model parameters.
- Model training with MLflow experiment tracking.
- Model deployment via MLflow Model Serving with REST API.
- Model versioning and registry management.
- Automated monitoring of deployed models with logging of performance metrics.

---

## Project Structure

├── data/ # Dataset storage

├── src/ # Source code modules

│ ├── ingestion.py # Data ingestion script

│ ├── preprocess.py # Data preprocessing functions

│ ├── model_selection.py # Model evaluation and selection

│ ├── train.py # Model training script

│ ├── tune.py # Hyperparameter tuning script

│ ├── promote.py # Model promotion script (registry)

│ ├── monitor.py # Model monitoring script

│ └── predict.py # Prediction interface script

├── start_services.py # Launch MLflow UI and model server

├── run_all.py # Run complete ML pipeline sequentially

├── requirements.txt # Python dependencies

└── README.md # This README file

---

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/laptop-price-mlops.git
    cd laptop-price-mlops
    ```

2. Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up Kaggle API credentials to enable automated data ingestion. Place your `kaggle.json` file in the appropriate directory (`~/.kaggle/` or `%USERPROFILE%\.kaggle\`).

---

## Usage

- **Data Ingestion:**
    ```bash
    python src/ingestion.py
    ```

- **Run Full Pipeline (selection, training, tuning, promotion, monitoring):**
    ```bash
    python run_all.py
    ```

- **Start MLflow UI and Model Server:**
    ```bash
    python start_services.py
    ```

- **Make Predictions:**
    ```bash
    python src/predict.py
    ```

---

## MLflow Integration

- Experiment tracking and model registry handled via MLflow.
- Models registered, promoted between stages (None, Staging, Production).
- Model serving exposes REST API at `http://127.0.0.1:1234/invocations`.
- MLflow UI accessible at `http://127.0.0.1:5000`.

---

## License

This project is licensed under the MIT License.
