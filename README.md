# Laptop Price Prediction MLOps Project
![Laptop Price Prediction Banner](https://www.tzvi.dev/posts/introducing-laptop-price-prediction/featured.png)

This project demonstrates an end-to-end MLOps pipeline for laptop price prediction using multiple regression models. It covers data ingestion, preprocessing, model selection, training, hyperparameter tuning, deployment, registry, monitoring, and user interaction — all automated and managed with MLflow and Streamlit.

---

## Project Overview

- **Objective:** Build and manage a robust machine learning pipeline for predicting laptop prices using real-world data.
- **Scope:** Automate data collection, model experimentation, deployment, and continuous monitoring within a reproducible MLOps framework.
- **Technologies:** Python, MLflow, Scikit-learn, Hyperopt, Streamlit, Kaggle API.

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
- SHAP visualizations and performance charts.
- Streamlit app for live predictions and user interface.

---

## Project Structure

├── data/ # Dataset storage  
│ ├── Laptop_price.csv  
│ ├── laptop_price_cleaned.csv  
│ └── predictions_log.csv  

├── mlruns/ # MLflow tracking directory  

├── src/ # Source code modules  
│ ├── ingestion.py # Download dataset from Kaggle  
│ ├── preprocess.py # Clean and transform features  
│ ├── save_clean_data.py # Save cleaned dataset  
│ ├── model_selection.py # Compare models by R² score  
│ ├── train.py # Retrain best model  
│ ├── tune.py # Hyperopt parameter tuning  
│ ├── promote.py # Promote model to registry stages  
│ ├── monitor.py # Evaluate production model periodically  
│ ├── visualize.py # Generate performance and SHAP plots  
│ └── predict.py # Make predictions using the deployed model  

├── streamlit_app.py # Interactive UI for live predictions  
├── start_service.py # Launch MLflow UI, model server, and Streamlit app  
├── run_all.py # Run complete ML pipeline sequentially  
├── promotion_log.txt # Text log of promotion actions  
├── requirements.txt # Python dependencies  
├── .gitattributes # Git LFS settings (if any)  
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
    venv\Scripts\activate  # On Windows
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up Kaggle API credentials to enable automated data ingestion. Place your `kaggle.json` file in the appropriate directory:
    ```
    %USERPROFILE%\.kaggle\
    ```

---

## Usage

- **Data Ingestion:**
    ```bash
    python -m src.ingestion
    ```

- **Run Full Pipeline (selection, training, tuning, promotion, monitoring):**
    ```bash
    python run_all.py
    ```

- **Start MLflow UI, Model Server, and Streamlit App:**
    ```bash
    python start_service.py
    ```

- **Make Predictions:**
    ```bash
    python -m src.predict
    ```
    
- **Run Streamlit App separately:**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## MLflow Integration

- Experiment tracking and model registry handled via MLflow.
- Models are registered and promoted between stages (None, Staging, Production).
- Model REST API available at `http://127.0.0.1:1234/invocations`
- MLflow Tracking UI available at `http://127.0.0.1:5000`
- Streamlit app available at `http://localhost:8501`

---

## License

This project is licensed under the MIT License.
