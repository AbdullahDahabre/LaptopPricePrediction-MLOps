# streamlit_app.py
# Streamlit front-end for Laptop Price Prediction using the Production MLflow model

import os
import pandas as pd
import streamlit as st
import mlflow.pyfunc
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Loading constants
RAW_DATA_PATH = "data/laptop_price.csv"
MODEL_URI = "models:/LaptopPriceBestModel/Production"
LOG_PATH = "data/predictions_log.csv"

# Caching the raw dataset to avoid reloading every time
@st.cache_resource
def load_raw_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")

# Loading production model registered in MLflow
@st.cache_resource
def load_model(uri: str):
    return mlflow.pyfunc.load_model(uri)

# Calling raw dataset and model
raw_df = load_raw_dataset(RAW_DATA_PATH)
model = load_model(MODEL_URI)

# Extracting column types for form generation
X_raw = raw_df.drop(columns=["Price"])
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_raw.select_dtypes(exclude=["object"]).columns.tolist()

# Custom dropdown features (limit to known values)
dropdown_numeric = {
    "RAM_Size": sorted(raw_df["RAM_Size"].dropna().unique().tolist()),
    "Storage_Capacity": sorted(raw_df["Storage_Capacity"].dropna().unique().tolist())
}
# Remove dropdown-numeric from general numeric_cols
numeric_cols = [col for col in numeric_cols if col not in dropdown_numeric]

# Setting up Streamlit page configuration
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("Laptop Price Prediction")
st.markdown("Provide laptop specifications to estimate the market price.")

# Creating a form for user input
with st.form("prediction_form"):
    user_inputs = {}

    # Generating dropdowns for categorical features
    for col in categorical_cols:
        options = sorted(raw_df[col].dropna().unique().tolist())
        default_val = options[0] if options else ""
        user_inputs[col] = st.selectbox(col, options, index=0)

    # Generating dropdowns for specific numeric features
    for col in dropdown_numeric:
        options = dropdown_numeric[col]
        default_val = options[0] if options else 0
        user_inputs[col] = st.selectbox(col, options, index=0)

    # Generating number inputs for general numerical features
    for col in numeric_cols:
        col_min = float(raw_df[col].min()) * 0.5
        col_max = float(raw_df[col].max()) * 1.5
        default_val = float(raw_df[col].median())

        step_val = 0.05 if col == "Weight" else 0.1
        user_inputs[col] = st.number_input(
            col,
            min_value=col_min,
            max_value=col_max,
            value=default_val,
            step=step_val,
        )

    submitted = st.form_submit_button("Predict Price")

# If form is submitted, proceed to prediction
if submitted:
    input_df = pd.DataFrame([user_inputs])

    # Keep a copy of original user selections for logging
    raw_log = user_inputs.copy()

    # Encode categorical values before prediction
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(raw_df[col])
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            st.error(f"Value '{input_df[col].iloc[0]}' for '{col}' is not recognised.")
            st.stop()

    # Ensuring consistent data types for prediction
    input_df = input_df.astype("float64")
    input_df = input_df[X_raw.columns]  # Ensure feature order matches training

    # Predicting the laptop price using MLflow model
    pred_price = model.predict(input_df)[0]
    st.success(f"Estimated Laptop Price: **${pred_price:,.2f}**")

    # Appending prediction to log file with original values
    raw_log["PredictedPrice"] = round(pred_price, 2)
    raw_log["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    log_df = pd.DataFrame([raw_log])
    log_df.to_csv(LOG_PATH, mode="a", index=False, header=not os.path.exists(LOG_PATH))

    st.info("Prediction logged to predictions_log.csv")