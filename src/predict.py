# predict.py
# Calling: Collect user input, preprocess it, send to MLflow model for prediction,
# and log both input and prediction with a timestamp to a CSV file

import json
import requests
import pandas as pd
from preprocess import load_and_preprocess_data
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# Load preprocessed data and raw data for structure and label encoders
X_train, X_test, y_train, y_test = load_and_preprocess_data()
df_full = pd.concat([X_train, X_test])

# Load original raw dataset to identify categorical columns
df_raw = pd.read_csv("data/laptop_price.csv", encoding='latin-1')
df_raw.dropna(inplace=True)
X_raw = df_raw.drop(columns=["Price"])
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()

# Display column names and expected data types
print("Please enter the following values exactly as shown below:\n")
for col in df_full.columns:
    dtype = "string (case-insensitive)" if col in categorical_cols else "float"
    print(f"- {col}: {dtype}")

# Collect user input for each feature
user_input = {}
for col in df_full.columns:
    value = input(f"\nEnter value for {col}: ")

    # Normalize text input to Title Case (e.g., aSuS → Asus)
    if col in categorical_cols:
        user_input[col] = value.strip().title()
    else:
        # Validate numeric input
        try:
            user_input[col] = float(value)
        except ValueError:
            print(f"Invalid input type for {col}. Please restart the script and try again.")
            exit()

# Convert input dictionary to a DataFrame
input_df = pd.DataFrame([user_input])

# Apply label encoding based on training data for categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(X_raw[col])
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        print(f"The value '{user_input[col]}' for column '{col}' is not recognized from training data.")
        exit()

# Ensure all values are float64 for MLflow compatibility
input_df = input_df.astype("float64")

# Prepare the payload in MLflow v2 scoring format (dataframe_split)
data = {
    "dataframe_split": {
        "columns": list(input_df.columns),
        "data": input_df.values.tolist()
    }
}

# Send POST request to the deployed MLflow model
response = requests.post(
    url="http://127.0.0.1:1234/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

# Parse and handle the response
try:
    result = response.json()

    # Handle MLflow 2.x structured response
    if isinstance(result, dict) and "predictions" in result:
        prediction = result["predictions"][0]

        # Limit prediction to 3 decimal places
        prediction_rounded = round(prediction, 3)

        print("\nPredicted Laptop Price:", prediction_rounded)

        # Prepare log entry using original (human-readable) inputs
        log_data = pd.DataFrame([user_input])
        log_data["PredictedPrice"] = prediction_rounded  # save rounded value

        # Add current timestamp
        log_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to CSV log file
        log_path = "data/predictions_log.csv"
        header = not os.path.exists(log_path)
        log_data.to_csv(log_path, mode="a", index=False, header=header)

        print(f"Prediction saved to: {log_path}")

    elif isinstance(result, list):
        # Handle older format (list of predictions)
        prediction_rounded = round(result[0], 3)
        print("\nPredicted Laptop Price:", prediction_rounded)

    else:
        # Unexpected format
        print("\nUnexpected response format:")
        print(result)

except Exception as e:
    print("\nFailed to parse response:", e)
    print("Raw response text:")
    print(response.text)