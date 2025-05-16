# preprocess.py
# Calling: Preprocess laptop_price.csv by cleaning, encoding, scaling, and saving it

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(path="data/laptop_price.csv"):
    # Load dataset
    df = pd.read_csv(path, encoding='latin-1')

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Separate features and target
    X = df.drop(columns=["Price"])
    y = df["Price"]

    # Apply label encoding to categorical columns
    for column in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Round float columns to 2 decimal places
    float_cols = X.select_dtypes(include=["float", "float64"]).columns
    X[float_cols] = X[float_cols].round(2)

    # Convert float columns that contain only whole numbers to int
    for col in float_cols:
        if (X[col] % 1 == 0).all():
            X[col] = X[col].astype("int")

    # Apply standard scaling to all numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Return train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)