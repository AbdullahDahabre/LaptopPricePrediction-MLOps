# save_cleaned_data.py
# Calling: Save the fully preprocessed and cleaned dataset into a single CSV file for visualization or reuse

import pandas as pd
from preprocess import load_and_preprocess_data

def save_cleaned_data():
    # Load and preprocess without saving inside function
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Combine train + test back to save the whole cleaned dataset
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    df_cleaned = X_full.copy()
    df_cleaned["Price"] = y_full.values

    output_path = "data/laptop_price_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    save_cleaned_data()
