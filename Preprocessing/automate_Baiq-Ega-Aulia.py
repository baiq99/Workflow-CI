# automate_processing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import os

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocess dataset from input_path and save the cleaned dataset to output_path.

    Steps:
    1. Handle missing values with most frequent strategy
    2. Remove duplicate rows
    3. Normalize/standardize numerical features
    4. Detect and remove outliers based on z-score
    5. Encode categorical features using LabelEncoder
    6. Bin 'Administrative_Duration' into quartiles (if present)

    Args:
        input_path (str): Filepath to the raw CSV dataset.
        output_path (str): Filepath to save the preprocessed CSV dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """

    # Load dataset
    data = pd.read_csv(input_path)
    print(f"Loaded data with shape: {data.shape}")

    # 1. Handling Missing Values
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed = data_imputed.convert_dtypes()
    print("Missing values handled.")

    # 2. Remove Duplicate Rows
    before_dupes = data_imputed.shape[0]
    data_imputed.drop_duplicates(inplace=True)
    after_dupes = data_imputed.shape[0]
    print(f"Removed {before_dupes - after_dupes} duplicate rows.")

    # 3. Normalize / Standardize Numerical Features
    numerical_features = data_imputed.select_dtypes(include=['number']).columns.tolist()

    # Convert possible numeric columns to numeric types (coerce errors)
    for col in numerical_features:
        if not pd.api.types.is_numeric_dtype(data_imputed[col]):
            data_imputed[col] = pd.to_numeric(data_imputed[col], errors='coerce')

    # Drop rows with NaNs caused by coercion
    data_imputed.dropna(subset=numerical_features, inplace=True)

    scaler = StandardScaler()
    data_imputed[numerical_features] = scaler.fit_transform(data_imputed[numerical_features])
    print("Numerical features standardized.")

    # 4. Outlier Detection and Removal (Z-score method)
    z_scores = np.abs(stats.zscore(data_imputed[numerical_features]))
    if z_scores.shape[1] == len(numerical_features):
        threshold = 3
        before_outliers = data_imputed.shape[0]
        data_imputed = data_imputed[(z_scores < threshold).all(axis=1)]
        after_outliers = data_imputed.shape[0]
        print(f"Removed {before_outliers - after_outliers} outlier rows based on z-score.")
    else:
        print("Skipping outlier detection due to dimension mismatch.")

    # 5. Encoding Categorical Features
    categorical_features = data_imputed.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in categorical_features:
        le = LabelEncoder()
        try:
            data_imputed[col] = le.fit_transform(data_imputed[col].astype(str))
        except Exception as e:
            print(f"Encoding failed on column {col}: {e}")
    print("Categorical features encoded.")

    # 6. Binning 'Administrative_Duration' (if present)
    if 'Administrative_Duration' in data_imputed.columns:
        if not data_imputed['Administrative_Duration'].isnull().all():
            try:
                data_imputed['Administrative_Duration_Bin'] = pd.qcut(
                    data_imputed['Administrative_Duration'],
                    q=4,
                    labels=False,
                    duplicates='drop'
                )
                print("'Administrative_Duration' binned into quartiles.")
            except Exception as e:
                print(f"Error during binning 'Administrative_Duration': {e}")
        else:
            print("'Administrative_Duration' column empty, skipping binning.")
    else:
        print("'Administrative_Duration' column not found, skipping binning.")

    # Save preprocessed dataset
    data_imputed.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path}")

    return data_imputed


if __name__ == "__main__":
    # Path relative to project root
    input_csv = os.path.join("Dataset", "online_shoppers_intention.csv")
    output_csv = os.path.join("Dataset", "online_shoppers_intention_preprocessed.csv")

    processed_df = preprocess_data(input_csv, output_csv)

    print("\n Preprocessing completed.")
    print(" Preview of preprocessed data:")
    print(processed_df.head())
