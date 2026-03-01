import pandas as pd
import numpy as np

def load_data(path):
    """Load data from a CSV file."""
    return pd.read_csv(path)

def clean_data(df):
    """Clean the dataframe by handling missing values and data types."""
    df = df.copy()
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Fill remaining NaN values for categorical columns
    df = df.fillna(df.mode().iloc[0])
    
    return df

def create_xy(df):
    """Split features and target variable."""
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])
    return X, y

def preprocess_pipeline(path):
    """Complete preprocessing pipeline."""
    df = load_data(path)
    df = clean_data(df)
    X, y = create_xy(df)
    return X, y