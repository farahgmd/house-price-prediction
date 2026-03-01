import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, clean_data, create_xy

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "str"]).columns

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


if __name__ == "__main__":
    df = load_data("data/train.csv")
    df = clean_data(df)
    X, y = create_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=None, n_jobs=-1)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    r = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=10, random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    feature_names = X_test.columns.tolist()

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv("feature_importance.csv", index=False, encoding="utf-8")
    print("Sauvé: feature_importance.csv")
    print(imp.head(15))