import json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from preprocess import load_data, clean_data, create_xy

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "str"]).columns

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


if __name__ == "__main__":
    
    df = load_data("data/train.csv")
    df = clean_data(df)
    X, y = create_xy(df)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    
    ridge = Ridge(random_state=42)

    ridge_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", ridge),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_grid = GridSearchCV(
        ridge_pipe,
        param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    ridge_grid.fit(X_train, y_train)

    ridge_best = ridge_grid.best_estimator_
    ridge_test_rmse = rmse(y_test, ridge_best.predict(X_test))

    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    rf_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    rf_grid = GridSearchCV(
        rf_pipe,
        param_grid={
            "model__n_estimators": [200],
            "model__max_depth": [None, 10],
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train, y_train)

    rf_best = rf_grid.best_estimator_
    rf_test_rmse = rmse(y_test, rf_best.predict(X_test))

    
    print("\n  BASELINE Ridge")
    print("Best params:", ridge_grid.best_params_)
    print("Best CV RMSE:", -ridge_grid.best_score_)
    print("TEST RMSE:", ridge_test_rmse)

    print("\n RandomForest (tuned)")
    print("Best params:", rf_grid.best_params_)
    print("Best CV RMSE:", -rf_grid.best_score_)
    print("TEST RMSE:", rf_test_rmse)

    results = {
        "target": "log1p(SalePrice)",
        "split": {"test_size": 0.2, "random_state": 42},
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 42},
        "models": {
            "ridge": {
                "best_params": ridge_grid.best_params_,
                "cv_rmse": float(-ridge_grid.best_score_),
                "test_rmse": ridge_test_rmse,
            },
            "random_forest": {
                "best_params": rf_grid.best_params_,
                "cv_rmse": float(-rf_grid.best_score_),
                "test_rmse": rf_test_rmse,
            }
        }
    }

    with open("results_classical.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n Résultats sauvegardés dans results_classical.json")
    