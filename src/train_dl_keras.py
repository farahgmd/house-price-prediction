import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from preprocess import load_data, clean_data, create_xy


class RMSEMetric(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse = keras.metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mse.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return tf.sqrt(self.mse.result())

    def reset_states(self):
        self.mse.reset_states()


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "str"]).columns

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def rmse_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


if __name__ == "__main__":
    # 1) Load + clean
    df = load_data("data/train.csv")
    df = clean_data(df)
    X, y = create_xy(df)

    # 2) Split train/test puis train/val
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # 3) Preprocess -> matrices
    preprocessor = build_preprocessor(X_train)
    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    # densify si sparse
    if not isinstance(X_train_np, np.ndarray):
        X_train_np = X_train_np.toarray()
        X_val_np = X_val_np.toarray()
        X_test_np = X_test_np.toarray()

    y_train_np = y_train.to_numpy().astype("float32")
    y_val_np = y_val.to_numpy().astype("float32")
    y_test_np = y_test.to_numpy().astype("float32")

    input_dim = X_train_np.shape[1]

    # 4) Model MLP
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[RMSEMetric()]  
    )

    # 5) Training + EarlyStopping (callback)
    early = keras.callbacks.EarlyStopping(
        monitor="val_rmse",
        patience=15,
        mode="min",
        restore_best_weights=True
    )

    history = model.fit(
        X_train_np, y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=200,
        batch_size=64,
        callbacks=[early],
        verbose=1
    )

    # 6) Test RMSE
    y_pred = model.predict(X_test_np).reshape(-1)
    test_rmse = rmse_np(y_test_np, y_pred)

    print("\n Deep Learning (Keras) terminé")
    print("TEST RMSE (log target):", test_rmse)

    results = {
        "model": "MLP (Keras)",
        "target": "log1p(SalePrice)",
        "test_rmse": float(test_rmse),
        "custom": "RMSEMetric (custom Keras metric)",
        "early_stopping": {"patience": 15, "restore_best_weights": True}
    }

    with open("results_dl.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("📄 Résultats sauvegardés dans results_dl.json")