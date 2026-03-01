import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess import load_data, clean_data, create_xy

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        preds.append(out.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return rmse(ys, preds)


def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    X, y = create_xy(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    if not isinstance(X_train_np, np.ndarray):
        X_train_np = X_train_np.toarray()
        X_val_np = X_val_np.toarray()
        X_test_np = X_test_np.toarray()

    input_dim = X_train_np.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_np, dtype=torch.float32),
        torch.tensor(y_val.to_numpy(), dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_np, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MLPRegressor(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    early = EarlyStopping(patience=15, min_delta=1e-4)
    n_epochs = 200

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = []
            val_ys = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                val_preds.append(out.detach().cpu().numpy())
                val_ys.append(yb.detach().cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_ys = np.concatenate(val_ys)
            val_mse = float(np.mean((val_preds - val_ys) ** 2))
            train_mse = float(np.mean(train_losses))

        stop = early.step(val_mse, model)

        if epoch == 1 or epoch % 10 == 0:
            val_rmse = np.sqrt(val_mse)
            print(f"Epoch {epoch:03d} | train_mse={train_mse:.5f} | val_rmse={val_rmse:.5f}")

        if stop:
            print(f"Early stopping at epoch {epoch}")
            break

    early.restore_best_weights(model)
    test_rmse = evaluate(model, test_loader, device)

    print("\nDeep Learning terminé")
    print("TEST RMSE (log target):", test_rmse)

    results = {
        "model": "MLP (PyTorch)",
        "target": "log1p(SalePrice)",
        "test_rmse": float(test_rmse),
        "device": device,
        "custom": {
            "name": "EarlyStopping",
            "patience": early.patience,
            "min_delta": early.min_delta
        }
    }

    with open("results_dl.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Résultats sauvegardés dans results_dl.json")


if __name__ == "__main__":
    main()