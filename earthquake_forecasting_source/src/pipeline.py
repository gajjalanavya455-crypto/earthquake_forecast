from pathlib import Path
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_catalogue, make_features
from .model import EarthquakeLSTM

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_windows(features, target, window, train_end, val_end):
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for target_idx in range(window, len(features)):
        x = features[target_idx-window:target_idx]
        y = target[target_idx]
        if target_idx < train_end:
            X_train.append(x); y_train.append(y)
        elif target_idx < val_end:
            X_val.append(x); y_val.append(y)
        else:
            X_test.append(x); y_test.append(y)
    def arr(x, shape):
        return np.asarray(x, dtype=np.float32).reshape(shape)
    n_features = features.shape[1]
    return (
        arr(X_train, (-1, window, n_features)), arr(y_train, (-1,)),
        arr(X_val, (-1, window, n_features)), arr(y_val, (-1,)),
        arr(X_test, (-1, window, n_features)), arr(y_test, (-1,))
    )

def train(args):
    set_seed(args.seed)
    out_dir = Path(args.output)
    model_dir, results_dir = out_dir / "models", out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_catalogue(args.csv, args.min_magnitude)
    df, feature_cols = make_features(raw)
    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    if train_end <= args.window or val_end <= train_end or n - val_end < 2:
        raise ValueError("Not enough events for this window and chronological split.")

    scaler = StandardScaler()
    scaler.fit(df.loc[:train_end-1, feature_cols])
    scaled_features = scaler.transform(df[feature_cols]).astype(np.float32)
    target_scaled = scaled_features[:, feature_cols.index("mag")]
    Xtr, ytr, Xv, yv, Xte, yte = make_windows(
        scaled_features, target_scaled, args.window, train_end, val_end
    )
    if min(len(Xtr), len(Xv), len(Xte)) == 0:
        raise ValueError("A split has no windows. Download more data or reduce --window.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(Xv), torch.tensor(yv)),
        batch_size=args.batch_size, shuffle=False
    )

    model = EarthquakeLSTM(
        len(feature_cols), args.hidden_size, args.layers, args.dropout
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    best_val, patience_left = float("inf"), args.patience
    history = {"train_loss": [], "val_loss": []}
    best_path = model_dir / "best_lstm.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        tr_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d}/{args.epochs} | train={tr_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val, patience_left = val_loss, args.patience
            torch.save({
                "model_state": model.state_dict(),
                "n_features": len(feature_cols),
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "window": args.window,
                "feature_cols": feature_cols,
                "mag_mean": float(scaler.mean_[feature_cols.index("mag")]),
                "mag_scale": float(scaler.scale_[feature_cols.index("mag")])
            }, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping: validation loss did not improve.")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")

    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    mag_idx = feature_cols.index("mag")
    y_true = yte * scaler.scale_[mag_idx] + scaler.mean_[mag_idx]
    y_pred = pred_scaled * scaler.scale_[mag_idx] + scaler.mean_[mag_idx]

    # Baselines: training-period mean magnitude and last magnitude in each input window.
    train_mean_scaled = float(np.mean(ytr))
    baseline_mean = np.full_like(y_true, train_mean_scaled) * scaler.scale_[mag_idx] + scaler.mean_[mag_idx]
    baseline_last = Xte[:, -1, mag_idx] * scaler.scale_[mag_idx] + scaler.mean_[mag_idx]

    def metrics(a, b):
        return {
            "MAE": float(mean_absolute_error(a, b)),
            "RMSE": float(np.sqrt(mean_squared_error(a, b))),
            "R2": float(r2_score(a, b)) if len(a) > 1 else None
        }
    report = {
        "records_after_cleaning": int(n),
        "train_windows": int(len(Xtr)),
        "validation_windows": int(len(Xv)),
        "test_windows": int(len(Xte)),
        "device": str(device),
        "LSTM": metrics(y_true, y_pred),
        "Baseline_mean": metrics(y_true, baseline_mean),
        "Baseline_last_event": metrics(y_true, baseline_last)
    }
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "actual_magnitude": y_true,
        "lstm_forecast": y_pred,
        "baseline_mean": baseline_mean,
        "baseline_last_event": baseline_last
    }).to_csv(results_dir / "test_predictions.csv", index=False)
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    plt.figure(figsize=(9, 4))
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Training and validation loss")
    plt.legend(); plt.tight_layout(); plt.savefig(results_dir / "loss_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="LSTM forecast")
    plt.xlabel("Test sample"); plt.ylabel("Magnitude"); plt.title("Actual vs forecast magnitude")
    plt.legend(); plt.tight_layout(); plt.savefig(results_dir / "actual_vs_predicted.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual magnitude"); plt.ylabel("Predicted magnitude")
    plt.title("Predicted vs actual magnitude"); plt.tight_layout()
    plt.savefig(results_dir / "prediction_scatter.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(y_true - y_pred, bins=20)
    plt.xlabel("Actual - predicted magnitude"); plt.ylabel("Frequency")
    plt.title("Forecast error distribution"); plt.tight_layout()
    plt.savefig(results_dir / "error_histogram.png", dpi=150); plt.close()

    # Forecast one next-event magnitude from the latest observed window.
    latest = scaled_features[-args.window:]
    with torch.no_grad():
        next_scaled = model(torch.tensor(latest[None, :, :], dtype=torch.float32, device=device)).item()
    next_mag = next_scaled * scaler.scale_[mag_idx] + scaler.mean_[mag_idx]
    forecast = {
        "forecast_target": "magnitude of the next catalogue event (not a date/location prediction)",
        "forecast_magnitude_estimate": float(next_mag),
        "warning": "Academic statistical estimate only; not an earthquake warning or reliable event prediction."
    }
    (results_dir / "next_event_forecast.json").write_text(json.dumps(forecast, indent=2))

    print("\nTEST METRICS")
    print(json.dumps(report, indent=2))
    print("\nNEXT-EVENT MAGNITUDE ESTIMATE")
    print(json.dumps(forecast, indent=2))
    print(f"\nSaved model: {best_path}")
    print(f"Saved plots and results in: {results_dir.resolve()}")

def forecast_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_dir = Path(args.output) / "models"
    checkpoint = torch.load(model_dir / "best_lstm.pt", map_location=device)
    scaler = joblib.load(model_dir / "feature_scaler.joblib")
    raw = load_catalogue(args.csv, args.min_magnitude)
    df, feature_cols = make_features(raw)
    if len(df) < checkpoint["window"]:
        raise ValueError("Not enough catalogue events for the saved input window.")
    scaled = scaler.transform(df[feature_cols]).astype(np.float32)
    model = EarthquakeLSTM(
        checkpoint["n_features"], checkpoint["hidden_size"],
        checkpoint["layers"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        value = model(torch.tensor(
            scaled[-checkpoint["window"]:][None, :, :],
            dtype=torch.float32, device=device
        )).item()
    idx = feature_cols.index("mag")
    magnitude = value * scaler.scale_[idx] + scaler.mean_[idx]
    print(f"Next catalogue-event magnitude estimate: {magnitude:.3f}")
    print("This is an academic estimate only, not an official warning.")
