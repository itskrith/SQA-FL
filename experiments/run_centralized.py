# experiments/run_centralized.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

from sqa_fl.data import load_heart_failure
from sqa_fl.models import HeartMLP


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        all_logits.append(logits.cpu())
        all_labels.append(y_batch)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "auc": auc}


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val, X_test, y_test = load_heart_failure()

    input_dim = X_train.shape[1]
    model = HeartMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=256, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics_val = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_loss:.4f} val_acc={metrics_val['accuracy']:.3f} "
            f"val_auc={metrics_val['auc']:.3f}"
        )

        if metrics_val["auc"] > best_val_auc:
            best_val_auc = metrics_val["auc"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_test = evaluate(model, test_loader, device)
    print("\n=== Centralized Test Performance ===")
    print(f"Accuracy: {metrics_test['accuracy']:.3f}")
    print(f"AUC:      {metrics_test['auc']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()
    main(args)
