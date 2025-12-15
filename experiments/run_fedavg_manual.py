# experiments/run_fedavg_manual.py
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

from sqa_fl.data import create_federated_partitions
from sqa_fl.models import HeartMLP


# --------------------- Data helpers ---------------------


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------- Model helpers ---------------------


def get_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a copy of model.state_dict() with tensors moved to CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Load a state_dict-like mapping into the model."""
    model.load_state_dict(weights)


def fedavg(
    client_weights: List[Dict[str, torch.Tensor]],
    client_sizes: List[int],
) -> Dict[str, torch.Tensor]:
    """Standard FedAvg: weighted average of client weights."""
    total_size = float(sum(client_sizes))
    avg_weights: Dict[str, torch.Tensor] = {}

    # assume all clients share same keys
    keys = client_weights[0].keys()
    for k in keys:
        weighted_sum = None
        for w, n_i in zip(client_weights, client_sizes):
            w_k = w[k].float()
            weight = n_i / total_size
            contrib = weight * w_k
            if weighted_sum is None:
                weighted_sum = contrib
            else:
                weighted_sum += contrib
        avg_weights[k] = weighted_sum

    return avg_weights


# --------------------- Training & evaluation ---------------------


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
) -> float:
    """Local training loop for one virtual hospital."""
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    total_loss, n_samples = 0.0, 0

    for _ in range(epochs):
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            opt.zero_grad()
            logits = model(X_b)
            loss = crit(logits, y_b)
            loss.backward()
            opt.step()

            bs = X_b.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_global(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    model.eval()

    loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)

    all_logits, all_labels = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        logits = model(X_b)
        all_logits.append(logits.cpu())
        all_labels.append(y_b)

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


# --------------------- Main FedAvg loop ---------------------


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simulate K virtual hospitals
    client_parts, (X_test, y_test) = create_federated_partitions(
        num_clients=args.num_clients,
        random_state=42,
    )

    input_dim = client_parts[0][0].shape[1]

    # Global model
    global_model = HeartMLP(input_dim=input_dim)
    global_weights = get_weights(global_model)

    # Pre-build loaders for each client
    client_loaders: List[Tuple[DataLoader, int]] = []
    for X_c, y_c in client_parts:
        loader = make_loader(X_c, y_c, batch_size=args.batch_size, shuffle=True)
        client_loaders.append((loader, len(loader.dataset)))

    # Training rounds
    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} ===")

        client_weights = []
        client_sizes = []

        # Each client trains on its local data starting from current global weights
        for cid, (loader, n_i) in enumerate(client_loaders):
            local_model = HeartMLP(input_dim=input_dim)
            set_weights(local_model, global_weights)

            loss = train_local(
                model=local_model,
                loader=loader,
                device=device,
                epochs=args.local_epochs,
                lr=args.lr,
            )

            w_i = get_weights(local_model)
            client_weights.append(w_i)
            client_sizes.append(n_i)

            print(f"  Client {cid}: n={n_i}, local_loss={loss:.4f}")

        # Server: aggregate client weights using FedAvg
        global_weights = fedavg(client_weights, client_sizes)
        set_weights(global_model, global_weights)

        # Evaluate on global test set
        metrics = evaluate_global(global_model, X_test, y_test, device=device)
        print(
            f"  [Round {r}] test_acc={metrics['accuracy']:.3f}, "
            f"test_auc={metrics['auc']:.3f}"
        )

    print("\n=== Final Global Performance ===")
    final_metrics = evaluate_global(global_model, X_test, y_test, device=device)
    print(f"Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"AUC:      {final_metrics['auc']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
