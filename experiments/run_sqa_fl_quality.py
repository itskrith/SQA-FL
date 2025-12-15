# experiments/run_sqa_fl_quality.py
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


# --------------------- Model / weight helpers ---------------------


def get_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a copy of model.state_dict() with tensors on CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Load a state_dict-like mapping into the model."""
    model.load_state_dict(weights)


def compute_update(
    new_weights: Dict[str, torch.Tensor],
    old_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return parameter-wise difference: new - old."""
    return {k: new_weights[k] - old_weights[k] for k in new_weights.keys()}


def flatten_update(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten parameter update dict into a single 1D vector."""
    flats = []
    for v in update.values():
        flats.append(v.view(-1).float())
    return torch.cat(flats)


def cosine_alignment(
    local_update: Dict[str, torch.Tensor],
    global_update_prev: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    """Cosine similarity between local and previous global update."""
    u = flatten_update(local_update)
    g = flatten_update(global_update_prev)

    denom = (u.norm() * g.norm()).item()
    if denom < eps:
        # If we can't measure (e.g., first round), treat as neutral
        return 1.0

    cos_sim = (u @ g).item() / denom
    # Robustify: clip negatives to 0 so adversarially opposite updates get no weight.
    score = max(0.0, cos_sim)
    # Also cap max at 1.0
    score = min(score, 1.0)
    return float(score)


def fedavg_quality(
    client_weights: List[Dict[str, torch.Tensor]],
    client_sizes: List[int],
    quality_scores: List[float],
) -> Dict[str, torch.Tensor]:
    """Quality-aware FedAvg: weight each client by (size * quality)."""
    # Effective weight for each client
    eff_weights = [n * max(q, 0.0) for n, q in zip(client_sizes, quality_scores)]
    total_eff = float(sum(eff_weights))
    if total_eff == 0.0:
        # Fall back to size-only FedAvg if all qualities are zero
        eff_weights = client_sizes
        total_eff = float(sum(eff_weights))

    avg: Dict[str, torch.Tensor] = {}
    keys = client_weights[0].keys()
    for k in keys:
        weighted_sum = None
        for w_i, w_eff in zip(client_weights, eff_weights):
            contrib = w_eff / total_eff * w_i[k].float()
            if weighted_sum is None:
                weighted_sum = contrib
            else:
                weighted_sum += contrib
        avg[k] = weighted_sum
    return avg


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


# --------------------- Main SQA-FL (no privacy yet) ---------------------


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simulate K virtual hospitals
    client_parts, (X_test, y_test) = create_federated_partitions(
        num_clients=args.num_clients,
        random_state=42,
    )

    input_dim = client_parts[0][0].shape[1]

    # Global model and its initial weights
    global_model = HeartMLP(input_dim=input_dim)
    global_weights = get_weights(global_model)

    # Pre-build client loaders
    client_loaders: List[Tuple[DataLoader, int]] = []
    for X_c, y_c in client_parts:
        loader = make_loader(X_c, y_c, batch_size=args.batch_size, shuffle=True)
        client_loaders.append((loader, len(loader.dataset)))

    prev_global_update: Dict[str, torch.Tensor] | None = None

    # Training rounds
    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} ===")

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_sizes: List[int] = []
        quality_scores: List[float] = []

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

            # Compute local update and quality score
            local_update = compute_update(w_i, global_weights)
            if prev_global_update is None:
                q_i = 1.0  # first round: no previous update, treat all equal
            else:
                q_i = cosine_alignment(local_update, prev_global_update)

            quality_scores.append(q_i)

            print(
                f"  Client {cid}: n={n_i}, loss={loss:.4f}, "
                f"quality={q_i:.3f}"
            )

        # Aggregate on server using quality-aware FedAvg
        new_global_weights = fedavg_quality(client_weights, client_sizes, quality_scores)

        # Compute global update for this round (for next round's alignment)
        prev_global_update = compute_update(new_global_weights, global_weights)

        # Update global weights/model
        global_weights = new_global_weights
        set_weights(global_model, global_weights)

        # Evaluate on global test set
        metrics = evaluate_global(global_model, X_test, y_test, device=device)
        print(
            f"  [Round {r}] test_acc={metrics['accuracy']:.3f}, "
            f"test_auc={metrics['auc']:.3f}"
        )

    print("\n=== Final Global Performance (SQA quality-weighted) ===")
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
