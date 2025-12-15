# experiments/run_sqa_fl_he.py
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
from sqa_fl.crypto.paillier_he import (
    generate_keypair,
    encrypt_quality,
    decrypt_sum_quality,
)


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
        return 1.0

    cos_sim = (u @ g).item() / denom
    score = max(0.0, min(1.0, cos_sim))  # clip into [0, 1]
    return float(score)


def scale_update(
    update: Dict[str, torch.Tensor],
    scalar: float,
) -> Dict[str, torch.Tensor]:
    """Multiply each tensor in an update dict by scalar."""
    return {k: scalar * v for k, v in update.items()}


def add_updates(
    agg: Dict[str, torch.Tensor] | None,
    update: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Add 'update' into 'agg' element-wise; create agg if None."""
    if agg is None:
        return {k: v.clone() for k, v in update.items()}
    for k in agg.keys():
        agg[k] += update[k]
    return agg


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


# --------------------- Main SQA-FL with HE ---------------------


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

    # Generate Paillier keypair (in a real system, keys would live in a secure module)
    pubkey, privkey = generate_keypair()

    prev_global_update: Dict[str, torch.Tensor] | None = None

    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} (SQA-FL + HE) ===")

        # Aggregated scaled update (only this lives on "server")
        agg_scaled_update: Dict[str, torch.Tensor] | None = None

        # Encrypted sum of quality-weights q_i * n_i
        enc_sum_qn = None

        for cid, (loader, n_i) in enumerate(client_loaders):
            # ----- Client side -----
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
            local_update = compute_update(w_i, global_weights)

            # Quality score based on gradient alignment
            if prev_global_update is None:
                q_i = 1.0  # first round: treat all equal
            else:
                q_i = cosine_alignment(local_update, prev_global_update)

            # Effective weight for this client: q_i * n_i
            eff_qn = q_i * float(n_i)

            # Scale local update by eff_qn (this is what participates in secure aggregation)
            scaled_update = scale_update(local_update, eff_qn)

            # Add into aggregated scaled update (server sees only the sum)
            agg_scaled_update = add_updates(agg_scaled_update, scaled_update)

            # Encrypt q_i; server will use it to compute sum(q_i * n_i) homomorphically
            enc_q_i = encrypt_quality(pubkey, q_i)

            # For Paillier: E(q_i * n_i) = (E(q_i) ** n_i)
            enc_qn_i = enc_q_i * n_i

            if enc_sum_qn is None:
                enc_sum_qn = enc_qn_i
            else:
                enc_sum_qn = enc_sum_qn + enc_qn_i  # homomorphic addition via ciphertext addition

            print(
                f"  Client {cid}: n={n_i}, loss={loss:.4f}, "
                f"quality={q_i:.3f}, eff_qn={eff_qn:.3f}"
            )

        # ----- Server side: finalize aggregation -----

        # Decrypt global sum of q_i * n_i (no individual q_i are decrypted)
        sum_qn = decrypt_sum_quality(privkey, enc_sum_qn)
        if sum_qn <= 0.0:
            # Fallback: avoid divide-by-zero, use unweighted sum
            sum_qn = 1.0

        # Convert agg_scaled_update into a proper global update
        global_update: Dict[str, torch.Tensor] = {}
        for k, v in agg_scaled_update.items():
            # Divide by sum_qn to normalize
            global_update[k] = v / sum_qn

        # Apply update to global weights
        new_global_weights = {}
        for k in global_weights.keys():
            new_global_weights[k] = global_weights[k] + global_update[k]

        # Prepare for next round
        prev_global_update = compute_update(new_global_weights, global_weights)

        global_weights = new_global_weights
        set_weights(global_model, global_weights)

        # Evaluate on global test set
        metrics = evaluate_global(global_model, X_test, y_test, device=device)
        print(
            f"  [Round {r}] test_acc={metrics['accuracy']:.3f}, "
            f"test_auc={metrics['auc']:.3f}"
        )

    print("\n=== Final Global Performance (SQA-FL + HE) ===")
    final_metrics = evaluate_global(global_model, X_test, y_test, device=device)
    print(f"Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"AUC:      {final_metrics['auc']:.3f}")
