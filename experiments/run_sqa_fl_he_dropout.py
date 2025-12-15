# experiments/run_sqa_fl_he_dropout.py
from __future__ import annotations

import argparse
import random
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


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(weights)


def compute_update(
    new_weights: Dict[str, torch.Tensor],
    old_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: new_weights[k] - old_weights[k] for k in new_weights.keys()}


def flatten_update(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    flats = []
    for v in update.values():
        flats.append(v.view(-1).float())
    return torch.cat(flats)


def cosine_alignment(
    local_update: Dict[str, torch.Tensor],
    global_update_prev: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    u = flatten_update(local_update)
    g = flatten_update(global_update_prev)

    denom = (u.norm() * g.norm()).item()
    if denom < eps:
        return 1.0

    cos_sim = (u @ g).item() / denom
    score = max(0.0, min(1.0, cos_sim))
    return float(score)


def scale_update(
    update: Dict[str, torch.Tensor],
    scalar: float,
) -> Dict[str, torch.Tensor]:
    return {k: scalar * v for k, v in update.items()}


def add_updates(
    agg: Dict[str, torch.Tensor] | None,
    update: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if agg is None:
        return {k: v.clone() for k, v in update.items()}
    for k in agg.keys():
        agg[k] += update[k]
    return agg


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
) -> float:
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


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    client_parts, (X_test, y_test) = create_federated_partitions(
        num_clients=args.num_clients,
        random_state=42,
    )

    input_dim = client_parts[0][0].shape[1]

    global_model = HeartMLP(input_dim=input_dim)
    global_weights = get_weights(global_model)

    client_loaders: List[Tuple[DataLoader, int]] = []
    for X_c, y_c in client_parts:
        loader = make_loader(X_c, y_c, batch_size=args.batch_size, shuffle=True)
        client_loaders.append((loader, len(loader.dataset)))

    pubkey, privkey = generate_keypair()

    prev_global_update: Dict[str, torch.Tensor] | None = None

    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} (SQA-FL + HE + dropout) ===")

        # Choose a subset of clients to participate this round
        all_client_ids = list(range(len(client_loaders)))
        k_active = max(1, int(len(all_client_ids) * args.participation_rate))
        active_ids = sorted(random.sample(all_client_ids, k_active))
        print(f"  Active clients this round: {active_ids}")

        agg_scaled_update: Dict[str, torch.Tensor] | None = None
        enc_sum_qn = None

        for cid in active_ids:
            loader, n_i = client_loaders[cid]

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

            if prev_global_update is None:
                q_i = 1.0
            else:
                q_i = cosine_alignment(local_update, prev_global_update)

            eff_qn = q_i * float(n_i)
            scaled_update = scale_update(local_update, eff_qn)
            agg_scaled_update = add_updates(agg_scaled_update, scaled_update)

            enc_q_i = encrypt_quality(pubkey, q_i)
            enc_qn_i = enc_q_i * n_i

            if enc_sum_qn is None:
                enc_sum_qn = enc_qn_i
            else:
                enc_sum_qn = enc_sum_qn + enc_qn_i

            print(
                f"  Client {cid}: n={n_i}, loss={loss:.4f}, "
                f"quality={q_i:.3f}, eff_qn={eff_qn:.3f}"
            )

        # If no one participated (shouldn't happen), skip
        if agg_scaled_update is None:
            print("  No active clients this round, skipping update.")
            continue

        sum_qn = decrypt_sum_quality(privkey, enc_sum_qn)
        if sum_qn <= 0.0:
            sum_qn = 1.0

        global_update: Dict[str, torch.Tensor] = {}
        for k, v in agg_scaled_update.items():
            global_update[k] = v / sum_qn

        new_global_weights = {}
        for k in global_weights.keys():
            new_global_weights[k] = global_weights[k] + global_update[k]

        prev_global_update = compute_update(new_global_weights, global_weights)

        global_weights = new_global_weights
        set_weights(global_model, global_weights)

        metrics = evaluate_global(global_model, X_test, y_test, device=device)
        print(
            f"  [Round {r}] test_acc={metrics['accuracy']:.3f}, "
            f"test_auc={metrics['auc']:.3f}"
        )

    print("\n=== Final Global Performance (SQA-FL + HE + dropout) ===")
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
    parser.add_argument(
        "--participation-rate",
        type=float,
        default=0.7,  # ~30% dropout
        help="Fraction of clients that participate each round",
    )
    args = parser.parse_args()
    main(args)
