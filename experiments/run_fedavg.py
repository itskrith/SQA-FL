# experiments/run_fedavg.py
from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

import flwr as fl

from sqa_fl.data import create_federated_partitions
from sqa_fl.fl.client import HeartClient, get_weights, set_weights
from sqa_fl.models import HeartMLP


def _make_test_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_on_data(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    model.eval()

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


def get_evaluate_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    device: torch.device,
) -> Callable[[int, List[np.ndarray], Dict], Tuple[float, Dict]]:
    test_loader = _make_test_loader(X_test, y_test)

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, Dict]:
        model = HeartMLP(input_dim=input_dim)
        set_weights(model, parameters)
        metrics = evaluate_on_data(model, test_loader, device=device)
        print(
            f"[Round {server_round:02d}] "
            f"test_acc={metrics['accuracy']:.3f} "
            f"test_auc={metrics['auc']:.3f}"
        )
        return 0.0, metrics  # loss, metrics

    return evaluate_fn


def make_client_fn(
    client_partitions,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
):
    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        X_c, y_c = client_partitions[cid_int]
        return HeartClient(
            X_train=X_c,
            y_train=y_c,
            device=device,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
        )

    return client_fn


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clients = args.num_clients
    client_parts, (X_test, y_test) = create_federated_partitions(
        num_clients=num_clients,
        random_state=42,
    )

    input_dim = client_parts[0][0].shape[1]

    eval_fn = get_evaluate_fn(
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
        device=device,
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=eval_fn,
    )

    client_fn = make_client_fn(
        client_partitions=client_parts,
        device=device,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
