# sqa_fl/fl/client.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl

from sqa_fl.models import HeartMLP


def _to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    return TensorDataset(X_tensor, y_tensor)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = _to_tensor_dataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as list of numpy arrays (Flower convention)."""
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Load numpy weights into model.state_dict()."""
    state_dict = model.state_dict()
    new_state = {}
    for (k, _), w in zip(state_dict.items(), weights):
        new_state[k] = torch.tensor(w)
    model.load_state_dict(new_state)


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
) -> float:
    """Local training loop for one client."""
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


class HeartClient(fl.client.NumPyClient):
    """Flower NumPyClient for one virtual hospital."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        device: torch.device,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> None:
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

        input_dim = X_train.shape[1]
        self.model = HeartMLP(input_dim=input_dim)

        self.train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)

    # ----- Flower NumPyClient API -----

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_weights(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        set_weights(self.model, parameters)

        loss = train_local(
            self.model,
            self.train_loader,
            self.device,
            epochs=self.local_epochs,
            lr=self.lr,
        )

        new_weights = get_weights(self.model)
        num_examples = len(self.train_loader.dataset)
        metrics = {"loss": loss}
        return new_weights, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        # We’re doing global eval on the server side, so client eval is dummy.
        set_weights(self.model, parameters)
        num_examples = len(self.train_loader.dataset)
        return 0.0, num_examples, {}
