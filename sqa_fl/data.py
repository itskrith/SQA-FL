# sqa_fl/data.py
from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00519/heart_failure_clinical_records_dataset.csv"
)


def download_heart_failure() -> pathlib.Path:
    """Download the heart failure dataset if not present."""
    import urllib.request

    csv_path = DATA_DIR / "heart_failure_clinical_records.csv"
    if csv_path.exists():
        return csv_path

    print(f"Downloading dataset to {csv_path} ...")
    urllib.request.urlretrieve(DATA_URL, csv_path)
    print("Download complete.")
    return csv_path


def load_heart_failure(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return standardized train/val/test splits.
    X_* shape: (n_samples, n_features), y_* shape: (n_samples,)
    """
    csv_path = download_heart_failure()
    df = pd.read_csv(csv_path)

    # target column in this dataset
    y = df["DEATH_EVENT"].values.astype(np.float32)
    X = df.drop(columns=["DEATH_EVENT"]).values.astype(np.float32)

    # train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )

    # val / test split from temp
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - rel_val, random_state=random_state, stratify=y_temp
    )

    # standardize features using training stats only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_federated_partitions(
    num_clients: int = 5,
    random_state: int = 42,
):
    """
    Create `num_clients` non-overlapping data partitions (virtual hospitals)
    plus a separate global test set.

    Returns:
        client_partitions: List[(X_client, y_client)]
        test_data: (X_test, y_test)
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_heart_failure(
        test_size=0.2,
        val_size=0.1,
        random_state=random_state,
    )

    # Combine train+val into a pool, then split into K shards
    X_pool = np.concatenate([X_train, X_val], axis=0)
    y_pool = np.concatenate([y_train, y_val], axis=0)

    skf = StratifiedKFold(
        n_splits=num_clients,
        shuffle=True,
        random_state=random_state,
    )

    client_partitions = []
    for _, idx in skf.split(X_pool, y_pool):
        X_client = X_pool[idx]
        y_client = y_pool[idx]
        client_partitions.append((X_client, y_client))

    return client_partitions, (X_test, y_test)