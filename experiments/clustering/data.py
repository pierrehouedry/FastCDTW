"""UCR splits and cross-validation folds for the 1-NN experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_ucr_split(name: str, znorm: str = "none"):
    """(X_train, y_train, X_test, y_test, class_names) from tslearn's UCR cache."""
    from tslearn.datasets import UCR_UEA_datasets

    X_tr, y_tr, X_te, y_te = UCR_UEA_datasets().load_dataset(name)
    if X_tr is None:
        raise RuntimeError("tslearn returned nothing (download failed / unknown name)")
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)

    classes = sorted(set(y_tr.tolist()) | set(y_te.tolist()), key=str)
    lut = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([lut[v] for v in y_tr.tolist()], dtype=int)
    y_te = np.array([lut[v] for v in y_te.tolist()], dtype=int)

    if znorm == "series":
        for X in (X_tr, X_te):
            mu = X.mean(axis=1, keepdims=True)
            sd = np.maximum(X.std(axis=1, keepdims=True), 1e-8)
            X[:] = (X - mu) / sd
    elif znorm != "none":
        raise ValueError(f"unknown znorm: {znorm}")

    return X_tr, y_tr, X_te, y_te, [str(c) for c in classes]


def dataset_names(json_path: str | None = None) -> list[str]:
    """Every univariate UCR dataset, or the 'datasets' key of a JSON file."""
    if json_path:
        return list(json.loads(Path(json_path).read_text())["datasets"])
    from tslearn.datasets import UCR_UEA_datasets
    return sorted(UCR_UEA_datasets().list_univariate_datasets())


def cv_splits(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random 2/3-1/3 (fit, held-out) splits - the softdtw gamma protocol."""
    rng = np.random.default_rng(seed)
    cut = max(int(round(2 * n / 3)), 1)
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        fit, held = perm[:cut], perm[cut:]
        if fit.size and held.size:
            splits.append((fit, held))
    return splits
