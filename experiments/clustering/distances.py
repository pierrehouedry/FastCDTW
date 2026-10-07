"""Cross-distance matrices for 1-NN, and the 1-NN rule itself."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcdtw import cdist_fm_dtw_fit

fastcdtw_cdist = cdist_fm_dtw_fit

METHODS = ["euclidean", "dtw", "softdtw", "fastcdtw_exact", "fastcdtw_mc"]


def cdist_euclidean(X_test, X_train) -> np.ndarray:
    a = np.asarray(X_test).reshape(len(X_test), -1)
    b = np.asarray(X_train).reshape(len(X_train), -1)
    return np.sqrt(np.maximum(
        (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0.0))


def cdist_dtw(X_test, X_train, n_jobs: int = -1) -> np.ndarray:
    from tslearn.metrics import cdist_dtw as _cdist
    return np.asarray(_cdist(X_test, X_train, n_jobs=n_jobs))


def cdist_softdtw(X_test, X_train, gamma: float) -> np.ndarray:
    """Raw ``cdist_soft_dtw``: negative, not minimised at x = y -- the paper's
    SDTW column, not a bug."""
    from tslearn.metrics import cdist_soft_dtw as _cdist
    return np.asarray(_cdist(X_test, X_train, gamma=gamma))


def one_nn_predict(D: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Label of the nearest training series, row by row."""
    return np.asarray(y_train)[np.asarray(D).argmin(axis=1)]


def one_nn_accuracy(D: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    return float((one_nn_predict(D, y_train) == np.asarray(y_test)).mean())
