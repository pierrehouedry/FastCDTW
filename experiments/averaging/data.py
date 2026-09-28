from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_FILE = DATA_DIR / "toy" / "toy_rate.arff"


def load(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read a flat ARFF. Returns (N, T, 1) series and (N,) int labels."""
    path = Path(path) if path is not None else DEFAULT_FILE
    rows, in_data = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if in_data:
                if line:
                    rows.append(line.split(","))
            elif line.lower().startswith("@data"):
                in_data = True
    if not rows:
        raise ValueError(f"no @data rows in {path}")
    arr = np.array(rows)                       # raises on ragged rows
    return arr[:, :-1].astype(np.float64)[:, :, None], arr[:, -1].astype(float).astype(int)


def sample_series(X: np.ndarray, labels: np.ndarray, n: int = 10, seed: int = 0,
                  klass: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """n series drawn uniformly without replacement, as (Y, idx)."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(X)) if klass is None else np.flatnonzero(labels == klass)
    if pool.size < n:
        raise ValueError(f"asked for {n} series, only {pool.size} available")
    idx = np.sort(rng.choice(pool, size=n, replace=False))
    return X[idx], idx
