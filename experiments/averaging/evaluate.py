from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcdtw import fm_dtw, phi_from_raw
from barycenters import _as_numpy, _normalise_weights, _optimize


def _pair(X, Y, weights):
    X = _as_numpy(X[None] if np.ndim(X) < 3 else X)[0]
    Y = _as_numpy(Y)
    w = np.ones(Y.shape[0]) if weights is None else np.asarray(weights, float)
    return X, Y, w


def euclidean_objective(X, Y, weights=None) -> float:
    X, Y, w = _pair(X, Y, weights)
    return float((w * ((Y - X[None]) ** 2).sum(axis=(1, 2))).sum())


def dtw_objective(X, Y, weights=None) -> float:
    from tslearn.metrics import dtw

    X, Y, w = _pair(X, Y, weights)
    return float(sum(wi * dtw(X, y) for wi, y in zip(w, Y)))


def softdtw_objective(X, Y, weights=None, gamma: float = 1.0) -> float:
    from tslearn.metrics import soft_dtw

    X, Y, w = _pair(X, Y, weights)
    return float(sum(wi * soft_dtw(X, y, gamma=gamma) for wi, y in zip(w, Y)))


def fastcdtw_objective(X, Y, weights=None, t_y=None, lr: float = 5e-3,
                    n_steps: int = 3000, device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.float32) -> float:
    """
    sum_i w_i min_phi FastCDTW(X o phi_i, Y_i): a cold refit at a fixed
    budget, so it over-states the objective.
    """
    device = torch.device(device)
    Xn, Yn, _ = _pair(X, Y, weights)
    n, T, _ = Yn.shape

    Yt = torch.as_tensor(Yn, dtype=dtype, device=device)
    C = torch.as_tensor(Xn, dtype=dtype, device=device).unsqueeze(0)
    t_y = (torch.linspace(0.0, 1.0, T, device=device, dtype=dtype) if t_y is None
           else torch.as_tensor(t_y, dtype=dtype, device=device))
    t_c = torch.linspace(0.0, 1.0, C.shape[1], device=device, dtype=dtype)
    w = _normalise_weights(weights, n, dtype, device)

    raw = torch.zeros(n, T - 1, device=device, dtype=dtype, requires_grad=True)
    series_idx = torch.arange(n, device=device)
    cluster_idx = torch.zeros(n, dtype=torch.long, device=device)

    def dist():
        return fm_dtw(phi_from_raw(raw), t_y, C, t_c, Yt, t_y,
                      series_idx, cluster_idx)

    _optimize([raw], lambda: (w * dist()).sum(), n_steps, lr)
    with torch.no_grad():
        return float((w * dist()).sum().item())


def cross_evaluate(barycenters: dict[str, np.ndarray], Y, weights=None,
                   gamma: float = 1.0, fastcdtw_steps: int = 3000,
                   fastcdtw_lr: float = 5e-3, device: str = "cpu",
                   verbose: bool = True) -> dict[str, dict[str, float]]:
    """{method: {discrepancy: value}} for every candidate barycenter."""
    out: dict[str, dict[str, float]] = {}
    for name, X in barycenters.items():
        if verbose:
            print(f"  evaluating {name} ...", flush=True)
        out[name] = {
            "euclidean": euclidean_objective(X, Y, weights),
            "dtw": dtw_objective(X, Y, weights),
            f"softdtw(g={gamma:g})": softdtw_objective(X, Y, weights, gamma=gamma),
            "fastcdtw(refit)": fastcdtw_objective(X, Y, weights, n_steps=fastcdtw_steps,
                                            lr=fastcdtw_lr, device=device),
        }
    return out


def format_table(table: dict[str, dict[str, float]],
                 as_fitted: dict[str, float] | None = None,
                 timings: dict[str, float] | None = None) -> str:
    """as_fitted: what our solvers reached on their own warpings."""
    cols = list(next(iter(table.values())))
    if as_fitted:
        cols += ["fastcdtw(as fitted)"]
    if timings:
        cols += ["time (s)"]
    width = max(len(k) for k in table) + 2
    head = f"{'barycenter':<{width}}" + "".join(f"{c:>18}" for c in cols)
    lines = [head, "-" * len(head)]

    def value(name, col):
        if col == "fastcdtw(as fitted)":
            return (as_fitted or {}).get(name)
        if col == "time (s)":
            return (timings or {}).get(name)
        return table[name][col]

    best = {c: min([v for n in table if (v := value(n, c)) is not None] or [None])
            for c in cols}
    for name in table:
        cells = ""
        for c in cols:
            v = value(name, c)
            if v is None:
                cells += f"{'-':>18}"
            else:
                cells += (f"{v:>17.2f}" if c == "time (s)" else f"{v:>17.4f}")
                cells += "*" if v == best[c] else " "
        lines.append(f"{name:<{width}}" + cells)
    lines += [
        "",
        "* lowest in column; every method minimises its own column, so this ranks",
        "  nothing. 'fastcdtw(refit)' is a cold refit at a fixed budget and over-states",
        "  the objective; 'fastcdtw(as fitted)' is what our solvers reached and exists",
        "  only for them. 'time (s)' excludes this table's own cost.",
    ]
    return "\n".join(lines)
