"""
Cross-distance matrix with one warping fitted per pair.
"""

from __future__ import annotations

import numpy as np
import torch

from .interp import interp_batch
from .loss import fm_dtw, fm_dtw_mc
from .warping import phi_from_raw

__all__ = ["as_tensor", "cdist_fm_dtw_fit"]


def as_tensor(X, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """(n, T) or (n, T, d) array -> (n, T, d) tensor."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, :, None]
    return torch.as_tensor(X, dtype=dtype, device=device)


def _exact(raw, t, train, test, test_idx, train_idx):
    return fm_dtw(phi_from_raw(raw), t, train, t, test, t,
                  series_idx=test_idx, cluster_idx=train_idx)


def _mc(raw, t, train, test, test_idx, train_idx, n_mc, generator):
    phi_knots = phi_from_raw(raw)
    C = train[train_idx]
    return fm_dtw_mc(x=lambda p: interp_batch(t, C, p),
                     Y=test[test_idx], t_y=t,
                     phi=lambda ts: interp_batch(t, phi_knots, ts),
                     n_mc=n_mc, generator=generator)


def cdist_fm_dtw_fit(X_test,
                     X_train,
                     exact: bool = True,
                     n_steps: int = 400,
                     lr: float = 1e-2,
                     n_mc: int = 800,
                     n_eval: int = 20,
                     chunk: int = 0,
                     device: str | torch.device = "cpu",
                     dtype: str | torch.dtype = torch.float32,
                     seed: int = 42,
                     verbose: bool = False) -> np.ndarray:
    """
    (n_test, n_train) FastCDTW distances, one warping fit per pair
    """
    device = torch.device(device)
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    te = as_tensor(X_test, dtype, device)
    tr = as_tensor(X_train, dtype, device)
    if te.shape[1] != tr.shape[1]:
        raise ValueError(f"length mismatch: test {te.shape[1]}, train {tr.shape[1]}")

    n_test, T, _ = te.shape
    n_train = tr.shape[0]
    t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    n_pairs = n_test * n_train
    pairs = torch.arange(n_pairs, device=device)
    test_idx_all, train_idx_all = pairs // n_train, pairs % n_train
    size = chunk if chunk else n_pairs

    out = torch.empty(n_pairs, device=device, dtype=dtype)
    for start in range(0, n_pairs, size):
        stop = min(start + size, n_pairs)
        test_idx, train_idx = test_idx_all[start:stop], train_idx_all[start:stop]

        raw = torch.zeros(stop - start, T - 1, device=device, dtype=dtype,
                          requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            loss = (_exact(raw, t, tr, te, test_idx, train_idx) if exact else
                    _mc(raw, t, tr, te, test_idx, train_idx, n_mc,
                        generator))
            loss.sum().backward()
            opt.step()
            if verbose and (step + 1) % max(n_steps // 4, 1) == 0:
                print(f"      pairs {start}:{stop}  step {step + 1}/{n_steps}  "
                      f"mean d={float(loss.mean()):.5f}", flush=True)

        if exact:
            with torch.no_grad():
                out[start:stop] = _exact(raw, t, tr, te, test_idx, train_idx)
        else:
            # fresh draws, so the minimum is not scored on the sample that found it
            with torch.enable_grad():
                draws = torch.stack([
                    _mc(raw, t, tr, te, test_idx, train_idx, n_mc,
                        generator).detach()
                    for _ in range(n_eval)])
            out[start:stop] = draws.mean(dim=0)

    return out.reshape(n_test, n_train).detach().cpu().numpy()
