"""Barycenter solvers, all alternating an E-step on the warpings with an
M-step on the centroid:

  fastcdtw_vec     free vector, exact integral
  fastcdtw_vec_mc  free vector, Monte Carlo integral
  fastcdtw_inr     implicit neural, Monte Carlo integral
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcdtw import (CumSumWarping, fm_dtw, fm_dtw_mc, interp_batch,  # noqa: E402
                      phi_from_raw)


def _as_numpy(Y) -> np.ndarray:
    """(N, T) or (N, T, d) -> float64 (N, T, d)."""
    Y = Y.detach().cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
    if Y.ndim == 2:
        Y = Y[:, :, None]
    if Y.ndim != 3:
        raise ValueError(f"expected (N, T) or (N, T, d), got {Y.shape}")
    return Y.astype(np.float64)


def _normalise_weights(weights, n: int, dtype, device) -> torch.Tensor:
    if weights is None:
        return torch.ones(n, dtype=dtype, device=device)
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64),
                        dtype=dtype, device=device).reshape(-1)
    if w.numel() != n:
        raise ValueError(f"weights has {w.numel()} entries, expected {n}")
    return w


def _optimize(params, loss_fn, n_steps: int, lr: float) -> list[float]:
    """n_steps Adam steps on params; returns the loss at every step."""
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
    return losses


def check_monotone(trace: list[float], rtol: float = 1e-6) -> tuple[bool, float]:
    """Largest relative increase along the trace."""
    worst = 0.0
    for prev, cur in zip(trace[:-1], trace[1:]):
        worst = max(worst, (cur - prev) / max(abs(prev), 1e-12))
    return worst <= rtol, worst


def euclidean_barycenter(Y, weights=None) -> np.ndarray:
    """Weighted arithmetic mean."""
    Y = _as_numpy(Y)
    w = np.ones(Y.shape[0]) if weights is None else np.asarray(weights, float)
    return np.tensordot(w / w.sum(), Y, axes=(0, 0))


def dtw_barycenter(Y, weights=None, max_iter: int = 50, tol: float = 1e-5,
                   init: np.ndarray | None = None) -> np.ndarray:
    """DBA (tslearn), from the Euclidean mean."""
    from tslearn.barycenters import dtw_barycenter_averaging

    Y = _as_numpy(Y)
    init = euclidean_barycenter(Y, weights) if init is None else init
    return np.asarray(dtw_barycenter_averaging(
        Y, weights=weights, init_barycenter=init, max_iter=max_iter, tol=tol))


def softdtw_barycenter(Y, weights=None, gamma: float = 1.0, max_iter: int = 200,
                       tol: float = 1e-5, init: np.ndarray | None = None) -> np.ndarray:
    """Soft-DTW barycenter (tslearn, L-BFGS), from the Euclidean mean."""
    from tslearn.barycenters import softdtw_barycenter as _softdtw

    Y = _as_numpy(Y)
    init = euclidean_barycenter(Y, weights) if init is None else init
    return np.asarray(_softdtw(Y, weights=weights, gamma=gamma, init=init,
                               max_iter=max_iter, tol=tol))


def _one_series(C) -> np.ndarray:
    """A single series, (T,), (T, d) or (1, T, d) -> float64 (T, d)."""
    C = C.detach().cpu().numpy() if isinstance(C, torch.Tensor) else np.asarray(C)
    C = C.astype(np.float64)
    if C.ndim == 3:
        C = C[0]
    return C[:, None] if C.ndim == 1 else C


def phi_from_dtw_path(C, Y) -> np.ndarray:
    """(N, T) warpings read off the DTW paths between C and each Y_i: phi_i(t_k)
    is the mean centroid time matched to t_k (series time -> centroid time)."""
    from tslearn.metrics import dtw_path

    C, Y = _one_series(C), _as_numpy(Y)
    t_c = np.linspace(0.0, 1.0, C.shape[0])
    phi = np.empty(Y.shape[:2])
    for i, y in enumerate(Y):
        path, _ = dtw_path(C, y)
        acc, cnt = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        for j, k in path:
            acc[k] += t_c[j]
            cnt[k] += 1
        phi[i] = acc / cnt
    return phi


def raw_from_phi(phi, floor: float = 1e-3) -> np.ndarray:
    """Inverse of ``phi_from_raw`` (softplus increments, then normalised), scaled so
    that the identity maps to raw = 0; increments are floored at ``floor`` times
    the identity's so that flat stretches of a DTW path stay trainable."""
    phi = np.asarray(phi, dtype=np.float64)
    inc = np.diff(phi, axis=-1) * (phi.shape[-1] - 1) * np.log(2.0)
    inc = np.maximum(inc, floor * np.log(2.0))
    return np.log(np.expm1(inc))


def _phi_on(init_phi, n: int) -> np.ndarray:
    """(N, T) warpings on a uniform grid -> (N, n) on a uniform grid."""
    init_phi = np.asarray(init_phi, dtype=np.float64)
    if init_phi.shape[-1] == n:
        return init_phi
    t_old, t_new = np.linspace(0, 1, init_phi.shape[-1]), np.linspace(0, 1, n)
    return np.stack([np.interp(t_new, t_old, p) for p in init_phi])


def dba_warm_start(Y, weights=None, max_iter: int = 50):
    """DBA barycenter (from the Euclidean mean) and its DTW warpings."""
    C = dtw_barycenter(Y, weights=weights, max_iter=max_iter)
    return C, phi_from_dtw_path(C, Y)


class INRCentroid(nn.Module):
    """Fourier-feature MLP, t in [0, 1] -> R^d. ``n_freq = 0`` feeds t raw."""

    def __init__(self, d_out: int = 1, n_freq: int = 8, hidden: int = 64,
                 n_layers: int = 5):
        super().__init__()
        d_in = 2 * n_freq if n_freq > 0 else 1
        layers = [nn.Linear(d_in, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, d_out))
        self.net = nn.Sequential(*layers)
        self.n_freq = n_freq

    def features(self, t: torch.Tensor) -> torch.Tensor:
        """(P,) -> (P, 2 n_freq), or (P, 1) when n_freq = 0."""
        if self.n_freq == 0:
            return t[:, None]
        freqs = 2 ** torch.arange(self.n_freq, dtype=t.dtype, device=t.device)
        proj = t[:, None] * freqs[None, :] * torch.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(next(self.parameters()).device)
        return self.net(self.features(t))


class WarpingBank:
    """N independent CumSumWarping(n), stacked and evaluated under vmap."""

    def __init__(self, size: int, n: int, device):
        modules = [CumSumWarping(n).to(device) for _ in range(size)]
        params, buffers = stack_module_state(modules)
        self.params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
        self.buffers = {k: v.detach().clone() for k, v in buffers.items()}
        self.base = copy.deepcopy(modules[0]).to("meta")

    def parameters(self) -> list[torch.Tensor]:
        return list(self.params.values())

    def forward(self, idx: torch.Tensor, t: torch.Tensor,
                detach: bool = False) -> torch.Tensor:
        """t (P, M) -> (P, M), row j evaluated with warping idx[j]."""
        p = {k: (v[idx].detach() if detach else v[idx]) for k, v in self.params.items()}
        b = {k: v[idx] for k, v in self.buffers.items()}
        out = torch.vmap(lambda p_, b_, t_: functional_call(self.base, (p_, b_), (t_,)))(p, b, t)
        return out.clamp(0.0, 1.0).to(t.dtype)


def _fit_inr_to_series(centroid: INRCentroid, series: torch.Tensor,
                       times: torch.Tensor, n_steps: int = 2000,
                       lr: float = 1e-2) -> float:
    """Regress the INR onto a fixed series. Returns the final MSE."""
    opt = torch.optim.Adam(centroid.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss = ((centroid(times) - series) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss.item())


def _mc_distances(bank: WarpingBank, centroid: INRCentroid, Y: torch.Tensor,
                  t_y: torch.Tensor, n_mc: int, device: torch.device,
                  detach_phi: bool = False) -> torch.Tensor:
    """(N,) Monte Carlo FastCDTW, centroid against each series."""
    idx = torch.arange(Y.shape[0], device=device)
    return fm_dtw_mc(
        x=lambda p: centroid(p.reshape(-1)).reshape(*p.shape, -1),
        Y=Y, t_y=t_y,
        phi=lambda ts: bank.forward(idx, ts, detach=detach_phi),
        n_mc=n_mc)


def fastcdtw_inr_barycenter(Y, t_y=None, weights=None, lr: float = 5e-3,
                         n_steps: int = 500, n_outer: int = 15, n_mc: int = 200,
                         n_eval: int = 20, c_hidden: int = 64, c_freq: int = 8,
                         c_layers: int = 5, warp_n: int = 0,
                         init_steps: int = 2000, seed: int = 42,
                         device: str | torch.device = "cpu", verbose: bool = True,
                         init=None, init_phi=None):
    """FastCDTW barycenter, INR centroid, Monte Carlo integral. ``init`` / ``init_phi``:
    centroid and (N, T) warpings to start from (default: Euclidean mean, identity)."""
    device = torch.device(device)
    torch.manual_seed(seed)

    Yn = _as_numpy(Y)
    n, T, d = Yn.shape
    warp_n = warp_n or T
    Yt = torch.as_tensor(Yn, dtype=torch.float32, device=device)
    t_y = (torch.linspace(0.0, 1.0, T, device=device) if t_y is None
           else torch.as_tensor(t_y, dtype=torch.float32, device=device))
    w = _normalise_weights(weights, n, Yt.dtype, device)

    centroid = INRCentroid(d, c_freq, c_hidden, c_layers).to(device)
    bank = WarpingBank(n, warp_n, device)
    if init_phi is not None:
        bank.params["a"] = torch.as_tensor(raw_from_phi(_phi_on(init_phi, warp_n)),
                                           dtype=torch.float32,
                                           device=device).requires_grad_(True)
    tag = "inr" if c_freq > 0 else "mlp"

    target = torch.as_tensor(euclidean_barycenter(Yn, weights) if init is None
                             else _one_series(init),
                             dtype=torch.float32, device=device)
    init_mse = _fit_inr_to_series(centroid, target, t_y, n_steps=init_steps)
    if verbose:
        print(f"  [{tag}] init fit to Euclidean mean: mse={init_mse:.3e}", flush=True)

    def freeze(module, flag: bool):
        for p in module.parameters():
            p.requires_grad_(flag)

    def objective_value(n_draws: int) -> float:
        with torch.enable_grad():
            draws = torch.stack([
                _mc_distances(bank, centroid, Yt, t_y, n_mc, device).detach()
                for _ in range(n_draws)]).mean(dim=0)
        return float((w * draws).sum().item())

    objective = [objective_value(n_eval)]
    if verbose:
        print(f"  [{tag}] objective at init: {objective[0]:.6f}", flush=True)

    for it in range(n_outer):
        freeze(centroid, False)
        _optimize(bank.parameters(),
                  lambda: (w * _mc_distances(bank, centroid, Yt, t_y, n_mc, device)).sum(),
                  n_steps, lr)
        objective.append(objective_value(n_eval))

        freeze(centroid, True)
        _optimize(list(centroid.parameters()),
                  lambda: (w * _mc_distances(bank, centroid, Yt, t_y, n_mc, device,
                                             detach_phi=True)).mean(),
                  n_steps, lr)
        objective.append(objective_value(n_eval))

        if verbose:
            print(f"    iter {it+1:2d}/{n_outer}: after E={objective[-2]:.6f}  "
                  f"after M={objective[-1]:.6f}", flush=True)

    freeze(centroid, False)
    with torch.no_grad():
        X = centroid(t_y).cpu().numpy()
        phi = bank.forward(torch.arange(n, device=device),
                           t_y.unsqueeze(0).expand(n, -1).contiguous()).cpu().numpy()

    ok, worst = check_monotone(objective, rtol=5e-2)
    info = {"objective": objective, "phi": phi, "t_phi": t_y.cpu().numpy(),
            "init_mse": init_mse,
            "init_label": "Euclidean mean" if init is None else "given",
            "monotone": ok, "worst_increase": worst,
            "solver": f"fastcdtw_{tag}", "c_freq": c_freq, "c_hidden": c_hidden,
            "c_layers": c_layers, "centroid": centroid, "bank": bank}
    return X, info


def fastcdtw_vector_barycenter(Y, t_y=None, weights=None, lr: float = 1e-3,
                            n_steps: int = 500, n_outer: int = 15, c_n: int = 0,
                            warp_n: int = 0, seed: int = 42, dtype=torch.float32,
                            device: str | torch.device = "cpu", chunk: int = 0,
                            verbose: bool = True, init=None, init_phi=None):
    """FastCDTW barycenter, free-vector centroid, exact integral; ``warp_n``
    phi knots (0 = one per timestamp), one per sample reproducing a DTW
    staircase. ``init`` / ``init_phi``: centroid and (N, T) warpings to start
    from (default: Euclidean mean, identity).
    """
    device = torch.device(device)
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    torch.manual_seed(seed)

    Yn = _as_numpy(Y)
    n, T, d = Yn.shape
    Yt = torch.as_tensor(Yn, dtype=dtype, device=device)
    t_y = (torch.linspace(0.0, 1.0, T, device=device, dtype=dtype) if t_y is None
           else torch.as_tensor(t_y, dtype=dtype, device=device))
    w = _normalise_weights(weights, n, dtype, device)

    t_phi = (t_y if not warp_n
             else torch.linspace(0.0, 1.0, warp_n, device=device, dtype=dtype))
    t_c = t_y if not c_n else torch.linspace(0.0, 1.0, c_n, device=device, dtype=dtype)

    C0 = euclidean_barycenter(Yn, weights) if init is None else _one_series(init)
    C = torch.as_tensor(C0, dtype=dtype, device=device).unsqueeze(0)
    if c_n:
        C = interp_batch(t_y, C, t_c.unsqueeze(0))
    C = C.clone().requires_grad_(True)

    if init_phi is None:
        raw = torch.zeros(n, len(t_phi) - 1, device=device, dtype=dtype,
                          requires_grad=True)
    else:
        raw = torch.as_tensor(raw_from_phi(_phi_on(init_phi, len(t_phi))),
                              dtype=dtype, device=device).requires_grad_(True)
    series_idx = torch.arange(n, device=device)
    cluster_idx = torch.zeros(n, dtype=torch.long, device=device)

    def dist(detach_phi: bool = False, detach_C: bool = False) -> torch.Tensor:
        phi = phi_from_raw(raw)
        if detach_phi:
            phi = phi.detach()
        Cc = C.detach() if detach_C else C
        if not chunk or n <= chunk:
            return fm_dtw(phi, t_phi, Cc, t_c, Yt, t_y, series_idx, cluster_idx)
        return torch.cat([
            fm_dtw(phi[s:s + chunk], t_phi, Cc, t_c, Yt, t_y,
                   series_idx[s:s + chunk], cluster_idx[s:s + chunk])
            for s in range(0, n, chunk)])

    def objective_value() -> float:
        with torch.no_grad():
            return float((w * dist()).sum().item())

    objective = [objective_value()]
    if verbose:
        print(f"  [vec] objective at init: {objective[0]:.6f}", flush=True)

    for it in range(n_outer):
        _optimize([raw], lambda: (w * dist(detach_C=True)).sum(), n_steps, lr)
        objective.append(objective_value())
        _optimize([C], lambda: (w * dist(detach_phi=True)).sum(), n_steps, lr)
        objective.append(objective_value())
        if verbose:
            print(f"    iter {it+1:2d}/{n_outer}: after E={objective[-2]:.6f}  "
                  f"after M={objective[-1]:.6f}", flush=True)

    with torch.no_grad():
        X = C[0].cpu().numpy()
        phi = phi_from_raw(raw).cpu().numpy()

    ok, worst = check_monotone(objective)
    if verbose and not ok:
        print(f"  [vec] objective increased by {worst:.2%} at some step -- "
              f"lower lr or fewer inner steps", flush=True)
    info = {"objective": objective, "phi": phi, "t_phi": t_phi.cpu().numpy(),
            "monotone": ok, "worst_increase": worst, "t_c": t_c.cpu().numpy(),
            "solver": "fastcdtw_vec"}
    return X, info


def fastcdtw_vec_mc_barycenter(Y, t_y=None, weights=None, lr: float = 1e-3,
                            n_steps: int = 500, n_outer: int = 15,
                            n_mc: int = 200, n_eval: int = 20, c_n: int = 0,
                            warp_n: int = 0, seed: int = 42,
                            dtype=torch.float32,
                            device: str | torch.device = "cpu",
                            verbose: bool = True):
    """FastCDTW barycenter, free-vector centroid, Monte Carlo integral."""
    device = torch.device(device)
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    torch.manual_seed(seed)

    Yn = _as_numpy(Y)
    n, T, d = Yn.shape
    Yt = torch.as_tensor(Yn, dtype=dtype, device=device)
    t_y = (torch.linspace(0.0, 1.0, T, device=device, dtype=dtype) if t_y is None
           else torch.as_tensor(t_y, dtype=dtype, device=device))
    w = _normalise_weights(weights, n, dtype, device)

    warp_n = warp_n or T
    t_phi = torch.linspace(0.0, 1.0, warp_n, device=device, dtype=dtype)
    t_c = t_y if not c_n else torch.linspace(0.0, 1.0, c_n, device=device, dtype=dtype)

    C0 = euclidean_barycenter(Yn, weights)
    C = torch.as_tensor(C0, dtype=dtype, device=device).unsqueeze(0)
    if c_n:
        C = interp_batch(t_y, C, t_c.unsqueeze(0))
    C = C.clone().requires_grad_(True)

    raw = torch.zeros(n, warp_n - 1, device=device, dtype=dtype,
                      requires_grad=True)

    def dist(detach_phi: bool = False, detach_C: bool = False) -> torch.Tensor:
        phi = phi_from_raw(raw)
        if detach_phi:
            phi = phi.detach()
        Cc = C.detach() if detach_C else C

        def centroid(p: torch.Tensor) -> torch.Tensor:
            return interp_batch(t_c, Cc.expand(p.shape[0], -1, -1), p)

        return fm_dtw_mc(x=centroid, Y=Yt, t_y=t_y,
                         phi=lambda ts: interp_batch(t_phi, phi, ts),
                         n_mc=n_mc)

    def objective_value(n_draws: int) -> float:
        with torch.enable_grad():
            draws = torch.stack([dist().detach() for _ in range(n_draws)]).mean(dim=0)
        return float((w * draws).sum().item())

    objective = [objective_value(n_eval)]
    if verbose:
        print(f"  [vec-mc] objective at init: {objective[0]:.6f}", flush=True)

    for it in range(n_outer):
        _optimize([raw], lambda: (w * dist(detach_C=True)).sum(), n_steps, lr)
        objective.append(objective_value(n_eval))
        _optimize([C], lambda: (w * dist(detach_phi=True)).sum(), n_steps, lr)
        objective.append(objective_value(n_eval))
        if verbose:
            print(f"    iter {it+1:2d}/{n_outer}: after E={objective[-2]:.6f}  "
                  f"after M={objective[-1]:.6f}", flush=True)

    with torch.no_grad():
        X = C[0].cpu().numpy()
        if c_n:
            X = interp_batch(t_c, C.detach(), t_y.unsqueeze(0))[0].cpu().numpy()
        phi = interp_batch(t_phi, phi_from_raw(raw),
                           t_y.unsqueeze(0).expand(n, -1).contiguous()).cpu().numpy()

    ok, worst = check_monotone(objective, rtol=5e-2)
    info = {"objective": objective, "phi": phi, "t_phi": t_y.cpu().numpy(),
            "monotone": ok, "worst_increase": worst, "t_c": t_c.cpu().numpy(),
            "solver": "fastcdtw_vec_mc"}
    return X, info
