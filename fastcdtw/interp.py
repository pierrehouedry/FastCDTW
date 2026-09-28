"""Piecewise-linear interpolation on sorted knots."""

import torch

EPS = 1e-12


def dedup_times(t: torch.Tensor) -> torch.Tensor:
    """Drop t[i] when t[i+1] - t[i] <= EPS. Keeps the last point. t: (N,)."""
    keep = torch.cat([t[1:] - t[:-1] > EPS, t.new_ones(1, dtype=torch.bool)])
    return t[keep]


def _bracket(t_known: torch.Tensor, t_query: torch.Tensor):
    """Left index and local weight of each query point. Both 1D."""
    n = t_known.shape[0]
    idx = torch.searchsorted(t_known.contiguous(),
                             t_query.detach().contiguous(), right=True)
    i0 = (idx - 1).clamp(0, n - 2)
    t0, t1 = t_known[i0], t_known[i0 + 1]
    return i0, (t_query - t0) / (t1 - t0 + EPS)


def interp(t_known: torch.Tensor, v_known: torch.Tensor,
           t_query: torch.Tensor) -> torch.Tensor:
    """t_known (N,) sorted, v_known (..., N), t_query (M,) -> (..., M)."""
    i0, w = _bracket(t_known, t_query)
    v0 = v_known.index_select(-1, i0)
    v1 = v_known.index_select(-1, i0 + 1)
    return v0 + w * (v1 - v0)


def interp_series(t_known: torch.Tensor, v_known: torch.Tensor,
                  t_query: torch.Tensor) -> torch.Tensor:
    """t_known (N,), v_known (..., N, d), t_query (M,) -> (..., M, d)."""
    return interp(t_known, v_known.transpose(-1, -2), t_query).transpose(-1, -2)


def interp_batch(t_known: torch.Tensor, v_known: torch.Tensor,
                 t_query: torch.Tensor) -> torch.Tensor:
    """Shared knots, one query row per series: v_known (P, N[, d]), t_query (P, M)."""
    n = t_known.shape[0]
    idx = torch.searchsorted(t_known.contiguous(),
                             t_query.detach().contiguous(), right=True)
    i0 = (idx - 1).clamp(0, n - 2)
    t0, t1 = t_known[i0], t_known[i0 + 1]
    w = (t_query - t0) / (t1 - t0 + EPS)

    if v_known.dim() == 2:
        v0 = v_known.gather(1, i0)
        v1 = v_known.gather(1, i0 + 1)
        return v0 + w * (v1 - v0)

    e = i0.unsqueeze(-1).expand(-1, -1, v_known.shape[-1])
    v0 = v_known.gather(1, e)
    v1 = v_known.gather(1, e + 1)
    return v0 + w.unsqueeze(-1) * (v1 - v0)


def interp_inverse(t_known: torch.Tensor, v_known: torch.Tensor,
                   t_query: torch.Tensor) -> torch.Tensor:
    """Per-row knots, shared values -- used for phi^-1."""
    n = t_known.shape[-1]
    idx = torch.searchsorted(t_known.detach().contiguous(),
                             t_query.detach().contiguous(), right=True)
    i0 = (idx - 1).clamp(0, n - 2)
    t0 = t_known.gather(1, i0)
    t1 = t_known.gather(1, i0 + 1)
    v0, v1 = v_known[i0], v_known[i0 + 1]
    w = (t_query - t0) / (t1 - t0 + EPS)
    return v0 + w * (v1 - v0)
