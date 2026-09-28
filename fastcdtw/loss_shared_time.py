import torch

from .interp import dedup_times, interp, interp_series
from .loss import SEG_EPS, eval_warping, sample_times, seg_integral


def breakpoints(t_x: torch.Tensor, t_y: torch.Tensor,
                phi_x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Sorted, deduplicated {phi^-1(t_x)} u {t_y} u {phi knots}."""
    phi_inv_x = interp(phi_x, grid, t_x)
    return dedup_times(torch.sort(torch.cat([phi_inv_x, t_y, grid]))[0])


def cdist_fm_dtw_ref(x: torch.Tensor, y: torch.Tensor,
                     t_x: torch.Tensor, t_y: torch.Tensor,
                     phi_x: torch.Tensor) -> torch.Tensor:
    
    assert x.shape[-1] == y.shape[-1], "feature dimensions must match"
    grid = torch.linspace(0, 1, len(phi_x), device=x.device, dtype=x.dtype)

    t = breakpoints(t_x, t_y, phi_x, grid)
    phi_at_t = interp(grid, phi_x, t)
    x_interp = interp_series(t_x, x, phi_at_t)
    y_interp = interp_series(t_y, y, t)

    diff = x_interp[:, None] - y_interp[None, :]
    dt = t[1:] - t[:-1]
    dphi = phi_at_t[1:] - phi_at_t[:-1]
    arc = torch.sqrt(dt * dt + dphi * dphi + SEG_EPS)
    return (seg_integral(diff[:, :, :-1], diff[:, :, 1:]) * arc).sum(-1)


def cdist_fm_dtw_mc(x: torch.Tensor, y: torch.Tensor,
                    t_x: torch.Tensor, t_y: torch.Tensor,
                    phi, n_mc: int = 200, generator=None) -> torch.Tensor:
    """Monte Carlo counterpart of ``cdist_fm_dtw``."""
    assert x.shape[-1] == y.shape[-1], "feature dimensions must match"
    t = sample_times((n_mc,), device=x.device, dtype=x.dtype,
                     generator=generator)
    phi_t, dphi = eval_warping(phi, t)

    xv = interp_series(t_x, x, phi_t)
    yv = interp_series(t_y, y, t)
    sq = (((xv * xv).sum(-1)[:, None] + (yv * yv).sum(-1)[None, :]
           - 2 * torch.einsum('imd,jmd->ijm', xv, yv)).clamp_min(0))
    return (sq * torch.sqrt(1.0 + dphi * dphi)).mean(-1)
