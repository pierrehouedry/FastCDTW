import torch

from .interp import interp_batch, interp_inverse

SEG_EPS = 1e-20  # guards sqrt(0) on zero-length segments


def seg_integral(d0: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:

    delta = d1 - d0
    return ((delta * delta).sum(-1) / 3.0
            + (d0 * delta).sum(-1)
            + (d0 * d0).sum(-1))


def fm_dtw(phi: torch.Tensor,
           t_phi: torch.Tensor,
           C: torch.Tensor,
           t_c: torch.Tensor,
           Y: torch.Tensor,
           t_y: torch.Tensor,
           series_idx: torch.Tensor,
           cluster_idx: torch.Tensor) -> torch.Tensor:

    P = phi.shape[0]
    phi_inv = interp_inverse(phi, t_phi, t_c.unsqueeze(0).expand(P, -1))

    t_all, _ = torch.sort(torch.cat([t_phi.unsqueeze(0).expand(P, -1),
                                     t_y.unsqueeze(0).expand(P, -1),
                                     phi_inv], dim=1), dim=1)
    phi_at_t = interp_batch(t_phi, phi, t_all)
    diff = (interp_batch(t_c, C[cluster_idx], phi_at_t)
            - interp_batch(t_y, Y[series_idx], t_all))

    dt = t_all[:, 1:] - t_all[:, :-1]
    dphi = phi_at_t[:, 1:] - phi_at_t[:, :-1]
    arc = torch.sqrt(dt * dt + dphi * dphi + SEG_EPS)
    return (seg_integral(diff[:, :-1], diff[:, 1:]) * arc).sum(dim=1)


def fm_dtw_matrix(phi: torch.Tensor, t_phi: torch.Tensor,
                  C: torch.Tensor, t_c: torch.Tensor,
                  Y: torch.Tensor, t_y: torch.Tensor) -> torch.Tensor:
    """All (series, cluster) pairs -> (N, K). phi is (N*K, n), row n*K + k."""
    N, K = Y.shape[0], C.shape[0]
    pairs = torch.arange(N * K, device=Y.device)
    return fm_dtw(phi, t_phi, C, t_c, Y, t_y,
                  series_idx=pairs // K, cluster_idx=pairs % K).reshape(N, K)


def sample_times(shape, device=None, dtype=torch.float32,
                 generator=None) -> torch.Tensor:
    """Uniform draws on [0, 1], sorted along the last axis, re-drawn every call."""
    u = torch.rand(*shape, device=device, dtype=dtype, generator=generator)
    return u.sort(dim=-1).values


def eval_warping(phi, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """phi(t) and phi'(t), the derivative by autodiff."""
    ts = t.detach().clone().requires_grad_(True)
    phi_t = phi(ts)
    dphi, = torch.autograd.grad(phi_t, ts, torch.ones_like(phi_t),
                                retain_graph=True, create_graph=True)
    return phi_t, dphi


def fm_dtw_mc(x, Y: torch.Tensor, t_y: torch.Tensor, phi,
              n_mc: int = 200, generator=None) -> torch.Tensor:

    t = sample_times((Y.shape[0], n_mc), device=Y.device, dtype=Y.dtype,
                     generator=generator)
    phi_t, dphi = eval_warping(phi, t)
    diff = x(phi_t) - interp_batch(t_y, Y, t)
    return ((diff ** 2).sum(-1) * torch.sqrt(1.0 + dphi * dphi)).mean(-1)
