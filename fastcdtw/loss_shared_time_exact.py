import torch

from .interp import interp, interp_series
from .loss import SEG_EPS
from .loss_shared_time import breakpoints, cdist_fm_dtw_mc


def cdist_fm_dtw(x: torch.Tensor, y: torch.Tensor,
                 t_x: torch.Tensor, t_y: torch.Tensor,
                 phi_x: torch.Tensor) -> torch.Tensor:
    
    assert x.shape[-1] == y.shape[-1], "feature dimensions must match"
    grid = torch.linspace(0, 1, len(phi_x), device=x.device, dtype=x.dtype)

    t = breakpoints(t_x, t_y, phi_x, grid)
    phi_at_t = interp(grid, phi_x, t)
    x_interp = interp_series(t_x, x, phi_at_t)
    y_interp = interp_series(t_y, y, t)

    xl, dx = x_interp[:, :-1], x_interp[:, 1:] - x_interp[:, :-1]
    yl, dy = y_interp[:, :-1], y_interp[:, 1:] - y_interp[:, :-1]

    nxl, ndx = (xl * xl).sum(-1), (dx * dx).sum(-1)
    nyl, ndy = (yl * yl).sum(-1), (dy * dy).sum(-1)
    px, py = (xl * dx).sum(-1), (yl * dy).sum(-1)

    xl_yl = torch.einsum('imd,jmd->ijm', xl, yl)
    dx_dy = torch.einsum('imd,jmd->ijm', dx, dy)
    xl_dy = torch.einsum('imd,jmd->ijm', xl, dy)
    dx_yl = torch.einsum('imd,jmd->ijm', dx, yl)

    C = nxl[:, None] + nyl[None, :] - 2 * xl_yl             # ||d0||^2
    A = ndx[:, None] + ndy[None, :] - 2 * dx_dy             # ||delta||^2
    B = 2 * (px[:, None] - xl_dy - dx_yl + py[None, :])     # 2 <d0, delta>

    dt = t[1:] - t[:-1]
    dphi = phi_at_t[1:] - phi_at_t[:-1]
    arc = torch.sqrt(dt * dt + dphi * dphi + SEG_EPS)
    return ((A / 3 + B / 2 + C) * arc).sum(-1)

class MAD_FMTW(torch.nn.Module):
    """Optimal-transport loss between two batches under the FM-DTW cost."""

    def __init__(self, phi_x, n_mc: int | None = None):
        super().__init__()
        self.phi_x = phi_x
        self.n_mc = n_mc
        self.ot_plan = None

    def cross_distances(self, x, y, t_x, t_y) -> torch.Tensor:
        if self.n_mc is None:
            return cdist_fm_dtw(x, y, t_x, t_y, self.phi_x())
        return cdist_fm_dtw_mc(x, y, t_x, t_y, self.phi_x, n_mc=self.n_mc)

    def forward(self, x, y, t_x, t_y) -> torch.Tensor:
        import ot

        cost = self.cross_distances(x, y, t_x, t_y)
        self.ot_plan = ot.emd([], [], cost)
        return (self.ot_plan * cost).sum()
