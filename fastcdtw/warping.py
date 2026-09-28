"""Monotone warpings phi: [0, 1] -> [0, 1], phi(0) = 0, phi(1) = 1."""

import torch
import torch.nn as nn

from .interp import interp

EPS = 1e-12


def phi_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """(..., n-1) unconstrained -> (..., n) monotone, softplus increments."""
    inc = torch.nn.functional.softplus(raw)
    inc = inc / (inc.sum(dim=-1, keepdim=True) + EPS)
    zero = inc.new_zeros(inc.shape[:-1] + (1,))
    return torch.cat([zero, torch.cumsum(inc, dim=-1)], dim=-1)


class CumSumWarping(nn.Module):
    """Piecewise linear, n knots on a uniform grid. One parameter per segment."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.register_buffer("t_ref", torch.linspace(0, 1, n))
        self.a = nn.Parameter(torch.zeros(n - 1))

    def __len__(self):
        return self.n

    def forward(self, t: torch.Tensor | None = None) -> torch.Tensor:
        phi = phi_from_raw(self.a)
        return phi if t is None else interp(self.t_ref, phi, t)


class INRWarping(nn.Module):
    """phi = normalised integral of a positive Fourier-feature MLP rate."""

    def __init__(self, n_grid: int = 100, n_freq: int = 8, hidden: int = 64,
                 n_layers: int = 4, use_monte_carlo: bool = True):
        super().__init__()
        layers = [nn.Linear(2 * n_freq, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)
        self.n_freq, self.n_grid = n_freq, n_grid
        self.use_monte_carlo = use_monte_carlo
        self.register_buffer("t_grid", torch.linspace(0.0, 1.0, n_grid))

    def _fourier(self, t: torch.Tensor) -> torch.Tensor:
        freqs = 2 ** torch.arange(self.n_freq, dtype=t.dtype, device=t.device)
        proj = t.unsqueeze(-1) * freqs * torch.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(next(self.parameters()).device)
        if self.use_monte_carlo:
            interior = torch.rand(self.n_grid - 2, dtype=t.dtype, device=t.device).sort().values
            grid = torch.cat([interior.new_zeros(1), interior, interior.new_ones(1)])
        else:
            grid = self.t_grid.to(dtype=t.dtype)

        rate = self.net(self._fourier(grid)).squeeze(-1)
        trapz = (rate[:-1] + rate[1:]) / 2 * (grid[1:] - grid[:-1])
        phi = torch.cat([trapz.new_zeros(1), torch.cumsum(trapz, 0)])
        phi = phi / (phi[-1] + EPS)
        return interp(grid, phi, t).clamp(0.0, 1.0)
