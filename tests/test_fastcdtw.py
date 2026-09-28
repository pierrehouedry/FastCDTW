import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastcdtw import (MAD_FMTW, CumSumWarping, INRWarping, as_tensor,
                      cdist_fm_dtw, cdist_fm_dtw_fit, cdist_fm_dtw_mc,
                      cdist_fm_dtw_ref, fm_dtw, fm_dtw_matrix, fm_dtw_mc,
                      interp, interp_batch, interp_inverse, interp_series,
                      phi_from_raw, seg_integral)

torch.manual_seed(0)
DT = torch.float64


def _pl(t, knots, values):
    """Piecewise-linear reference, for quadrature."""
    return interp(knots, values, t)


def _quadrature(x, t_x, y, t_y, phi, t_phi, n=200_001):
    """Riemann reference for int ||x(phi(t)) - y(t)||^2 sqrt(1 + phi'^2) dt."""
    t = torch.linspace(0, 1, n, dtype=DT)
    ph = _pl(t, t_phi, phi)
    xv = interp_series(t_x, x, ph)
    yv = interp_series(t_y, y, t)
    dphi = torch.gradient(ph, spacing=(t,))[0]
    f = ((xv - yv) ** 2).sum(-1) * torch.sqrt(1 + dphi ** 2)
    return torch.trapezoid(f, t).squeeze()


def _instance(T=9, n_phi=7, d=2):
    t = torch.linspace(0, 1, T, dtype=DT)
    x = torch.randn(T, d, dtype=DT)
    y = torch.randn(T, d, dtype=DT)
    phi = phi_from_raw(torch.randn(n_phi - 1, dtype=DT))
    return x, y, t, phi, torch.linspace(0, 1, n_phi, dtype=DT)


def test_interp_matches_numpy_shapes():
    t = torch.linspace(0, 1, 5, dtype=DT)
    v = torch.randn(3, 5, 2, dtype=DT)
    q = torch.tensor([0.0, 0.13, 0.5, 1.0], dtype=DT)
    out = interp_series(t, v, q)
    assert out.shape == (3, 4, 2)
    torch.testing.assert_close(out[:, 0], v[:, 0])
    torch.testing.assert_close(out[:, -1], v[:, -1])


def test_interp_batch_equals_interp_per_row():
    t = torch.linspace(0, 1, 6, dtype=DT)
    v = torch.randn(4, 6, 3, dtype=DT)
    q = torch.rand(4, 10, dtype=DT).sort(-1).values
    out = interp_batch(t, v, q)
    for i in range(4):
        torch.testing.assert_close(out[i], interp_series(t, v[i], q[i]))


def test_interp_inverse_inverts_phi():
    phi = phi_from_raw(torch.randn(3, 9, dtype=DT)).sort(-1).values
    grid = torch.linspace(0, 1, 10, dtype=DT)
    q = torch.rand(3, 20, dtype=DT)
    back = interp_batch(grid, phi, interp_inverse(phi, grid, q))
    torch.testing.assert_close(back, q, atol=1e-9, rtol=0)


def test_seg_integral_matches_quadrature():
    d0, d1 = torch.randn(5, 3, dtype=DT), torch.randn(5, 3, dtype=DT)
    s = torch.linspace(0, 1, 100_001, dtype=DT)
    ref = torch.trapezoid((((d0[:, None] + s[:, None] * (d1 - d0)[:, None]) ** 2)
                           .sum(-1)), s, dim=1)
    torch.testing.assert_close(seg_integral(d0, d1), ref, atol=1e-8, rtol=1e-8)


def test_fm_dtw_matches_quadrature():
    x, y, t, phi, t_phi = _instance()
    got = fm_dtw(phi[None], t_phi, x[None], t, y[None], t,
                 torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(got[0], _quadrature(x[None], t, y[None], t, phi, t_phi),
                               atol=1e-6, rtol=1e-6)


def test_fm_dtw_identity_warping_is_l2():
    T, d = 11, 3
    t = torch.linspace(0, 1, T, dtype=DT)
    x, y = torch.randn(1, T, d, dtype=DT), torch.randn(1, T, d, dtype=DT)
    phi = t[None].clone()
    got = fm_dtw(phi, t, x, t, y, t, torch.zeros(1, dtype=torch.long),
                 torch.zeros(1, dtype=torch.long))
    diff = x[0] - y[0]
    ref = (seg_integral(diff[:-1], diff[1:]) * (t[1:] - t[:-1])).sum() * 2 ** 0.5
    torch.testing.assert_close(got[0], ref)


def test_fm_dtw_matrix_layout():
    N, K, T, d = 3, 2, 7, 2
    t = torch.linspace(0, 1, T, dtype=DT)
    Y, C = torch.randn(N, T, d, dtype=DT), torch.randn(K, T, d, dtype=DT)
    phi = phi_from_raw(torch.randn(N * K, T - 1, dtype=DT))
    M = fm_dtw_matrix(phi, t, C, t, Y, t)
    assert M.shape == (N, K)
    for i in range(N):
        for k in range(K):
            one = fm_dtw(phi[i * K + k][None], t, C, t, Y, t,
                         torch.tensor([i]), torch.tensor([k]))
            torch.testing.assert_close(M[i, k], one[0])


def test_cdist_exact_matches_reference():
    b_x, b_y, T, d, n_phi = 4, 3, 9, 2, 6
    t_x = torch.linspace(0, 1, T, dtype=DT)
    t_y = torch.linspace(0, 1, T + 2, dtype=DT)
    x = torch.randn(b_x, T, d, dtype=DT)
    y = torch.randn(b_y, T + 2, d, dtype=DT)
    phi = phi_from_raw(torch.randn(n_phi - 1, dtype=DT))
    torch.testing.assert_close(cdist_fm_dtw(x, y, t_x, t_y, phi),
                               cdist_fm_dtw_ref(x, y, t_x, t_y, phi))


def test_cdist_matches_quadrature():
    T, d, n_phi = 8, 2, 5
    t = torch.linspace(0, 1, T, dtype=DT)
    x, y = torch.randn(1, T, d, dtype=DT), torch.randn(1, T, d, dtype=DT)
    phi = phi_from_raw(torch.randn(n_phi - 1, dtype=DT))
    t_phi = torch.linspace(0, 1, n_phi, dtype=DT)
    got = cdist_fm_dtw(x, y, t, t, phi)[0, 0]
    torch.testing.assert_close(got, _quadrature(x, t, y, t, phi, t_phi),
                               atol=1e-6, rtol=1e-6)


def test_cdist_agrees_with_fm_dtw():
    T, d, n_phi = 8, 2, 8
    t = torch.linspace(0, 1, T, dtype=DT)
    x, y = torch.randn(2, T, d, dtype=DT), torch.randn(3, T, d, dtype=DT)
    phi = phi_from_raw(torch.randn(n_phi - 1, dtype=DT))
    cd = cdist_fm_dtw(x, y, t, t, phi)
    for i in range(2):
        for j in range(3):
            one = fm_dtw(phi[None], t, x[i][None], t, y[j][None], t,
                         torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
            torch.testing.assert_close(cd[i, j], one[0])


def _pl_batch(values, knots, rows=1):
    """A piecewise-linear callable of a (P, M) query."""
    v = values if values.dim() == 2 else values[None].expand(rows, -1)
    return lambda q: interp_batch(knots, v, q)


def test_fm_dtw_mc_converges_to_exact():
    x, y, t, phi, t_phi = _instance()
    zero = torch.zeros(1, dtype=torch.long)
    exact = fm_dtw(phi[None], t_phi, x[None], t, y[None], t, zero, zero)[0]
    g = torch.Generator().manual_seed(0)
    got = fm_dtw_mc(x=lambda p: interp_batch(t, x[None], p), Y=y[None], t_y=t,
                    phi=_pl_batch(phi, t_phi), n_mc=200_000, generator=g)[0]
    assert abs(float((got - exact).detach())) / float(exact) < 2e-2


def test_fm_dtw_mc_resamples():
    x, y, t, phi, t_phi = _instance()
    kw = dict(x=lambda p: interp_batch(t, x[None], p), Y=y[None], t_y=t,
              phi=_pl_batch(phi, t_phi), n_mc=1000)
    draw = lambda: float(fm_dtw_mc(**kw)[0].detach())
    assert draw() != draw()
    seeded = [float(fm_dtw_mc(**kw, generator=torch.Generator().manual_seed(3))[0].detach())
              for _ in range(2)]
    assert seeded[0] == seeded[1]


def test_cdist_mc_converges_to_exact():
    T, d, n_phi = 8, 2, 6
    t = torch.linspace(0, 1, T, dtype=DT)
    t_phi = torch.linspace(0, 1, n_phi, dtype=DT)
    x, y = torch.randn(2, T, d, dtype=DT), torch.randn(3, T, d, dtype=DT)
    phi = phi_from_raw(torch.randn(n_phi - 1, dtype=DT))
    exact = cdist_fm_dtw(x, y, t, t, phi)
    g = torch.Generator().manual_seed(0)
    got = cdist_fm_dtw_mc(x, y, t, t, lambda s: interp(t_phi, phi, s),
                          n_mc=200_000, generator=g)
    assert got.shape == exact.shape
    assert float(((got - exact).abs() / exact).max().detach()) < 3e-2


def test_mc_gradients_reach_the_warping():
    T, d = 10, 2
    t = torch.linspace(0, 1, T)
    x, y = torch.randn(2, T, d), torch.randn(3, T, d)
    w = CumSumWarping(T)
    cdist_fm_dtw_mc(x, y, t, t, w, n_mc=500).sum().backward()
    assert w.a.grad is not None and torch.isfinite(w.a.grad).all()
    assert float(w.a.grad.abs().sum()) > 0


def test_mad_fmtw_exact_and_mc():
    pytest.importorskip("ot")
    T, d = 12, 2
    t = torch.linspace(0, 1, T)
    x, y = torch.randn(4, T, d), torch.randn(4, T, d)
    for n_mc in (None, 300):
        mad = MAD_FMTW(CumSumWarping(T), n_mc=n_mc)
        loss = mad(x, y, t, t)
        loss.backward()
        assert torch.isfinite(loss) and mad.ot_plan.shape == (4, 4)
        assert float(mad.phi_x.a.grad.abs().sum()) > 0


def test_gradients_flow():
    x, y, t, _, t_phi = _instance()
    raw = torch.zeros(1, len(t_phi) - 1, dtype=DT, requires_grad=True)
    loss = fm_dtw(phi_from_raw(raw), t_phi, x[None], t, y[None], t,
                  torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)).sum()
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


@pytest.mark.parametrize("make", [
    lambda: CumSumWarping(12),
    lambda: INRWarping(n_grid=32, n_freq=4, hidden=16, n_layers=2, use_monte_carlo=False),
])
def test_warpings_are_monotone_and_normalised(make):
    torch.manual_seed(1)
    w = make()
    with torch.no_grad():
        for p in w.parameters():
            p.add_(0.5 * torch.randn_like(p))
        phi = w(torch.linspace(0, 1, 101))
    assert (phi[1:] - phi[:-1] >= -1e-6).all()
    assert abs(float(phi[0])) < 1e-5 and abs(float(phi[-1]) - 1) < 1e-5


def test_warping_is_differentiable_in_t():
    w = CumSumWarping(10)
    t = torch.rand(50).sort().values.requires_grad_(True)
    phi = w(t)
    dphi, = torch.autograd.grad(phi, t, torch.ones_like(phi), create_graph=True)
    assert (dphi >= 0).all()


def two_class_bumps(n_per_class=3, T=40, seed=0):
    """Two shapes, each read on a slightly different clock."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    X, y = [], []
    for k, centre in enumerate((0.35, 0.65)):
        for _ in range(n_per_class):
            shift = rng.uniform(-0.04, 0.04)
            X.append(np.exp(-((t - centre - shift) ** 2) / 0.01))
            y.append(k)
    return np.asarray(X)[:, :, None], np.asarray(y)


def test_optimising_never_increases_the_distance():
    X, _ = two_class_bumps(n_per_class=2, T=30)
    kw = dict(dtype=torch.float64, device="cpu", seed=0)
    D0 = cdist_fm_dtw_fit(X[2:], X[:2], exact=True, n_steps=0, **kw)
    D = cdist_fm_dtw_fit(X[2:], X[:2], exact=True, n_steps=60, lr=1e-2, **kw)
    assert (D <= D0 + 1e-9).all()
    assert (D > 0).all()


def test_the_discrepancy_is_asymmetric():
    """The warping goes on the train series, the integral on the test axis."""
    X, _ = two_class_bumps(n_per_class=2, T=30)
    kw = dict(exact=True, n_steps=80, lr=1e-2, dtype=torch.float64,
              device="cpu", seed=0)
    forward = cdist_fm_dtw_fit(X[2:], X[:2], **kw)
    backward = cdist_fm_dtw_fit(X[:2], X[2:], **kw)
    assert not np.allclose(forward, backward.T, rtol=1e-3)


def test_exact_and_mc_reach_the_same_distances_once_converged():
    """Same objective, two gradient estimators: a small budget compares
    trajectories, not minima (MC descends faster early, 0.11 vs 0.28 at 60 steps).
    """
    X, _ = two_class_bumps(n_per_class=2, T=24)
    kw = dict(lr=1e-2, dtype=torch.float64, device="cpu", seed=0)
    D0 = cdist_fm_dtw_fit(X[2:], X[:2], exact=True, n_steps=0, **kw)
    D_ex = cdist_fm_dtw_fit(X[2:], X[:2], exact=True, n_steps=600, **kw)
    D_mc = cdist_fm_dtw_fit(X[2:], X[:2], exact=False, n_steps=600, n_mc=2000,
                          n_eval=20, **kw)
    assert (D_ex < 0.05 * D0).all() and (D_mc < 0.05 * D0).all()
    assert (np.abs(D_ex - D_mc) < 0.02 * D0).all()


def test_chunking_changes_nothing_in_the_exact_path():
    X, _ = two_class_bumps(n_per_class=3, T=24)
    kw = dict(exact=True, n_steps=40, lr=1e-2, dtype=torch.float64,
              device="cpu", seed=0)
    whole = cdist_fm_dtw_fit(X[3:], X[:3], chunk=0, **kw)
    parts = cdist_fm_dtw_fit(X[3:], X[:3], chunk=2, **kw)
    assert np.allclose(whole, parts, atol=1e-10)


def test_length_mismatch_is_refused():
    X, _ = two_class_bumps(n_per_class=2, T=24)
    with pytest.raises(ValueError, match="length mismatch"):
        cdist_fm_dtw_fit(X[2:], X[:2, :20], exact=True, n_steps=1, device="cpu")


def test_as_tensor_adds_the_feature_axis():
    assert as_tensor(np.zeros((4, 7)), torch.float32, "cpu").shape == (4, 7, 1)
