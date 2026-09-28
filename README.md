# FastCDTW

Fast Marching Time Warping. For a warping $\varphi:[0,1]\to[0,1]$ increasing with
$\varphi(0)=0$, $\varphi(1)=1$,

$$F(\varphi)=\int_0^1 \lVert x(\varphi(t))-y(t)\rVert_2^2\,\sqrt{1+\varphi'(t)^2}\,\mathrm{d}t,$$

the length of the curve $(\varphi,\mathrm{id})$ in $[0,1]^2$ under the conformal
metric $c(u,v)=\lVert x(u)-y(v)\rVert_2^2$. Squared cost only ($p=2$).

## Closed form

With $x$, $y$, $\varphi$ piecewise linear on a common breakpoint set
$\{\tau_k\}$,

$$F(\varphi)=\frac{1}{3}\sum_k \sqrt{\Delta\tau_k^2+\Delta\varphi_k^2}\,
\left(\lVert d_k^0\rVert^2+\langle d_k^0,d_k^1\rangle+\lVert d_k^1\rVert^2\right),$$

$d_k^0,d_k^1$ being the endpoint differences on segment $k$. Exact, with no
quadrature, no logs and no degenerate branches: the integrand is quadratic in
the local parameter, and $\Delta\tau_k\sqrt{1+\varphi'^2}=\sqrt{\Delta\tau_k^2+\Delta\varphi_k^2}$
removes the derivative.

**The breakpoint set must contain $\varphi$'s own knots**, otherwise $\varphi'$
is not constant on a segment and the formula is only approximate. The union
used here is $\{\varphi^{-1}(t_{x,i})\}\cup\{t_{y,j}\}\cup\{k/N_\varphi\}$;
$\{t_{x,i}\}$ is harmless but unnecessary. On a random instance, dropping the
$\varphi$ knots costs 0.75% relative error against fine quadrature, against
1e-7 (the quadrature's own error) with them.

## Layout

```
fastcdtw/
  interp.py                 piecewise-linear interpolation, four indexing patterns
  loss.py                   fm_dtw (one warping per pair, exact), fm_dtw_matrix, fm_dtw_mc
  loss_shared_time.py       cdist_fm_dtw_ref (materialises b_x b_y M d), cdist_fm_dtw_mc
  loss_shared_time_exact.py cdist_fm_dtw -- expanded closed form; MAD_FMTW
  cdist_fit.py              cdist_fm_dtw_fit -- cross-distance matrix, one warping fitted per pair
  warping.py                phi_from_raw, CumSumWarping, INRWarping
data/toy/                   synthetic families (ARFF + ground truth), from experiments/averaging/make_toy.py
experiments/averaging/      barycenters (Blondel et al. 2021, 4.1-4.2) on the synthetic families
experiments/clustering/     Blondel et al. 2021, Table 2: 1-NN on the UCR archive
tests/                      closed form vs quadrature, MC vs closed form, gradients
```

Two problem shapes, each with an exact and a sampled path — same objective, so
the sampled ones converge to the exact ones (tested):

|                                | exact (piecewise linear) | sampled (anything) |
|--------------------------------|--------------------------|--------------------|
| one warping per pair           | `fm_dtw`                 | `fm_dtw_mc`        |
| one warping, all cross-pairs   | `cdist_fm_dtw`           | `cdist_fm_dtw_mc`  |

The exact ones take $\varphi$ as knot values; the sampled ones take it as a
callable and differentiate it with `autograd` — that is the whole difference in
use. Reach for a sampled path when $x$ or $\varphi$ is not piecewise linear (an
INR centroid, an MLP warping), where the closed form does not apply. Both draw
fresh times on every call and accept a `generator` when you want that fixed.

`cdist_fm_dtw` never materialises anything larger than $(b_x,b_y,M)$: with
$\varphi$ shared, $d^0$ and $d^1$ split into per-series terms plus four cross
terms, each an einsum contracting $d$ on the fly. `cdist_fm_dtw_mc` expands
$\lVert a-b\rVert^2$ the same way, for the same reason. At $b=128$, $T=50$,
$d=128$ that is 0.25 s and 0.04 GB against 10.3 s and 1.26 GB for
`cdist_fm_dtw_ref` — the reference exists to check the expansion, not to train
with.

## Usage

```python
import torch
from fastcdtw import fm_dtw, phi_from_raw

t = torch.linspace(0, 1, T)
raw = torch.zeros(N, T - 1, requires_grad=True)      # identity warping
loss = fm_dtw(phi_from_raw(raw), t, C, t, Y, t,
              series_idx=torch.arange(N),
              cluster_idx=torch.zeros(N, dtype=torch.long)).sum()
loss.backward()
```

`phi_from_raw` (normalised softplus increments) is the cheapest
parametrisation and is what the solvers use. `CumSumWarping` wraps it as a
module; `INRWarping` integrates a positive MLP rate.

Sampled, with a warping module and an INR centroid:

```python
from fastcdtw import CumSumWarping, cdist_fm_dtw_mc, fm_dtw_mc

phi = CumSumWarping(T)                                   # any callable of t
cost = cdist_fm_dtw_mc(x, y, t_x, t_y, phi, n_mc=200)    # (b_x, b_y)

loss = fm_dtw_mc(x=lambda p: centroid(p.reshape(-1)).reshape(*p.shape, -1),
                 Y=Y, t_y=t, phi=lambda ts: bank(idx, ts), n_mc=200).sum()
```

`fm_dtw_mc` takes $x$ and $\varphi$ as callables of $t$ and $Y$ as the data
tensor; `eval_warping` gets $\varphi(t)$ and $\varphi'(t)$ in one pass, so the
gradient reaches the warping's parameters through both. Grad must be enabled —
under `torch.no_grad()` wrap the call in `torch.enable_grad()`.

## Data conventions

Everything is plain tensors plus explicit knot vectors: `(N, T, d)` values with
a `(T,)` grid, or `(N, n)` warpings with their own grid. The four interpolation
patterns the losses need are named functions in `interp.py`.

## Experiments

```bash
pip install -e ".[experiments,ot,dev]"
pytest tests -q
```

- `experiments/averaging/` -- barycenters of synthetic families built from a
  known template on a known clock, so a barycenter can be scored rather than
  eyeballed: Euclidean mean, DTW (DBA) and soft-DTW (tslearn) against FastCDTW
  with a free-vector centroid (exact integral) or an INR centroid (Monte Carlo
  integral). Its README lists the commands behind every figure.
- `experiments/clustering/` -- Table 2: 1-NN classification on the UCR archive
  under five discrepancies. FastCDTW is a minimum over warpings, so a
  cross-distance matrix costs one warping fit per (test, train) pair -- a
  different cost model from the barycenter experiments, and the reason that
  folder has its own checkpointing. See its README.

Every script writes into a `results/` folder next to it, created on first run.
