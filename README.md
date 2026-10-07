# FastCDTW

Code for *Fast Gradient-Based Approximation of Continuous Dynamic Time Warping*.
FastCDTW computes Continuous DTW (CDTW) [1], the continuous-time counterpart of DTW [2], by gradient descent on a parametrized warping function.
The cost of a warping is computed in closed form for piecewise-linear (PL) warpings, or by an unbiased Monte Carlo (MC) estimate for smooth ones such as implicit neural representations [3]; both give gradients with respect to the warping and the series, and run batched on GPU.
`fastcdtw/` is the library; `experiments/averaging/` compares FastCDTW barycenters with DBA [4] and Soft-DTW [5] (run `make_toy.py` first to build the data), and `experiments/clustering/` runs 1-NN classification on the UCR archive [6].

![Barycenters of three synthetic families](figures/barycenters_grid.png)

*Figure 3 of the paper: barycenters of `accel` (top), `ecg` (middle) and `rate` (bottom). Grey: the averaged series.*

## References

1. K. Buchin, A. Nusser, S. Wong. Computing continuous dynamic time warping of time series in polynomial time. *Journal of Computational Geometry*, 2025.
2. H. Sakoe, S. Chiba. Dynamic programming algorithm optimization for spoken word recognition. *IEEE Trans. Acoustics, Speech, and Signal Processing*, 1978.
3. E. Le Naour et al. Time series continuous modeling for imputation and forecasting with implicit neural representations. *TMLR*, 2024.
4. F. Petitjean, A. Ketterlin, P. Gançarski. A global averaging method for dynamic time warping, with applications to clustering. *Pattern Recognition*, 2011.
5. M. Cuturi, M. Blondel. Soft-DTW: a differentiable loss function for time-series. *ICML*, 2017.
6. H. A. Dau et al. The UCR time series archive. *IEEE/CAA Journal of Automatica Sinica*, 2019.

## Citation

```bibtex
@misc{houedry2026fastcdtw,
  title  = {Fast Gradient-Based Approximation of Continuous Dynamic Time Warping},
  author = {Houedry, Pierre and Tavenard, Romain and Courty, Nicolas and Gaudel, Romaric},
  year   = {2026}
}
```
