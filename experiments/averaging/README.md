# Averaging experiments

Two experiments, both on synthetic families where the true template and the
true clocks are known.

1. **The final grid** — one row per dataset (`toy_accel`, `toy_ecg`,
   `toy_rate`), one column per method: Euclidean mean, DTW (DBA), soft-DTW,
   FastCDTW (vector, exact), FastCDTW (INR, MC). Plus the objective traces and
   one standalone panel per (dataset, method) couple, FastCDTW (vector, MC)
   included.
2. **The Monte-Carlo sweep** — the INR barycenter against the number of MC
   draws per step, everything else held fixed.

## Data

`make_toy.py` builds each family from a template $z$ read on a known clock and
writes `data/toy/toy_<family>.arff` plus a `_truth.json` holding $z$, the true
warpings and the interpolant. Deterministic given `--seed`.

| family | n | what it tests |
|---|---|---|
| `rate` | 8 | one peak, eight reading speeds $t^s$, $s \in [0.65, 1.55]$ |
| `accel` | 12 | narrow peak, sine clocks $t + a\sin(2\pi t)/2\pi$, $a \in [-0.9, 0.9]$: the rate goes above **and** below 1 inside each series |
| `ecg` | 12 | a P-QRS-T complex on the `accel` clocks — three time scales and a negative lobe |

The generator still writes the other families (`sharp`, `damped`, `shift`,
`slide`, `spread`, `morph`); nothing here reads them.

## Flow

Everything runs from this folder.

```bash
python3 make_toy.py                                  # -> data/toy/

COMMON="--seed 42 --methods euclidean dtw softdtw fastcdtw_vec fastcdtw_inr fastcdtw_vec_mc \
        --c-freq 10 --n-outer 30 --n-steps 500 --n-mc 200 --n-eval 50 \
        --no-eval --tag final10"
python3 run.py --data ../../data/toy/toy_rate.arff  --n-series 8  $COMMON
python3 run.py --data ../../data/toy/toy_accel.arff --n-series 12 $COMMON
python3 run.py --data ../../data/toy/toy_ecg.arff   --n-series 12 $COMMON \
    --lr-inr 1.5e-3

# FastCDTW (INR, MC) panel on ecg: larger MC budget
python3 run.py --data ../../data/toy/toy_ecg.arff --n-series 12 --seed 42 \
    --methods fastcdtw_inr --c-freq 10 --n-outer 30 --n-steps 500 \
    --n-mc 800 --n-eval 100 --lr-inr 0.015 --no-eval --tag mc800_0.015

# Soft-DTW panels: temperature sweep, plot_final.py keeps the lowest DTW loss
for spec in accel:12 ecg:12 rate:8; do
    python3 gamma_sweep.py --data ../../data/toy/toy_${spec%%:*}.arff \
        --n-series ${spec##*:} --seed 42 --gamma 0.0001 0.001 0.01 0.1 1 10 \
        --out-dir results/gammasweep_full
done

python3 plot_final.py                                # -> results/figures/final10/
python3 mc_sweep.py                                  # -> results/
```

## Settings that the figures depend on

- `--c-freq 10`: 10 Fourier frequencies, so 20 inputs (sin and cos of
  $2^k\pi t$, $k < 10$). Chosen by a sweep over the encoding budget; 10 is the
  minimum of the objective on `rate` and on `accel`.
- `--n-outer 30 --n-steps 500 --n-mc 200`: the objective is flat over the last
  three outer iterations for every solver except the INR on `ecg`, which
  plateaus at a higher value. `--n-eval 50` keeps MC noise out of the trace.
- `toy_ecg` uses `--lr-inr 1.5e-3`. Every solver starts from the Euclidean
  mean with identity warpings; there is no DBA warm start. On `ecg` the INR can
  still invent extrema and the trace is not always monotone. On `rate` and
  `accel` the default learning rate is better.
- The INR centroid is a function, so `run.py --dense N` also stores a dense
  read of it and `plot_final.py` draws that instead of the sample grid.

## Output

- `results/averaging_toy_<family>_n<N>_seed42_final10.json` — barycenters,
  objective traces, warpings, timings, config.
- `results/figures/final10/` — `barycenters_grid`, `barycenters_<dataset>`,
  `losses`, `single/<dataset>_<method>` (PNG and PDF) and `convergence.json`.
- `results/mcsweep_toy_rate_n8_seed42.json` and
  `results/figures/mcsweep_toy_rate_n8_seed42.png`.
- `results/gammasweep_full/` -- the soft-DTW sweep the Soft-DTW panels are
  taken from.

## Other scripts

- `gamma_sweep.py` -- the soft-DTW barycenter against its temperature, nothing
  else moving (used above for the Soft-DTW panels).
- `inr_continuous.py` -- the discrete barycenters against the INR one, on a
  coarse sample grid.
- `sampling/` -- DTW, Soft-DTW and FastCDTW between two toy series resampled at
  $n$ timestamps on $[0, 1]$; `softdtw_gammas.py` redraws the Soft-DTW curve
  for several gammas from the `sampling_linspace` JSON:

  ```bash
  python3 sampling/sampling.py --pairs accel-accel rate-rate ecg-ecg --grid linspace --n-rep 5
  python3 sampling/sampling.py --pairs accel-accel rate-rate ecg-ecg --grid random   --n-rep 5
  python3 sampling/softdtw_gammas.py
  ```
- `run.py --indices i j --weights 0.25 0.75` -- a weighted barycenter of two
  chosen series, the interpolation experiment (Blondel et al. 2021, 4.2).
- `evaluate.py` -- without `--no-eval`, `run.py` also writes a cross-evaluation
  table, `results/<stem>_crosseval.txt`.

That table is a diagnostic, not a ranking: every method minimises its own
column. `fastcdtw(refit)` refits the warpings *cold* against a frozen centroid at
a fixed budget, so it over-states the objective for every candidate, ours
included; `fastcdtw(as fitted)` is what our solvers reached with warm-started
warpings and has no counterpart for the baselines.
