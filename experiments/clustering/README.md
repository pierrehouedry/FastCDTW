# 1-NN classification on UCR

Table 2 of Blondel, Mensch & Vert 2021: 1-nearest-neighbour classification on
the UCR archive under five discrepancies.

| method | what it is |
|---|---|
| `euclidean` | flat $\ell_2$ |
| `dtw` | tslearn `cdist_dtw` |
| `softdtw` | tslearn `cdist_soft_dtw`, raw values, gamma cross-validated |
| `fastcdtw_exact` | FastCDTW, exact closed form (`fm_dtw`) |
| `fastcdtw_mc` | FastCDTW, Monte Carlo integral (`fm_dtw_mc`) |

The last two are the same discrepancy under two estimators of the same
integral, so the pair measures what the sampling costs, nothing else.

## Cost model

FastCDTW is a minimum over warpings, so a cross-distance matrix needs **one
warping optimisation per (test, train) pair**, not one per series. All pairs
are fitted as one batch of $(P, T-1)$ raw increments and chunked by `--chunk`
(default 20000). `--max-pairs` (default 2e6) skips datasets whose
$n_\text{train} \times n_\text{test}$ is too large. GPU is the intended target.

The discrepancy is **asymmetric**: the warping goes on the *train* series and
the integral runs over the *test* series' time axis. $d(\text{test},
\text{train})$ is what 1-NN needs and it is not $d(\text{train}, \text{test})$.
Pinned by a test.

## The budget is the regulariser

Every pair gets the same `--nn-steps` (default 300) from raw = 0, the identity
warping. The optimisation does not converge and cannot: the softplus-cumsum
warping collapses, so the objective keeps falling as $\varphi$ degenerates.
300 steps is early stopping and it is the only thing holding $\varphi$ back.
Raising the budget lowers every entry of the table without improving accuracy,
so **matrices computed under different budgets are not comparable**. The
learning rate is cross-validated on train over $[0.001, 0.005, 0.01]$ with 5
random 2/3–1/3 splits, the same protocol the paper uses for gamma.

## Flow

```bash
pip install -e "../..[experiments]"
python3 run.py --list --device cpu     # sizes, pair counts -> results/ucr_nn_datasets.csv
python3 run.py Coffee --device cpu     # one dataset
./run_all.sh                           # every dataset marked 'keep', sequentially
python3 summarize.py --csv             # the win matrix
python3 time_matrices.py Coffee        # wall-clock of one matrix per method
```

Results append to a file-locked JSON checkpoint, so re-running recomputes
nothing already done (`--force` overrides) and parallel jobs can write
concurrently. Output lands in `results/`: `checkpoints/ucr_nn_checkpoint.json`,
`table2_ucr_nn.txt`, `accuracy_ucr_nn.csv`, `logs/`.

`datasets.txt` is the list of UCR datasets used in our runs.
