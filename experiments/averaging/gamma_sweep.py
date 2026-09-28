"""The soft-DTW barycenter against its temperature gamma; nothing else moves."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

COLOUR = "#6ACC64" 


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(HERE.parents[1] / "data" / "toy" / "toy_rate.arff"))
    p.add_argument("--n-series", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, nargs="+",
                   default=[0.01, 0.1, 1.0, 10.0])
    p.add_argument("--lbfgs-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--replot", default=None, help="redraw from a sweep JSON")
    p.add_argument("--exclude-gamma", type=float, nargs="+", default=(),
                   help="with --replot: leave these gammas out of the figure")
    p.add_argument("--suffix", default="", help="with --replot: appended to the figure name")
    p.add_argument("--color", default=COLOUR)
    args = p.parse_args()

    if args.replot:
        payload = json.loads(Path(args.replot).read_text())
        if args.exclude_gamma:
            payload["runs"] = [r for r in payload["runs"]
                               if not any(np.isclose(r["gamma"], g)
                                          for g in args.exclude_gamma)]
        stem = Path(args.replot).stem + args.suffix
        out = Path(args.replot).parent / "figures" / f"{stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot(payload, out, args.color)
        return

    from barycenters import softdtw_barycenter
    from data import load, sample_series
    from evaluate import dtw_objective, softdtw_objective

    X, labels = load(args.data)
    Y, idx = sample_series(X, labels, n=args.n_series, seed=args.seed)
    dataset = Path(args.data).stem
    print(f"{dataset}: {Y.shape[0]} series, {Y.shape[1]} samples")

    runs = []
    for gamma in args.gamma:
        t0 = time.time()
        C = softdtw_barycenter(Y, gamma=gamma, max_iter=args.lbfgs_iter,
                               tol=args.tol)
        dt = time.time() - t0
        runs.append({"gamma": gamma, "barycenter": np.asarray(C).tolist(),
                     "dtw": dtw_objective(C, Y),
                     "softdtw_own": softdtw_objective(C, Y, gamma=gamma),
                     "time": dt})
        print(f"gamma={gamma:<8g} DTW objective={runs[-1]['dtw']:.6f}  "
              f"{dt:6.1f} s", flush=True)

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    stem = f"gammasweep_{dataset}_n{args.n_series}_seed{args.seed}"
    payload = {"config": vars(args), "dataset": dataset,
               "indices": idx.tolist(), "series": Y[:, :, 0].tolist(),
               "runs": runs}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload))
    print(f"\n-> {out_dir / (stem + '.json')}")

    plot(payload, out_dir / "figures" / f"{stem}.png", args.color)


def plot(payload: dict, out: Path, color: str = COLOUR) -> None:
    """One bare panel per gamma, laid out like barycenters_grid.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

    Y = np.asarray(payload["series"])
    runs = payload["runs"]
    t = np.arange(Y.shape[1])

    ncol = len(runs)
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 2.3), squeeze=False)
    for ax, r in zip(axes[0], runs):
        for y in Y:
            ax.plot(t, y, color="0.78", lw=0.8, zorder=1)
        ax.plot(t, np.asarray(r["barycenter"])[:, 0], color=color, lw=2.0,
                zorder=2)
        ax.set_title(rf"$\gamma = {r['gamma']:g}$", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
