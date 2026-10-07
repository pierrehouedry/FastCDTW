"""
The discrete barycenters against the INR one, on a coarse sample grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

METHODS = {
    "euclidean": ("Euclidean Mean", "#4878CF"),
    "dtw": ("DTW", "#EE854A"),
    "softdtw": ("Soft-DTW", "#6ACC64"),
    "fastcdtw_vec": ("FastCDTW (vector, PL)", "#956CB4"),
    "fastcdtw_inr": ("FastCDTW (INR, MC)", "#D65F5F"),
}
DISCRETE = ["euclidean", "dtw", "softdtw", "fastcdtw_vec"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(HERE.parents[1] / "data" / "toy" / "toy_rate.arff"))
    p.add_argument("--n-series", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=100, help="samples kept per series")
    p.add_argument("--dense", type=int, default=2000, help="points for the INR curve")
    p.add_argument("--n-outer", type=int, default=15)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--gamma", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                   help="Soft-DTW temperatures; the lowest DTW loss is kept, "
                        "as in the barycenter grid")
    p.add_argument("--c-freq", type=int, default=10)
    p.add_argument("--ms", type=float, default=2.5, help="marker size")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--replot", default=None, help="redraw from a saved npz")
    args = p.parse_args()

    if args.replot:
        f = Path(args.replot)
        d = np.load(f)
        bary = {k[5:]: d[k] for k in d.files if k.startswith("bary_")}
        out = f.parent / "figures" / f"{f.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot(d["t"], bary, d["t_dense"], d["C_dense"], out, args.ms)
        return

    import torch
    from barycenters import (dtw_barycenter, euclidean_barycenter,
                             fastcdtw_inr_barycenter,
                             fastcdtw_vector_barycenter, softdtw_barycenter)
    from data import load, sample_series
    from evaluate import dtw_objective

    X, labels = load(args.data)
    Y_full, idx = sample_series(X, labels, n=args.n_series, seed=args.seed)
    keep = np.unique(np.linspace(0, Y_full.shape[1] - 1, args.T).round().astype(int))
    Y = Y_full[:, keep, :]
    t = keep / (Y_full.shape[1] - 1)
    dataset = Path(args.data).stem
    print(f"{dataset}: {Y.shape[0]} series, {Y_full.shape[1]} -> {len(keep)} samples")
    best = None
    for g in args.gamma:
        C = np.asarray(softdtw_barycenter(Y, gamma=g, max_iter=200))
        o = dtw_objective(C, Y)
        print(f"  softdtw gamma={g:<8g} DTW={o:.4f}")
        if best is None or o < best[1]:
            best = (g, o, C)
    softdtw_gamma, softdtw_loss, C_softdtw = best
    print(f"  -> Soft-DTW at gamma={softdtw_gamma:g} (DTW={softdtw_loss:.4f})")

    bary = {"euclidean": euclidean_barycenter(Y),
            "dtw": dtw_barycenter(Y, max_iter=50),
            "softdtw": C_softdtw}
    bary["fastcdtw_vec"], _ = fastcdtw_vector_barycenter(
        Y, t_y=t, n_outer=args.n_outer, n_steps=args.n_steps, seed=args.seed,
        device=args.device, verbose=False)
    C_inr, info = fastcdtw_inr_barycenter(
        Y, t_y=t, n_outer=args.n_outer, n_steps=args.n_steps, n_mc=args.n_mc,
        c_freq=args.c_freq, seed=args.seed, device=args.device, verbose=False)
    bary["fastcdtw_inr"] = C_inr

    t_dense = np.linspace(0.0, 1.0, args.dense)
    with torch.no_grad():
        C_dense = info["centroid"](
            torch.as_tensor(t_dense, dtype=torch.float32,
                            device=args.device)).cpu().numpy()[:, 0]
    print(f"peak: {np.asarray(C_inr)[:, 0].max():.3f} on the grid, "
          f"{C_dense.max():.3f} read densely")

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    stem = f"inrcont_{dataset}_n{args.n_series}_T{len(keep)}_seed{args.seed}"
    flat = {f"bary_{k}": np.asarray(v)[:, 0] for k, v in bary.items()}
    plot(t, {k[5:]: v for k, v in flat.items()}, t_dense, C_dense,
         out_dir / "figures" / f"{stem}.png", args.ms)
    np.savez(out_dir / f"{stem}.npz", t=t, Y=Y[:, :, 0], t_dense=t_dense,
             C_dense=C_dense, indices=idx, softdtw_gamma=softdtw_gamma, **flat)
    print(f"-> {out_dir / (stem + '.npz')}")


def plot(t, bary, t_dense, C_dense, out: Path, ms: float = 2.5,
         ystep: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for name in DISCRETE:                 # vectors: no value between samples
        label, colour = METHODS[name]
        ax.plot(t, bary[name], "o", ms=ms, alpha=0.9, color=colour,
                label=label, zorder=2)
    label, colour = METHODS["fastcdtw_inr"]
    ax.plot(t_dense, C_dense, color=colour, lw=1.6, label=label, zorder=3)

    if ystep:                             # one y label every ystep
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ystep))
    ax.set_xlabel(r"$t$")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
