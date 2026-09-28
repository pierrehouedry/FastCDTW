"""The INR barycenter against the Monte-Carlo budget; nothing else moves."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(HERE.parents[1] / "data" / "toy" / "toy_rate.arff"))
    p.add_argument("--n-series", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-mc", type=int, nargs="+",
                   default=[5, 10, 25, 50, 100, 200, 400])
    p.add_argument("--n-outer", type=int, default=30)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--lr-inr", type=float, default=1e-3)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--c-freq", type=int, default=10)
    p.add_argument("--c-hidden", type=int, default=64)
    p.add_argument("--warp-n", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--replot", default=None, help="redraw from a sweep JSON")
    p.add_argument("--color", default="#2C6FB5", help="hue of the light-to-dark ramp")
    args = p.parse_args()

    if args.replot:
        payload = json.loads(Path(args.replot).read_text())
        out = Path(args.replot).parent / "figures" / f"{Path(args.replot).stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot(payload, out, args.color)
        return

    from barycenters import fastcdtw_inr_barycenter
    from data import load, sample_series

    X, labels = load(args.data)
    Y, idx = sample_series(X, labels, n=args.n_series, seed=args.seed)
    dataset = Path(args.data).stem

    runs = []
    for n_mc in args.n_mc:
        t0 = time.time()
        C, info = fastcdtw_inr_barycenter(
            Y, lr=args.lr_inr, n_steps=args.n_steps, n_outer=args.n_outer,
            n_mc=n_mc, n_eval=args.n_eval, c_hidden=args.c_hidden,
            c_freq=args.c_freq, warp_n=args.warp_n, seed=args.seed,
            device=args.device, verbose=False)
        dt = time.time() - t0
        runs.append({"n_mc": n_mc, "barycenter": np.asarray(C).tolist(),
                     "objective": list(map(float, info["objective"])),
                     "phi": np.asarray(info["phi"]).tolist(),
                     "t_phi": np.asarray(info["t_phi"]).tolist(),
                     "monotone": bool(info["monotone"]), "time": dt})
        print(f"n_mc={n_mc:5d}  objective={info['objective'][-1]:.6f}  "
              f"{dt:6.1f} s", flush=True)

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    stem = f"mcsweep_{dataset}_n{args.n_series}_seed{args.seed}"
    payload = {"config": vars(args), "dataset": dataset,
               "indices": idx.tolist(), "series": Y[:, :, 0].tolist(),
               "runs": runs}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload))
    print(f"\n-> {out_dir / (stem + '.json')}")

    plot(payload, out_dir / "figures" / f"{stem}.png", args.color)


def plot(payload: dict, out: Path, color: str = "#D65F5F") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

    Y = np.asarray(payload["series"])
    runs = payload["runs"]
    n_mc = np.array([r["n_mc"] for r in runs], float)
    T = Y.shape[1]
    idx = np.arange(T)

    norm = colors.LogNorm(vmin=n_mc.min(), vmax=n_mc.max())
    base = np.array(colors.to_rgb(color))
    cmap = colors.LinearSegmentedColormap.from_list(
        "ramp", [1 - 0.25 * (1 - base), 0.45 * base])
    alphas = np.full(len(runs), 0.95)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for y in Y:
        ax.plot(idx, y, color="0.85", lw=0.8, zorder=1)
    for a, r in zip(alphas, runs):
        ax.plot(idx, np.asarray(r["barycenter"])[:, 0], lw=1.8, alpha=a,
                color=cmap(norm(r["n_mc"])), zorder=2)
    ax.set_xlabel(r"$t$")
    ax.grid(alpha=0.25)

    # zoom on the peak, where the curves actually differ
    peak = int(np.argmax(np.asarray(runs[-1]["barycenter"])[:, 0]))
    lo, hi = max(0, peak - 45), min(T, peak + 45)
    sub = ax.inset_axes([0.05, 0.30, 0.33, 0.50])
    for a, r in zip(alphas, runs):
        C = np.asarray(r["barycenter"])[:, 0]
        sub.plot(idx[lo:hi], C[lo:hi], lw=1.6, alpha=a,
                 color=cmap(norm(r["n_mc"])))
    sub.set_xlim(lo, hi)
    sub.tick_params(labelsize=7)
    sub.grid(alpha=0.25)
    sub.set_title("peak, zoomed", fontsize=8)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=r"MC draws per step $n_{\mathrm{mc}}$")

    ax = axes[1]
    obj = [r["objective"][-1] for r in runs]
    ax.plot(n_mc, obj, "o-", color="0.2", lw=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{mc}}$")
    ax.set_ylabel("objective")
    ax.grid(alpha=0.25, which="both")
    ax2 = ax.twinx()
    ax2.plot(n_mc, [r["time"] for r in runs], "s--", color=color, lw=1.0,
             alpha=0.8)
    ax2.set_ylabel("wall time (s)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
