from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from barycenters import (dtw_barycenter, euclidean_barycenter,
                         fastcdtw_inr_barycenter, fastcdtw_vec_mc_barycenter,
                         fastcdtw_vector_barycenter, softdtw_barycenter)
from data import DEFAULT_FILE, load, sample_series
from evaluate import cross_evaluate, format_table

HERE = Path(__file__).resolve().parent

METHODS = {
    "euclidean": ("Euclidean mean", "#4878CF"),
    "dtw":       ("DTW (DBA)", "#EE854A"),
    "softdtw":   ("Soft-DTW", "#6ACC64"),
    "fastcdtw_inr": ("FastCDTW (INR, MC)", "#D65F5F"),
    "fastcdtw_vec": ("FastCDTW (vector, exact)", "#956CB4"),
    "fastcdtw_vec_mc": ("FastCDTW (vector, MC)", "#8C613C"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--data", default=str(DEFAULT_FILE))
    d.add_argument("--n-series", type=int, default=8)
    d.add_argument("--seed", type=int, default=42, help="draw and solver seed")
    d.add_argument("--indices", type=int, nargs="+", default=None,
                   help="explicit rows; overrides the random draw")
    d.add_argument("--class", dest="klass", type=int, default=None,
                   help="restrict the draw to one gesture (1-8)")
    d.add_argument("--znorm", action="store_true")
    d.add_argument("--weights", type=float, nargs="+", default=None,
                   help="barycentric weights; two of them = interpolation")

    m = p.add_argument_group("methods")
    m.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    m.add_argument("--gamma", type=float, default=1.0, help="soft-DTW temperature")
    m.add_argument("--dba-iter", type=int, default=50)
    m.add_argument("--lbfgs-iter", type=int, default=200, help="soft-DTW iterations")

    o = p.add_argument_group("ours")
    o.add_argument("--n-outer", type=int, default=15, help="E/M alternations")
    o.add_argument("--n-steps", type=int, default=500, help="Adam steps per half-step")
    o.add_argument("--lr", type=float, default=1e-3, help="Adam lr, exact solver")
    o.add_argument("--lr-inr", type=float, default=5e-3, help="Adam lr, INR solver")
    o.add_argument("--n-mc", type=int, default=200, help="MC draws per step")
    o.add_argument("--n-eval", type=int, default=20, help="MC draws per reported objective")
    o.add_argument("--c-freq", type=int, default=8, help="INR Fourier frequencies")
    o.add_argument("--c-hidden", type=int, default=64, help="INR width")
    o.add_argument("--c-layers", type=int, default=5, help="INR depth")
    o.add_argument("--warp-n", type=int, default=0,
                   help="warping knots; 0 = one per timestamp")
    o.add_argument("--c-n", type=int, default=0,
                   help="centroid knots (exact solver); 0 = the series' grid")

    r = p.add_argument_group("run")
    r.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    r.add_argument("--out-dir", default=str(HERE / "results"))
    r.add_argument("--tag", default="")
    r.add_argument("--no-eval", action="store_true", help="skip the cross-evaluation")
    r.add_argument("--eval-steps", type=int, default=3000, help="refit budget in the table")
    r.add_argument("--overlay", action="store_true", help="extra panel, all barycenters")
    r.add_argument("--dense", type=int, default=2000,
                   help="points at which the INR centroid is read back; 0 = none")
    r.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


def plot_barycenters(Y, bary, out: Path, title: str, overlay: bool = False):
    import matplotlib.pyplot as plt

    names = list(bary)
    n_panels = len(names) + (1 if overlay and len(names) > 1 else 0)
    nrows, ncols = (1, n_panels) if n_panels <= 3 else (2, (n_panels + 1) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    t = np.arange(Y.shape[1])

    for ax, name in zip(flat, names):
        for y in Y:
            ax.plot(t, y[:, 0], color="0.75", lw=0.8, zorder=1)
        label, colour = METHODS[name]
        ax.plot(t, bary[name][:, 0], color=colour, lw=2.0, zorder=2)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)

    if overlay and len(names) > 1:
        ax = flat[len(names)]
        for y in Y:
            ax.plot(t, y[:, 0], color="0.85", lw=0.7, zorder=1)
        for name in names:
            label, colour = METHODS[name]
            ax.plot(t, bary[name][:, 0], color=colour, lw=1.4, label=label, zorder=2)
        ax.set_title("All barycenters", fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)

    for ax in flat[n_panels:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Time")
    for row in axes:
        row[0].set_ylabel("Amplitude")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_objectives(infos, out: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(infos), figsize=(4.6 * len(infos), 3.4), squeeze=False)
    for ax, (name, info) in zip(axes[0], infos.items()):
        obj = info["objective"]
        label, colour = METHODS[name]
        ax.plot(range(len(obj)), obj, marker="o", ms=3, color=colour)
        for x in range(2, len(obj), 2):
            ax.axvline(x, color="0.9", lw=0.6, zorder=0)
        ax.set_title(f"{label}\nmonotone={info['monotone']}  "
                     f"worst step +{info['worst_increase']:.2%}", fontsize=9)
        ax.set_xlabel("half-step (E, M, E, M, ...)")
        ax.set_ylabel(r"$\sum_i w_i\,$FastCDTW$(X, Y_i)$")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("FastCDTW objective along the alternating descent", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_warpings(infos, out: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(infos), figsize=(3.8 * len(infos), 3.4), squeeze=False)
    for ax, (name, info) in zip(axes[0], infos.items()):
        label, colour = METHODS[name]
        t = info["t_phi"]
        for phi in info["phi"]:
            ax.plot(t, phi, color=colour, lw=1.0, alpha=0.7)
        ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1.0, label="identity")
        dev = float(np.abs(info["phi"] - t[None, :]).max())
        ax.set_title(f"{label}\n" + rf"$\max_t |\varphi(t) - t| = {dev:.3f}$", fontsize=9)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\varphi(t)$")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle("Learned warpings", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    verbose = not args.quiet
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    X_all, labels = load(args.data)
    if args.indices:
        idx = np.array(sorted(args.indices))
        Y = X_all[idx]
    else:
        Y, idx = sample_series(X_all, labels, n=args.n_series, seed=args.seed,
                               klass=args.klass)
    if args.znorm:
        sd = Y.std(axis=1, keepdims=True)
        Y = (Y - Y.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)

    w = np.asarray(args.weights, float) if args.weights else None
    if w is not None and len(w) != len(Y):
        raise SystemExit(f"--weights has {len(w)} entries for {len(Y)} series")

    n, T, d = Y.shape
    dataset = Path(args.data).stem.removesuffix("_TRAIN").removesuffix("_TEST")
    stem = f"averaging_{dataset}_n{n}_seed{args.seed}"
    if args.klass is not None:
        stem += f"_class{args.klass}"
    if args.tag:
        stem += f"_{args.tag}"

    print(f"{dataset}: {X_all.shape[0]} series of length {X_all.shape[1]}")
    print(f"Averaging {n} series, rows {idx.tolist()}, classes {labels[idx].tolist()}")
    print(f"Weights: {'uniform' if w is None else w.tolist()}")

    bary, infos, timings = {}, {}, {}

    for name in args.methods:
        print(f"\n>> {METHODS[name][0]}")
        t0 = time.perf_counter()
        if name == "euclidean":
            bary[name] = euclidean_barycenter(Y, w)
        elif name == "dtw":
            bary[name] = dtw_barycenter(Y, weights=w, max_iter=args.dba_iter)
        elif name == "softdtw":
            bary[name] = softdtw_barycenter(Y, weights=w, gamma=args.gamma,
                                            max_iter=args.lbfgs_iter)
        elif name == "fastcdtw_inr":
            bary[name], infos[name] = fastcdtw_inr_barycenter(
                Y, weights=w, lr=args.lr_inr, n_steps=args.n_steps,
                n_outer=args.n_outer, n_mc=args.n_mc, n_eval=args.n_eval,
                c_hidden=args.c_hidden, c_freq=args.c_freq,
                c_layers=args.c_layers, warp_n=args.warp_n,
                seed=args.seed, device=args.device, verbose=verbose)
        elif name == "fastcdtw_vec":
            bary[name], infos[name] = fastcdtw_vector_barycenter(
                Y, weights=w, lr=args.lr, n_steps=args.n_steps,
                n_outer=args.n_outer, c_n=args.c_n, warp_n=args.warp_n,
                seed=args.seed, device=args.device, verbose=verbose)
        elif name == "fastcdtw_vec_mc":
            bary[name], infos[name] = fastcdtw_vec_mc_barycenter(
                Y, weights=w, lr=args.lr, n_steps=args.n_steps,
                n_outer=args.n_outer, n_mc=args.n_mc, n_eval=args.n_eval,
                c_n=args.c_n, warp_n=args.warp_n, seed=args.seed,
                device=args.device, verbose=verbose)
        timings[name] = time.perf_counter() - t0
        print(f"   done in {timings[name]:.1f}s")

    dense = {}       # the INR centroid is a function: read it off the grid too
    if args.dense:
        import torch
        t_dense = np.linspace(0.0, 1.0, args.dense)
        for name, info in infos.items():
            net = info.get("centroid")
            if net is None:
                continue
            with torch.no_grad():
                dense[name] = net(torch.as_tensor(t_dense, dtype=torch.float32,
                                                  device=args.device)).cpu().numpy()

    as_fitted = {k: v["objective"][-1] for k, v in infos.items()}
    table = None
    if not args.no_eval:
        print("\nCross-evaluation (each method minimises its own column):")
        table = cross_evaluate(bary, Y, weights=w, gamma=args.gamma,
                               device=args.device, fastcdtw_steps=args.eval_steps,
                               verbose=verbose)
        rendered = format_table(table, as_fitted, timings)
        print(rendered)
        (out_dir / f"{stem}_crosseval.txt").write_text(
            f"{dataset}  n={n}  seed={args.seed}  rows {idx.tolist()}\n"
            f"gamma={args.gamma:g}  refit budget={args.eval_steps} steps\n\n"
            + rendered + "\n")

    _style()
    fig_dir = out_dir / "figures"
    plot_barycenters(Y, bary, fig_dir / f"{stem}.png",
                     f"{dataset} -- barycenter of {n} series (seed {args.seed})",
                     overlay=args.overlay)
    if infos:
        plot_objectives(infos, fig_dir / f"{stem}_objective.png")
        plot_warpings(infos, fig_dir / f"{stem}_warpings.png")

    (out_dir / f"{stem}.json").write_text(json.dumps({
        "config": vars(args),
        "data": {"dataset": dataset, "file": str(args.data),
                 "indices": idx.tolist(), "classes": labels[idx].tolist(),
                 "shape": [n, T, d]},
        "timings": timings,
        "barycenters": {k: v.tolist() for k, v in bary.items()},
        "barycenters_dense": {k: v.tolist() for k, v in dense.items()},
        "t_dense": (np.linspace(0.0, 1.0, args.dense).tolist()
                    if args.dense and dense else None),
        "objective_traces": {k: v["objective"] for k, v in infos.items()},
        "warpings": {k: {"t": v["t_phi"].tolist(), "phi": v["phi"].tolist()}
                     for k, v in infos.items()},
        "fastcdtw_as_fitted": as_fitted,
        "monotone": {k: {"ok": v["monotone"], "worst_increase": v["worst_increase"]}
                     for k, v in infos.items()},
        "cross_evaluation": table,
    }, indent=2))
    print(f"\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
