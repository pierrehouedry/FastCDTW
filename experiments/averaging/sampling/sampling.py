"""DTW, Soft-DTW and FastCDTW between two toy series, resampled at n timestamps on [0, 1]."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load
from fastcdtw import fm_dtw, phi_from_raw

METHODS = {"dtw": ("DTW", "#EE854A"),
           "softdtw": ("Soft-DTW", "#6ACC64"),
           "fastcdtw": ("FastCDTW", "#956CB4")}


DATA = ROOT / "data" / "toy"
# two series of one family: same template, two clocks
PAIRS = {"accel-accel": (("accel", 3), ("accel", 8)),
         "ecg-ecg": (("ecg", 3), ("ecg", 8)),
         "rate-rate": (("rate", 2), ("rate", 5))}
# best Soft-DTW gamma of the averaging experiment (results/gammasweep_full, lowest DTW loss)
GAMMA = {"accel-accel": 1e-2, "ecg-ecg": 1e-2, "rate-rate": 1e-3}


def series(family: str, k: int):
    """Series k of the toy set, read as a piecewise-linear function on [0, 1]."""
    X, _ = load(DATA / f"toy_{family}.arff")
    t = np.linspace(0.0, 1.0, X.shape[1])
    return lambda s, v=X[k, :, 0]: np.interp(s, t, v)


def timestamps(n: int, grid: str, rng) -> np.ndarray:
    if grid == "linspace":
        return np.linspace(0.0, 1.0, n)
    # iid uniform, endpoints kept so both interpolants cover [0, 1]
    return np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0.0, 1.0, n - 2)]))


def dtw_value(x, y) -> float:
    from tslearn.metrics import dtw
    return float(dtw(x[:, None], y[:, None]) ** 2)  # tslearn returns sqrt(sum sq)


def softdtw_value(x, y, gamma) -> float:
    from tslearn.metrics import soft_dtw
    return float(soft_dtw(x[:, None], y[:, None], gamma=gamma))


def _fit(x, t_x, y, t_y, n_knots, n_steps, lr) -> tuple[float, float]:
    f64 = dict(dtype=torch.float64)
    X = torch.tensor(x, **f64)[None, :, None]
    Y = torch.tensor(y, **f64)[None, :, None]
    tx, ty = torch.tensor(t_x, **f64), torch.tensor(t_y, **f64)
    t_phi = torch.linspace(0.0, 1.0, n_knots, **f64)
    raw = torch.zeros(1, n_knots - 1, requires_grad=True, **f64)
    idx = torch.zeros(1, dtype=torch.long)

    def loss():
        return fm_dtw(phi_from_raw(raw), t_phi, X, tx, Y, ty, idx, idx).sum()

    opt = torch.optim.Adam([raw], lr=lr)
    tail = None
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        L = loss()
        if step == n_steps - n_steps // 10:
            tail = L.item()
        L.backward()
        opt.step()
    with torch.no_grad():
        final = loss().item()
    return final, (0.0 if tail is None else (tail - final) / max(abs(final), 1e-12))


def fastcdtw_identity(x, t_x, y, t_y, n_knots) -> float:
    """FastCDTW at phi = id: the unaligned cost, used as the scale of the problem."""
    return _fit(x, t_x, y, t_y, n_knots, 0, 0.0)[0]


def fastcdtw_value(x, t_x, y, t_y, n_knots, n_steps, lrs) -> dict:
    """min_phi FastCDTW, best of one cold fit per lr. phi's grid ignores the sampling."""
    fits = {lr: _fit(x, t_x, y, t_y, n_knots, n_steps, lr) for lr in lrs}
    lr = min(fits, key=lambda k: fits[k][0])
    return {"fastcdtw": fits[lr][0], "fastcdtw_tail_drop": fits[lr][1],
            "fastcdtw_lr": lr, "fastcdtw_all": {str(k): v[0] for k, v in fits.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    p.add_argument("--n", type=int, nargs="+",
                   default=[100, 200, 500, 1000, 2000, 5000])
    p.add_argument("--grid", choices=["linspace", "random"], default="linspace")
    p.add_argument("--n-rep", type=int, default=5, help="draws per n (random grid only)")
    p.add_argument("--gamma", type=float, default=None,
                   help="same gamma for every pair (default: GAMMA)")
    p.add_argument("--ref-n", type=int, default=200,
                   help="FastCDTW reference on a linspace grid of this size (200 = native)")
    p.add_argument("--n-knots", type=int, default=201,
                   help="knots of phi; must differ from every n (equal grids stall the fit)")
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, nargs="+", default=[5e-3, 1e-2, 5e-2],
                   help="one cold fit per lr, the lowest value is kept")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--replot", default=None, help="redraw from a results JSON")
    p.add_argument("--min-n", type=int, default=100,
                   help="figure and table only use min_n <= n <= max_n (the JSON keeps every n)")
    p.add_argument("--max-n", type=int, default=5000)
    args = p.parse_args()

    if args.replot:
        f = Path(args.replot)
        res = trim(json.loads(f.read_text()), args.min_n, args.max_n)
        table(res, f.with_name(f"{f.stem}_table.txt"))
        plot(res, f.parent / "figures" / f"{f.stem}.png")
        return

    rng = np.random.default_rng(args.seed)
    n_rep = 1 if args.grid == "linspace" else args.n_rep
    out = {"config": vars(args), "pairs": {}}

    for fam in args.pairs:
        fx, fy = (series(*s) for s in PAIRS[fam])
        ij = PAIRS[fam]
        kw = dict(n_knots=args.n_knots, n_steps=args.n_steps, lrs=args.lr)

        gamma = args.gamma if args.gamma is not None else GAMMA[fam]
        t = np.linspace(0.0, 1.0, args.ref_n)
        ref = {"dtw": dtw_value(fx(t), fy(t)), "softdtw": softdtw_value(fx(t), fy(t), gamma),
               "fastcdtw": fastcdtw_value(fx(t), t, fy(t), t, **kw)["fastcdtw"],
               # unaligned costs: identity path for DTW / Soft-DTW, phi = id for FastCDTW
               "unaligned_dtw": float(((fx(t) - fy(t)) ** 2).sum()),
               "unaligned_fastcdtw": fastcdtw_identity(fx(t), t, fy(t), t, args.n_knots)}
        print(f"{fam} {ij}: gamma={gamma:g}  ref n={args.ref_n}: "
              + "  ".join(f"{k}={v:.4e}" for k, v in ref.items()), flush=True)

        rows = []
        for n in args.n:
            for r in range(n_rep):
                t_x, t_y = timestamps(n, args.grid, rng), timestamps(n, args.grid, rng)
                x, y = fx(t_x), fy(t_y)
                t0 = time.time()
                fc = fastcdtw_value(x, t_x, y, t_y, **kw)
                rows.append({"n": n, "rep": r, "dtw": dtw_value(x, y),
                             "softdtw": softdtw_value(x, y, gamma), **fc,
                             "time_fastcdtw": time.time() - t0})
                print(f"  n={n:<5d} rep={r}  DTW={rows[-1]['dtw']:.4e}  "
                      f"SDTW={rows[-1]['softdtw']:.4e}  FastCDTW={fc['fastcdtw']:.4e}  "
                      f"(lr {fc['fastcdtw_lr']:g}, tail drop {fc['fastcdtw_tail_drop']:.1e}, "
                      f"{time.time() - t0:.0f} s)", flush=True)
        out["pairs"][fam] = {"series": [list(s) for s in ij], "gamma": gamma,
                                "ref": ref, "rows": rows}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"sampling_{args.grid}_seed{args.seed}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"-> {f}")
    out = trim(out, args.min_n, args.max_n)
    table(out, f.with_name(f"{f.stem}_table.txt"))
    plot(out, out_dir / "figures" / f"{f.stem}.png")


def trim(res: dict, min_n: int, max_n: int) -> dict:
    """Copy of res keeping only the rows with min_n <= n <= max_n."""
    res = json.loads(json.dumps(res))
    for info in res["pairs"].values():
        info["rows"] = [r for r in info["rows"] if min_n <= r["n"] <= max_n]
    return res


def drift(info: dict, m: str) -> np.ndarray:
    """(value - value at ref_n) / unaligned cost at ref_n, one entry per row."""
    ref = info["ref"]
    scale = ref["unaligned_fastcdtw" if m == "fastcdtw" else "unaligned_dtw"]
    return (np.array([r[m] for r in info["rows"]]) - ref[m]) / scale


def table(res: dict, path: Path) -> str:
    """min / max / mean / std of the normalised drift over every n (and draw)."""
    lines = [f"{'pair':<12}{'method':<10}{'min':>11}{'max':>11}{'mean':>11}{'std':>11}"]
    for name, info in res["pairs"].items():
        for m, (label, _) in METHODS.items():
            d = drift(info, m)
            lines.append(f"{name:<12}{label:<10}" + "".join(
                f"{v:>11.2e}" for v in (d.min(), d.max(), d.mean(), d.std())))
    txt = "\n".join(lines)
    path.write_text(txt + "\n")
    print(txt)
    print(f"-> {path}")
    return txt


def plot(res: dict, path: Path) -> None:
    """One panel per pair, all methods on one axis: (value - value at ref_n) / unaligned
    cost at ref_n, i.e. the drift as a fraction of the distance between the two series
    before alignment. 0 = unchanged; symlog y so small and large drifts both show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                                "font.serif": ["CMU Serif", "DejaVu Serif"]})

    pairs = list(res["pairs"])
    ref_n = res["config"].get("ref_n", 200)
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 2.8), squeeze=False)
    for ax, name in zip(axes[0], pairs):
        info = res["pairs"][name]
        rows = info["rows"]
        ns = sorted({r["n"] for r in rows})
        n_of = np.array([r["n"] for r in rows])
        for m, (label, col) in METHODS.items():
            d = drift(info, m)
            v = [d[n_of == n] for n in ns]
            mu, sd = np.array([a.mean() for a in v]), np.array([a.std() for a in v])
            if m == "softdtw":
                label += rf" ($\gamma$={info['gamma']:g})"
            ax.plot(ns, mu, "o-", color=col, ms=3, lw=1.5, label=label)
            ax.fill_between(ns, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.axvline(ref_n, color="0.8", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(name)
        ax.set_xlabel("n samples")
    axes[0, 0].set_ylabel("drift / unaligned cost")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
