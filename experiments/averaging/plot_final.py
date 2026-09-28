"""Final averaging figures from the ``final10`` JSONs -> results/figures/final10/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = RESULTS / "figures" / "final10"

DATASETS = [
    ("toy_accel", "averaging_toy_accel_n12_seed42_final10.json"),
    ("toy_ecg", "averaging_toy_ecg_n12_seed42_final10.json"),
    ("toy_rate", "averaging_toy_rate_n8_seed42_final10.json"),
]
OVERRIDES = {
    ("toy_ecg", "fastcdtw_inr"): "averaging_toy_ecg_n12_seed42_mc800_0.015.json",
}
SOFTDTW_SWEEP = {
    "toy_accel": "gammasweep_full/gammasweep_toy_accel_n12_seed42.json",
    "toy_ecg": "gammasweep_full/gammasweep_toy_ecg_n12_seed42.json",
    "toy_rate": "gammasweep_full/gammasweep_toy_rate_n8_seed42.json",
}

# key, label, colour
ROW_METHODS = [
    ("euclidean", "Euclidean Mean", "#4878CF"),
    ("dtw", "DTW", "#EE854A"),
    ("softdtw", "Soft-DTW", "#6ACC64"),
    ("fastcdtw_vec", "FastCDTW (vector, exact)", "#956CB4"),
    ("fastcdtw_inr", "FastCDTW (INR, MC)", "#D65F5F"),
]
EXTRA_METHODS = [("fastcdtw_vec_mc", "FastCDTW (vector, MC)", "#8C613C")]
ALL_METHODS = ROW_METHODS + EXTRA_METHODS


def style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def apply_overrides(ds: str, res: dict) -> dict:
    """Replace single method curves with the run named in OVERRIDES."""
    for (dataset, key), fname in OVERRIDES.items():
        if dataset != ds:
            continue
        src = load(fname)
        res["barycenters"][key] = src["barycenters"][key]
        dense = (src.get("barycenters_dense") or {}).get(key)
        if dense is not None:
            res.setdefault("barycenters_dense", {})[key] = dense
            res["t_dense"] = src["t_dense"]
        for field in ("objective_traces", "warpings", "monotone",
                      "fastcdtw_as_fitted", "timings"):
            if key in (src.get(field) or {}):
                res.setdefault(field, {})[key] = src[field][key]
        res.setdefault("overrides", {})[key] = fname
    return res


def apply_softdtw_gamma(ds: str, res: dict) -> dict:
    """Take the Soft-DTW curve from the gamma sweep, at its lowest DTW loss."""
    name = SOFTDTW_SWEEP.get(ds)
    if name is None:
        return res
    sweep = load(name)
    run = min(sweep["runs"], key=lambda r: r["dtw"])
    if not np.allclose(np.asarray(sweep["series"]), series_of(res)[:, :, 0]):
        raise SystemExit(f"{ds}: the sweep averaged a different draw of series")
    res["barycenters"]["softdtw"] = run["barycenter"]
    res.setdefault("overrides", {})["softdtw"] = f"{name} @ gamma={run['gamma']:g}"
    print(f"{ds:11s} softdtw  gamma={run['gamma']:<8g} DTW={run['dtw']:.4f}")
    return res


def series_of(res: dict) -> np.ndarray:
    """The series the run averaged, re-read from the ARFF."""
    from data import load as load_arff
    path = Path(res["data"]["file"])
    if not path.is_absolute():
        path = (HERE / path).resolve()
    X, _ = load_arff(path)
    return X[np.array(res["data"]["indices"])]


def curve_of(res: dict, key: str, T: int):
    """(t, x) for one method, dense when the run stored a dense read."""
    dense = (res.get("barycenters_dense") or {}).get(key)
    if dense is not None and res.get("t_dense"):
        return np.asarray(res["t_dense"]) * (T - 1), np.asarray(dense)[:, 0]
    return np.arange(T), np.asarray(res["barycenters"][key])[:, 0]


def bare(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel(ax, Y, curve, label, colour):
    t = np.arange(Y.shape[1])
    for y in Y:
        ax.plot(t, y[:, 0], color="0.78", lw=0.8, zorder=1)
    ax.plot(*curve, color=colour, lw=2.0, zorder=2)
    ax.set_title(label, fontsize=10)
    bare(ax)


def main():
    style()
    FIGS.mkdir(parents=True, exist_ok=True)
    data = {}
    for ds, fname in DATASETS:
        res = apply_softdtw_gamma(ds, apply_overrides(ds, load(fname)))
        data[ds] = (res, series_of(res))

    nrow, ncol = len(DATASETS), len(ROW_METHODS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.1 * nrow),
                             squeeze=False)
    for i, (ds, _) in enumerate(DATASETS):
        res, Y = data[ds]
        for j, (key, label, colour) in enumerate(ROW_METHODS):
            panel(axes[i][j], Y, curve_of(res, key, Y.shape[1]),
                  label if i == 0 else "", colour)
    fig.tight_layout()
    fig.savefig(FIGS / "barycenters_grid.png", dpi=200)
    fig.savefig(FIGS / "barycenters_grid.pdf")
    plt.close(fig)

    for ds, _ in DATASETS:
        res, Y = data[ds]
        fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 2.3), squeeze=False)
        for j, (key, label, colour) in enumerate(ROW_METHODS):
            panel(axes[0][j], Y, curve_of(res, key, Y.shape[1]), label, colour)
        fig.tight_layout()
        fig.savefig(FIGS / f"barycenters_{ds}.png", dpi=200)
        fig.savefig(FIGS / f"barycenters_{ds}.pdf")
        plt.close(fig)

    single = FIGS / "single"
    single.mkdir(exist_ok=True)
    for ds, _ in DATASETS:
        res, Y = data[ds]
        for key, label, colour in ALL_METHODS:
            if key not in res["barycenters"]:
                continue
            fig, ax = plt.subplots(figsize=(3.2, 2.6))
            panel(ax, Y, curve_of(res, key, Y.shape[1]), label, colour)
            fig.tight_layout()
            fig.savefig(single / f"{ds}_{key}.png", dpi=200)
            fig.savefig(single / f"{ds}_{key}.pdf")
            plt.close(fig)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.0 * len(DATASETS), 3.0),
                             squeeze=False)
    summary = {}
    for ax, (ds, _) in zip(axes[0], DATASETS):
        res, _ = data[ds]
        for key, label, colour in ALL_METHODS:
            trace = res["objective_traces"].get(key)
            if not trace:
                continue
            ax.plot(range(len(trace)), trace, color=colour, lw=1.4, label=label)
            tail = trace[-min(6, len(trace)):]
            summary.setdefault(ds, {})[key] = {
                "final": trace[-1],
                "rel_drop_last_3_outer": (tail[0] - trace[-1]) / abs(tail[0]),
                "monotone": res["monotone"][key]["ok"],
                "worst_increase": res["monotone"][key]["worst_increase"],
            }
        ax.set_yscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("half-step (E, M, E, M, ...)")
        ax.set_ylabel(r"$\sum_i w_i\,\mathrm{FM\text{-}DTW}(X, Y_i)$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "losses.png", dpi=200)
    fig.savefig(FIGS / "losses.pdf")
    plt.close(fig)

    (FIGS / "convergence.json").write_text(json.dumps(summary, indent=2))
    for ds, per in summary.items():
        for key, s in per.items():
            print(f"{ds:11s} {key:14s} final={s['final']:.6g}  "
                  f"tail drop={s['rel_drop_last_3_outer']:.2%}  "
                  f"monotone={s['monotone']}  worst +{s['worst_increase']:.2%}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
