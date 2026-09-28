"""Synthetic families: a template read on known clocks, with the truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[1] / "data" / "toy"

DENSE = 4001


def _gauss(t, mu, sigma, height):
    return height * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

def template_peak(t):
    z = _gauss(t, 0.60, 0.09, 1.0)
    z = np.where(t > 0.60, _gauss(t, 0.60, 0.045, 1.0), z)
    return z

def template_sharp(t):
    """template_peak with both widths halved: the narrow-peak families."""
    z = _gauss(t, 0.60, 0.045, 1.0)
    return np.where(t > 0.60, _gauss(t, 0.60, 0.022, 1.0), z)

def template_damped(t):
    """Silence, an onset, then a ring-down. Signed and multi-extremum."""
    t = np.asarray(t, dtype=float)
    u = np.clip(t - 0.28, 0.0, None)
    return np.where(t < 0.28, 0.0, np.exp(-5.0 * u) * np.sin(2 * np.pi * 5.0 * u))

def template_ecg(t):
    """P-QRS-T complex: three scales at once, and a negative lobe."""
    t = np.asarray(t, dtype=float)
    return (_gauss(t, 0.30, 0.032, 0.18)
            - _gauss(t, 0.445, 0.008, 0.12)
            + _gauss(t, 0.470, 0.009, 1.00)
            - _gauss(t, 0.500, 0.011, 0.28)
            + _gauss(t, 0.660, 0.048, 0.32))

def template_single(t, sigma: float = 0.09):
    return _gauss(t, 0.50, sigma, 1.0)

def template_pairbumps(t, sigma: float = 0.050):
    return _gauss(t, 0.38, sigma, 1.0) + _gauss(t, 0.62, sigma, 1.0)


def random_warp(rng, amp: float, n_freq: int = 3):
    """Unused by the current families; kept for new random-warp toys."""
    td = np.linspace(0.0, 1.0, DENSE)
    a = rng.normal(0.0, amp, size=n_freq)
    log_rate = sum(a[k] * np.sin((k + 1) * np.pi * td) for k in range(n_freq))
    rate = np.exp(log_rate)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(td))])
    w_vals = cum / cum[-1]

    def w(x):
        return np.interp(x, td, w_vals)

    def w_inv(x):
        return np.interp(x, w_vals, td)

    return w, w_inv

def sine_warp(a: float):
    def w(x):
        x = np.asarray(x, dtype=float)
        return x + a * np.sin(2 * np.pi * x) / (2 * np.pi)

    td = np.linspace(0.0, 1.0, DENSE)
    wd = w(td)

    def w_inv(x):
        return np.interp(x, wd, td)

    return w, w_inv


def rate_warp(speed: float):
    def w(x):
        return np.asarray(x) ** speed

    def w_inv(x):
        return np.asarray(x) ** (1.0 / speed)

    return w, w_inv

def build(family: str, T: int, n: int, sigma: float, amp: float, seed: int):
    """Returns (Y, labels, truth) with Y of shape (n, T)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    truth: dict = {"family": family, "T": T, "sigma": sigma, "amp": amp,
                   "seed": seed}

    if family == "rate":
        z = template_peak
        speeds = np.linspace(0.65, 1.55, n)
        warps = [rate_warp(s) for s in speeds]
        truth["speeds"] = speeds.tolist()
        truth["peaks"] = [{"pos": 0.60, "height": 1.00}]
    elif family == "sharp":
        z = template_sharp
        speeds = np.linspace(0.65, 1.55, n)
        warps = [rate_warp(s) for s in speeds]
        truth["speeds"] = speeds.tolist()
        truth["peaks"] = [{"pos": 0.60, "height": 1.00}]
    elif family == "accel":
        z = template_sharp
        amps = np.linspace(-amp, amp, n)
        warps = [sine_warp(a) for a in amps]
        truth["amps"] = amps.tolist()
        truth["peaks"] = [{"pos": 0.60, "height": 1.00}]
    elif family in ("damped", "ecg"):
        z = template_damped if family == "damped" else template_ecg
        amps = np.linspace(-amp, amp, n)
        warps = [sine_warp(a) for a in amps]
        truth["amps"] = amps.tolist()
    elif family in ("shift", "slide", "spread", "morph"):
        z = None
        warps = []
    else:
        raise ValueError(f"unknown family {family!r}")

    if family == "shift":
        Y = np.stack([_gauss(t, 0.25, 0.030, 1.0),
                      _gauss(t, 0.75, 0.030, 1.0)])
        truth["endpoints"] = [{"peaks": [{"pos": 0.25, "height": 1.0}]},
                              {"peaks": [{"pos": 0.75, "height": 1.0}]}]
        truth["expected"] = ("one peak of height 1 sliding from 0.75 to 0.25. "
                             "The endpoints are disjoint, so this is the hard "
                             "case: a linear blend gives two peaks of height pi "
                             "and 1 - pi and never a single one.")
        truth["template"] = None
    elif family == "morph":
        Y = np.stack([_gauss(t, 0.38, 0.055, 1.00),
                      _gauss(t, 0.60, 0.100, 0.55)])
        truth["endpoints"] = [{"peaks": [{"pos": 0.38, "height": 1.00}]},
                              {"peaks": [{"pos": 0.60, "height": 0.55}]}]
        truth["expected"] = ("one peak that both moves (0.38 -> 0.60) and "
                             "shrinks (1.0 -> 0.55) as pi falls; height and "
                             "position should be monotone in pi and the peak "
                             "count should stay 1. A linear blend gives two.")
        truth["template"] = None
    elif family in ("slide", "spread"):
        if family == "slide":
            z = template_single
            w = [rate_warp(0.66)[0], rate_warp(1.50)[0]]
            peaks = [0.50]
        else:
            z = template_pairbumps
            w = [sine_warp(-0.85)[0], sine_warp(0.85)[0]]
            peaks = [0.38, 0.62]
        Y = np.stack([z(w[0](t)), z(w[1](t))])
        truth["template"] = z(t).tolist()
        truth["template_peaks"] = peaks
        truth["endpoint_warps"] = [w[0](t).tolist(), w[1](t).tolist()]
        truth["endpoints"] = [
            {"peaks": [{"pos": float(t[np.argmin(np.abs(w[k](t) - q))],),
                        "height": 1.0} for q in peaks]}
            for k in (0, 1)]
        truth["expected"] = (
            "both endpoints are the same template read on a different clock, "
            "so the interpolant should be that template on the pi-blended "
            "clock: same peak height, peak positions moving with pi.")
    else:
        Y = np.empty((n, T))
        phis = []
        for i, (w, w_inv) in enumerate(warps):
            Y[i] = z(w(t))
            phis.append(w_inv(t).tolist())
        truth["template"] = z(t).tolist()
        truth["phi_true"] = [w(t).tolist() for w, _ in warps]
        truth["phi_true_inverse"] = phis

    Y = Y + rng.normal(0.0, sigma, size=Y.shape)
    truth["t"] = t.tolist()
    labels = np.ones(len(Y), dtype=int)
    return Y, labels, truth


def write_arff(Y: np.ndarray, labels: np.ndarray, path: Path, relation: str):
    T = Y.shape[1]
    lines = [f"@Relation {relation}", ""]
    lines += [f"@attribute att{j + 1} numeric" for j in range(T)]
    classes = sorted(set(int(c) for c in labels))
    lines.append("@attribute target {" + ",".join(str(c) for c in classes) + "}")
    lines += ["", "@data"]
    for row, lab in zip(Y, labels):
        lines.append(",".join(f"{v:.6f}" for v in row) + f",{int(lab)}")
    path.write_text("\n".join(lines) + "\n")


FAMILIES = {
    # family:   (n series, warp amplitude, noise)
    "rate":    dict(n=8,  amp=0.00, sigma=0.010),
    "sharp":   dict(n=16, amp=0.00, sigma=0.010),
    "accel":   dict(n=12, amp=0.90, sigma=0.010),
    "damped":  dict(n=12, amp=0.90, sigma=0.010),
    "ecg":     dict(n=12, amp=0.90, sigma=0.010),
    "shift":   dict(n=2,  amp=0.00, sigma=0.000),
    "slide":   dict(n=2,  amp=0.00, sigma=0.000),
    "morph":   dict(n=2,  amp=0.00, sigma=0.000),
    "spread":  dict(n=2,  amp=0.00, sigma=0.000),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", nargs="+", default=list(FAMILIES),
                   choices=list(FAMILIES))
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma", type=float, default=None,
                   help="override the per-family noise level")
    p.add_argument("--amp", type=float, default=None,
                   help="override the per-family warp amplitude")
    p.add_argument("--n", type=int, default=None,
                   help="override the per-family number of series")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for family in args.family:
        cfg = dict(FAMILIES[family])
        if args.sigma is not None:
            cfg["sigma"] = args.sigma
        if args.amp is not None:
            cfg["amp"] = args.amp
        if args.n is not None:
            cfg["n"] = args.n
        Y, labels, truth = build(family, T=args.T, seed=args.seed, **cfg)
        name = f"toy_{family}"
        write_arff(Y, labels, out / f"{name}.arff", name)
        (out / f"{name}_truth.json").write_text(json.dumps(truth))
        print(f"{name}: {Y.shape[0]} x {Y.shape[1]}  "
              f"sigma={cfg['sigma']:g} amp={cfg['amp']:g}  -> {out / (name + '.arff')}")


if __name__ == "__main__":
    main()
